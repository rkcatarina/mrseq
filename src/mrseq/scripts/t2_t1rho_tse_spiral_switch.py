"""Spiral turbo-spin echo sequence with with switchable continuous-wave (CW) or adiabatic spin-lock preparation."""

from pathlib import Path
from typing import Literal

import ismrmrd
import numpy as np
import pypulseq as pp

from mrseq.preparations import add_t1rho_prep, add_t1rho_adiabatic_prep
from mrseq.utils import round_to_raster
from mrseq.utils import spiral_acquisition
from mrseq.utils import sys_defaults
from mrseq.utils import write_sequence
from mrseq.utils.constants import GOLDEN_ANGLE_HALF_CIRCLE
from mrseq.utils.ismrmrd import Fov
from mrseq.utils.ismrmrd import Limits
from mrseq.utils.ismrmrd import MatrixSize
from mrseq.utils.ismrmrd import create_header


def t2_t1rho_tse_spiral_prep_switch_kernel(
    system: pp.Opts,
    prep_type: Literal['cw', 'adiabatic'],
    spin_lock_times: np.ndarray,
    te: float | None,
    n_echoes: int,
    tr: float,
    fov_xy: float,
    fov_z: float,
    spiral_type: Literal['out', 'in-out'],
    n_readout: int,
    readout_oversampling: Literal[1, 2, 4],
    spiral_undersampling: float,
    n_slice_encoding: int,
    gx_pre_duration: float,
    rf_ex_duration: float,
    rf_ex_bwt: float,
    rf_ref_duration: float,
    rf_ref_bwt: float,
    rf_ref_width_scale_factor: float,
    gz_crusher_duration: float,
    gz_crusher_area: float,
    rf_spin_lock_duration: float,
    spin_lock_amplitude: float,
    adiabatic_rf_duration: float,
    adiabatic_rf_amplitude: float,
    adiabatic_pulse_gap: float,
    add_spin_lock_spoiler: bool,
    spin_lock_spoiler_ramp_time: float,
    spin_lock_spoiler_flat_time: float,
    mrd_header_file: str | Path | None,
) -> tuple[pp.Sequence, float]:
    """Generate a spiral TSE-based sequence for T1rho mapping with one preparation pulse before every readout.

    Parameters
    ----------
    system
        PyPulseq system limits object.
    prep_type
        Type of T1rho preparation:
        'cw' - on-resonant continuous-wave spin lock (90° tip-down, spin-lock pulse, 90° tip-up).
        'adiabatic' - pairs of hyperbolic secant adiabatic full passage pulses.
    spin_lock_times
        Array of spin lock times (in seconds). For prep_type='adiabatic', each spin lock time must be a multiple of
        2 * adiabatic_rf_duration (= duration of one HS pulse pair).
    te
        Desired echo time (TE) in seconds. Minimum echo time is used if set to None.
    n_echoes
        Number of echoes in one TSE train.
    tr
        Desired repetition time (TR) in seconds.
    fov_xy
        Field of view in the x and y directions in meters.
    fov_z
        Field of view in the z direction (slice thickness) in meters.
    spiral_type
        Type of spiral trajectory.
    n_readout
        Number of frequency encoding steps.
    readout_oversampling
        Readout oversampling. Determines the number of ADC samples along a spiral and the bandwidth.
    spiral_undersampling
        Angular undersampling of the spiral trajectoy.
    n_slice_encoding
        Number of slice encoding steps.
    gx_pre_duration
        Duration of the readout pre-winder gradient in seconds..
    rf_ex_duration
        Duration of the excitation RF pulse in seconds.
    rf_ex_bwt
        Bandwidth-time product of the excitation RF pulse.
    rf_ref_duration
        Duration of the refocusing RF pulse in seconds.
    rf_ref_bwt
        Bandwidth-time product of the refocusing RF pulse.
    rf_ref_width_scale_factor
        Factor to scale the slice thickness of the refocusing pulse.
    readout_oversampling
        Readout oversampling.
    gz_crusher_duration
        Duration of the crusher gradients applied around the 180° pulse.
    gz_crusher_area : float
        Area (zeroth gradient moment) of the crusher gradients applied around the 180° pulse.
    rf_spin_lock_duration
        Duration of 90° tip-down/tip-up pulses for spin lock preparation (in seconds).
    spin_lock_amplitude
        Amplitude of the CW spin-lock pulse (in T). Only used for prep_type='cw'.
    adiabatic_rf_duration
        Duration of each HS pulse (in seconds). Only used for prep_type='adiabatic'.
    adiabatic_rf_amplitude
        Peak B1 amplitude of the HS pulses (in T). Only used for prep_type='adiabatic'.
    adiabatic_pulse_gap
        Delay between the two HS pulses of each pair (in seconds). Only used for prep_type='adiabatic'.
    add_spin_lock_spoiler
        Toggles addition of spoiler gradients after the spin lock pulse.
    spin_lock_spoiler_ramp_time
        Duration of gradient spoiler ramps (in seconds).
    spin_lock_spoiler_flat_time
        Duration of gradient spoiler plateau (in seconds).
    mrd_header_file
        Filename of the ISMRMRD header file to be created. If None, no header file is created.

    Returns
    -------
    seq
        PyPulseq Sequence object
    time_to_first_tr_block
        End point of first TR block.
    min_te
        Shortest possible echo time.
    
    Raises
    ------
    ValueError
        If prep_type is unknown or a spin lock time is not achievable with adiabatic HS pulse pairs.
    """
    # translate spin lock times into number of HS pulse pairs for the adiabatic preparation
    if prep_type == 'adiabatic':
        n_pairs_float = spin_lock_times / (2 * adiabatic_rf_duration)
        n_pairs = np.round(n_pairs_float).astype(int)
        if not np.allclose(n_pairs_float, n_pairs, rtol=0, atol=1e-6):
            raise ValueError(
                f'For adiabatic T1rho preparation, spin lock times must be multiples of '
                f'{2 * adiabatic_rf_duration * 1000:.2f} ms (duration of one HS pulse pair).'
            )
    elif prep_type != 'cw':
        raise ValueError(f"Unknown prep_type '{prep_type}'. Must be 'cw' or 'adiabatic'.")
    
    # create PyPulseq Sequence object and set system limits
    print(system.rf_ringdown_time)
    print(system.B0)
    print(spiral_type)
    seq = pp.Sequence(system=system)

    # create slice selective excitation pulse and gradient
    rf_ex, gz_ex, gzr_ex = pp.make_sinc_pulse(
        flip_angle=90 / 180 * np.pi,
        duration=rf_ex_duration,
        slice_thickness=fov_z,
        apodization=0.5,
        phase_offset=np.pi / 2,
        time_bw_product=rf_ex_bwt,
        delay=system.rf_dead_time,  # delay should equal at least the dead time of the RF pulse
        system=system,
        return_gz=True,
        use='excitation',
    )

    # create slice selective refocusing pulse and gradient
    rf_ref, gz_ref, _ = pp.make_sinc_pulse(
        flip_angle=np.pi,
        duration=rf_ref_duration,
        slice_thickness=fov_z * rf_ref_width_scale_factor,
        apodization=0.5,
        phase_offset=0,
        time_bw_product=rf_ref_bwt,
        delay=system.rf_dead_time,  # delay should equal at least the dead time of the RF pulse
        system=system,
        return_gz=True,
        use='refocusing',
    )
    # create readout gradient and ADC
    gx, gy, adc, trajectory, time_to_echo, max_grad_duration = spiral_acquisition(
        system,
        n_readout,
        fov_xy,
        spiral_undersampling,
        readout_oversampling=readout_oversampling,
        n_spirals=None,
        max_pre_duration=gx_pre_duration,
        spiral_type=spiral_type,
    )
    
   
    delta_array = np.pi / len(gx) * np.arange(len(gx))  # angle difference between subsequent spirals

    # phase encoding along slice direction
    gz_areas = (np.arange(n_slice_encoding) - n_slice_encoding // 2) * 1 / fov_z

    # create crusher gradients
    gz_crush = pp.make_trapezoid(channel='z', system=system, area=gz_crusher_area, duration=gz_crusher_duration)

    # calculate minimum delays
    # tau1: between excitation pulse and first refocusing pulse
    min_tau1 = rf_ex.shape_dur / 2
    min_tau1 += max(rf_ex.ringdown_time, gz_ex.fall_time)
    min_tau1 += pp.calc_duration(gzr_ex)
    min_tau1 += gz_crusher_duration
    min_tau1 += max(rf_ref.delay, gz_ref.delay + gz_ref.rise_time)
    min_tau1 += rf_ref.shape_dur / 2

    # tau2: between refocusing pulses and readout
    min_tau2 = rf_ref.shape_dur / 2
    min_tau2 += max(rf_ref.ringdown_time, gz_ref.fall_time)
    min_tau2 += gz_crusher_duration
    min_tau2 += time_to_echo

    # tau3: between readout and next refocusing pulse
       
    min_tau3 = time_to_echo
    min_tau3 += max_grad_duration
    min_tau3 += gz_crusher_duration
    min_tau3 += max(rf_ref.delay, gz_ref.delay + gz_ref.rise_time)
    min_tau3 += rf_ref.shape_dur / 2

    min_te = 2 * max(
        round_to_raster(min_tau1, system.block_duration_raster),
        round_to_raster(min_tau2, system.block_duration_raster),
        round_to_raster(min_tau3, system.block_duration_raster),
    )

    # calculate echo time delay (te_delay)
    te = min_te if te is None else round_to_raster(te, system.block_duration_raster)
    if te < min_te:
        raise ValueError(f'TE must be larger than {min_te * 1000:.3f} ms. Current value is {te * 1000:.3f} ms.')

    tau1 = round_to_raster(te / 2 - min_tau1, raster_time=system.grad_raster_time)
    tau2 = round_to_raster(te / 2 - min_tau2, raster_time=system.grad_raster_time)
    tau3 = round_to_raster(te / 2 - min_tau3, raster_time=system.grad_raster_time)
    print(f'\nCurrent echo time = {(te) * 1000:.3f} ms')

    # create header
    if mrd_header_file:
        hdr = create_header(
            traj_type='other',
            encoding_fov=Fov(x=fov_xy, y=fov_xy, z=fov_z),
            recon_fov=Fov(x=fov_xy, y=fov_xy, z=fov_z),
            encoding_matrix=MatrixSize(n_x=int(n_readout), n_y=int(n_readout), n_z=n_slice_encoding),
            recon_matrix=MatrixSize(n_x=n_readout, n_y=n_readout, n_z=n_slice_encoding),
            dwell_time=adc.dwell,
            k1_limits=Limits(min=0, max=len(gx), center=0),
            k2_limits=Limits(min=0, max=n_slice_encoding, center=n_slice_encoding // 2),
            slice_limits=Limits(),
        )

        # write header to file
        prot = ismrmrd.Dataset(mrd_header_file, 'w')
        prot.write_xml_header(hdr.toXML('utf-8'))

    # obtain noise samples
    seq.add_block(pp.make_label(label='LIN', type='SET', value=0), pp.make_label(label='SLC', type='SET', value=0))
    seq.add_block(adc, pp.make_label(label='NOISE', type='SET', value=True))
    seq.add_block(pp.make_label(label='NOISE', type='SET', value=False))
    seq.add_block(pp.make_delay(system.rf_dead_time))

    if mrd_header_file:
        acq = ismrmrd.Acquisition()
        acq.resize(trajectory_dimensions=3, number_of_samples=adc.num_samples)
        prot.append_acquisition(acq)

    # add all events to the sequence
    for tsl_idx, tsl in enumerate(spin_lock_times):
        for se in range(n_slice_encoding):
            se_label = pp.make_label(type='SET', label='PAR', value=int(se))

            # phase encoding along se
            gz_rew = pp.make_trapezoid(
                channel='z',
                area=-gz_areas[se],
                duration=gx_pre_duration,
                system=system,
            )

            # Combine phase encoding along se and crusher along z-direction
            gz_crush_pre = pp.make_trapezoid(
                channel='z', system=system, area=gz_crusher_area + gz_areas[se], duration=gz_crusher_duration
            )
            gz_crush_rew = pp.make_trapezoid(
                channel='z', system=system, area=gz_crusher_area - gz_areas[se], duration=gz_crusher_duration
            )

            for spiral_ in range(len(gx)):
                pe_label = pp.make_label(type='SET', label='LIN', value=spiral_)

                _start_time_tr_block = sum(seq.block_durations.values())

                # add T1rho preparation block
                if tsl > 0 and prep_type == 'cw':
                    seq, _ = add_t1rho_prep(
                        seq=seq,
                        system=system,
                        duration_90=rf_spin_lock_duration,
                        spin_lock_time=tsl,
                        spin_lock_amplitude=spin_lock_amplitude,
                        add_spoiler=add_spin_lock_spoiler,
                        spoiler_ramp_time=spin_lock_spoiler_ramp_time,
                        spoiler_flat_time=spin_lock_spoiler_flat_time,
                    )
                elif tsl > 0 and prep_type == 'adiabatic':
                    seq, _, _ = add_t1rho_adiabatic_prep(
                        seq=seq,
                        system=system,
                        n_pairs=int(n_pairs[tsl_idx]),
                        rf_duration=adiabatic_rf_duration,
                        rf_amplitude=adiabatic_rf_amplitude,
                        pulse_gap=adiabatic_pulse_gap,
                        add_spoiler=add_spin_lock_spoiler,
                        spoiler_ramp_time=spin_lock_spoiler_ramp_time,
                        spoiler_flat_time=spin_lock_spoiler_flat_time,
                    )

                # add excitation pulse
                seq.add_block(rf_ex, gz_ex)
                seq.add_block(gzr_ex)
                seq.add_block(pp.make_delay(tau1))

                for echo in range(n_echoes):
                    contrast_label = pp.make_label(type='SET', label='ECO', value=int(echo + tsl_idx * n_echoes))

                    # calculate theoretical golden angle rotation for current shot
                    angle_idx = spiral_ + echo * len(gx) + tsl_idx * len(gx) * n_echoes
                    golden_angle = np.mod(GOLDEN_ANGLE_HALF_CIRCLE * angle_idx, np.pi)

                    # find closest unique spiral to current golden angle rotation
                    diff = np.abs(delta_array - golden_angle)
                    spiral_idx = np.argmin(diff)

                    # add refocusing pulse with crusher gradients
                    seq.add_block(gz_crush if echo == 0 else gz_crush_rew)
                    seq.add_block(rf_ref, gz_ref)
                    seq.add_block(gz_crush_pre)

                    seq.add_block(pp.make_delay(tau2))

                    # add pre gradients and all labels
                    labels = [se_label, pe_label, contrast_label]
                    seq.add_block(gx[spiral_idx], gy[spiral_idx], adc, *labels)

                    if echo < n_echoes - 1:
                        seq.add_block(pp.make_delay(tau3))

                    if mrd_header_file:
                        # add acquisitions to metadata
                        spiral_trajectory = np.zeros((trajectory.shape[1], 3), dtype=np.float32)

                        # the spiral trajectory is calculated in units of delta_k.
                        # For image reconstruction we use delta_k = 1
                        spiral_trajectory[:, 0] = trajectory[spiral_idx, :, 0] * fov_xy
                        spiral_trajectory[:, 1] = trajectory[spiral_idx, :, 1] * fov_xy
                        spiral_trajectory[:, 2] = se - n_slice_encoding // 2

                        acq = ismrmrd.Acquisition()
                        acq.resize(trajectory_dimensions=3, number_of_samples=adc.num_samples)
                        acq.traj[:] = spiral_trajectory
                        prot.append_acquisition(acq)

                # Add final rewinder
                seq.add_block(gz_rew)

                duration_tr_block = sum(seq.block_durations.values()) - _start_time_tr_block
                tr_delay = round_to_raster(tr - duration_tr_block, system.block_duration_raster)
                
                # save time for sequence plot
                if tsl_idx == 0 and spiral_ == 0:
                    time_to_first_tr_block = duration_tr_block
                if tr_delay < 0:
                    raise ValueError('Desired TR too short for given sequence parameters.')
                seq.add_block(pp.make_delay(tr_delay))

    return seq, time_to_first_tr_block,min_te


def main(
    system: pp.Opts | None = None,
    prep_type: Literal['cw', 'adiabatic'] = 'cw',
    spin_lock_times: np.ndarray | None = None,
    te: float | None = None,
    n_echoes: int = 10,
    tr: float = 2,
    fov_xy: float = 200e-3,
    fov_z: float = 8e-3,
    spiral_type: Literal['out', 'in-out'] = 'out',
    n_readout: int = 200,
    readout_oversampling: Literal[1, 2, 4] = 2,
    n_spiral_arms: int = 50,
    n_slice_encoding: int = 1,
    show_plots: bool = True,
    test_report: bool = True,
    timing_check: bool = True,
    v141_compatibility: bool = True,
) -> tuple[pp.Sequence, Path]:
    """Generate a spiral TSE-based sequence for T1rho mapping with one preparation pulse before every readout.

    Parameters
    ----------
    system
        PyPulseq system limits object.
    prep_type
        Type of T1rho preparation:
        'cw' - on-resonant continuous-wave spin lock (90° tip-down, spin-lock pulse, 90° tip-up).
        'adiabatic' - pairs of hyperbolic secant adiabatic full passage pulses.
    spin_lock_times
        Array of spin lock times (in seconds).
        If set to None, default values are [0.025, 0.050, 0.1] s for prep_type='cw' and
        [0.03, 0.06, 0.12] s (= 1, 2 and 4 HS pulse pairs of 15 ms pulses) for prep_type='adiabatic'
    te
        Desired echo time (TE) (in seconds). Minimum echo time is used if set to None.
    n_echoes
        Number of echoes in one TSE train.
    tr
        Desired repetition time (TR) (in seconds).
    fov_xy
        Field of view in x and y direction (in meters).
    fov_z
        Field of view along z (in meters).
    spiral_type 
        Type of spiral trajectory. 
    n_readout
        Number of frequency encoding steps.
    readout_oversampling
        Readout oversampling. Determines the number of ADC samples along a spiral and the bandwidth.
    n_spiral_arms
        Number of spiral arms.
    n_slice_encoding
        Number of phase encoding steps along the slice direction.
    show_plots
        Toggles sequence plot.
    test_report
        Toggles advanced test report.
    timing_check
        Toggles timing check of the sequence.
    v141_compatibility
        Save the sequence in pulseq v1.4.1 for backwards compatibility.
    Raises
    ------
    ValueError
        If prep_type is unknown.
    """
    if system is None:
        system = sys_defaults

    if prep_type not in ('cw', 'adiabatic'):
        raise ValueError(f"Unknown prep_type '{prep_type}'. Must be 'cw' or 'adiabatic'.")
    
  
    if spin_lock_times is None:
        spin_lock_times = (0,) if prep_type == 'cw' else np.array([0.03, 0.06, 0.120])
    

    # define settings of CW spin-lockpreparation pulse
    rf_spin_lock_duration = 2e-3
    spin_lock_amplitude = 4e-6

    # define settings of adiabatic spin-lock preparation (HS pulse pairs)
    adiabatic_rf_duration = 15e-3
    adiabatic_rf_amplitude = 11.7e-6
    adiabatic_pulse_gap = 350e-6

    # define spoiler settings (shared by both preparation types)
    add_spin_lock_spoiler = True
    spin_lock_spoiler_ramp_time = 6e-4
    spin_lock_spoiler_flat_time = 8.4e-3

    # define ADC and gradient timing
    gx_pre_duration = 1.81e-3  # duration of readout pre-winder gradient [s]

    gz_crusher_duration = 1.95e-3  # duration of crusher gradients [s]
    gz_crusher_area = 4 / (fov_z / n_slice_encoding)

    # define settings of rf excitation pulse
    rf_ex_duration = 2e-3  # duration of the rf excitation pulse [s]
    rf_ex_bwt = 4  # bandwidth-time product of rf excitation pulse [Hz*s]

    rf_ref_width_scale_factor = 3.5  # width of refocusing pulse is increased compared to excitation pulse

    # define sequence filename
    filename = (
        f'{Path(__file__).stem}_{prep_type}_{spiral_type}_{int(fov_xy * 1000)}fov_{n_readout}nx_{n_spiral_arms}ny_'
        f'{len(spin_lock_times)}TSLs_{spin_lock_amplitude*1000000}splAmp'
    )


    output_path = Path.cwd() / 'output'
    output_path.mkdir(parents=True, exist_ok=True)

    # delete existing header file
    if (output_path / Path(filename + '_header.h5')).exists():
        (output_path / Path(filename + '_header.h5')).unlink()

    seq, time_to_first_tr_block,min_te = t2_t1rho_tse_spiral_prep_switch_kernel(
        system=system,
        prep_type=prep_type,
        spin_lock_times=spin_lock_times,
        te=te,
        n_echoes=n_echoes,
        tr=tr,
        fov_xy=fov_xy,
        fov_z=fov_z,
        spiral_type=spiral_type,
        n_readout=n_readout,
        readout_oversampling=readout_oversampling,
        spiral_undersampling=n_readout / n_spiral_arms,
        n_slice_encoding=n_slice_encoding,
        gx_pre_duration=gx_pre_duration,
        rf_ex_duration=rf_ex_duration,
        rf_ex_bwt=rf_ex_bwt,
        rf_ref_duration=rf_ex_duration * 2,
        rf_ref_bwt=rf_ex_bwt,
        rf_ref_width_scale_factor=rf_ref_width_scale_factor,
        gz_crusher_duration=gz_crusher_duration,
        gz_crusher_area=gz_crusher_area,
        rf_spin_lock_duration=rf_spin_lock_duration,
        spin_lock_amplitude=spin_lock_amplitude,
        adiabatic_rf_duration=adiabatic_rf_duration,
        adiabatic_rf_amplitude=adiabatic_rf_amplitude,
        adiabatic_pulse_gap=adiabatic_pulse_gap,
        add_spin_lock_spoiler=add_spin_lock_spoiler,
        spin_lock_spoiler_ramp_time=spin_lock_spoiler_ramp_time,
        spin_lock_spoiler_flat_time=spin_lock_spoiler_flat_time,
        mrd_header_file=output_path / Path(filename + '_header.h5'),
    )

    # check timing of the sequence
    if timing_check and not test_report:
        ok, error_report = seq.check_timing()
        if ok:
            print('\nTiming check passed successfully')
        else:
            print('\nTiming check failed! Error listing follows\n')
            print(error_report)

    # show advanced rest report
    if test_report:
        print('\nCreating advanced test report...')
        print(seq.test_report())

    # write all required parameters in the seq-file header/definitions
    seq.set_definition('FOV', [fov_xy, fov_xy, fov_z])
    seq.set_definition('ReconMatrix', (n_readout, n_readout, n_slice_encoding))
    te_list = np.cumsum((te,) * n_echoes if te else (min_te,) * n_echoes)
    seq.set_definition('TE', te_list.tolist())
    seq.set_definition('TR', tr)
    seq.set_definition('ReadoutOversamplingFactor', readout_oversampling)
    seq.set_definition('SpiralType', spiral_type)
    seq.set_definition('T1rhoPrepType', prep_type)
    seq.set_definition('SpinLockTimes', spin_lock_times)
    seq.set_definition('AdiabaticRFDuration', adiabatic_rf_duration)
    seq.set_definition('AdiabaticRFAmplitude', adiabatic_rf_amplitude)
    seq.set_definition('AdiabaticPulseGap', adiabatic_pulse_gap)    
    seq.set_definition('SpinLockSpoilerRampTime', spin_lock_spoiler_ramp_time)
    seq.set_definition('SpinLockSpoilerFlatTime', spin_lock_spoiler_flat_time)
    seq.set_definition('AddSpinLockSpoiler', add_spin_lock_spoiler)
    seq.set_definition('RFSpinLockDuration', rf_spin_lock_duration)
    seq.set_definition('SpinLockAmplitude', spin_lock_amplitude)
    seq.set_definition('RFExcitationDuration', rf_ex_duration)
    seq.set_definition('RFExcitationBWT', rf_ex_bwt)
    seq.set_definition('RFRefocusingDuration', rf_ex_duration * 2)
    seq.set_definition('RFRefocusingBWT', rf_ex_bwt)    
    seq.set_definition('RFRefocusingWidthScaleFactor', rf_ref_width_scale_factor)
    seq.set_definition('GxPreDuration', gx_pre_duration)
    seq.set_definition('GzCrusherDuration', gz_crusher_duration)
    seq.set_definition('GzCrusherArea', gz_crusher_area)
    seq.set_definition('NumberOfSliceEncodingSteps', n_slice_encoding)
    seq.set_definition('NumberOfReadoutSteps', n_readout)
    seq.set_definition('NumberOfSpiralArms', n_spiral_arms)
    seq.set_definition('SpiralUndersampling', n_readout / n_spiral_arms)
    seq.set_definition('NumberOfEchoes', n_echoes)
    seq.set_definition('SystemLimits-MaxGrad', system.max_grad)
    seq.set_definition('SystemLimits-MaxSlew', system.max_slew)
    seq.set_definition('SystemLimits-RFRingdownTime', system.rf_ringdown_time)
    seq.set_definition('SystemLimits-RFDeadTime', system.rf_dead_time)
    seq.set_definition('SystemLimits-GradRasterTime', system.grad_raster_time)
    seq.set_definition('SystemLimits-ADCRasterTime', system.adc_raster_time)
    seq.set_definition('SystemLimits-RFRasterTime', system.rf_raster_time)
    seq.set_definition('SystemLimits-BlockDurationRaster', system.block_duration_raster)
    seq.set_definition('SystemLimits-Gamma', system.gamma)
    seq.set_definition('SystemLimits-B0', system.B0)
    
    # save seq-file to disk
    print(f"\nSaving sequence file '{filename}.seq' into folder '{output_path}'.")
    write_sequence(seq, str(output_path / filename), create_signature=True, v141_compatibility=v141_compatibility)


    if show_plots:
        seq.plot(time_range=(0, time_to_first_tr_block))

    return seq, output_path / filename


if __name__ == '__main__':
    main(tr=1,spin_lock_times = np.array([0.03,0.06]),show_plots=True,spiral_type='out', v141_compatibility=True)

#spin_lock_times = np.array([0.02, 0.04, 0.06])