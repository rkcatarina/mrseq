"""Adiabatic T1rho preparation block using pairs of hyperbolic secant (HS) full passage pulses."""

import numpy as np
import pypulseq as pp

from mrseq.utils import sys_defaults
from mrseq.utils.constants import GYROMAGNETIC_RATIO_PROTON


def add_t1rho_adiabatic_prep(
    seq: pp.Sequence | None = None,
    system: pp.Opts | None = None,
    n_pairs: int = 1,
    rf_duration: float = 15e-3,
    rf_amplitude: float = 11.7e-6,
    rf_bandwidth: float = 800.0,
    rf_mu: float = 4.0,
    rf_dwell: float = 8e-6,
    pulse_gap: float = 350e-6,
    add_spoiler: bool = True,
    spoiler_ramp_time: float = 6e-4,
    spoiler_flat_time: float = 8.4e-3,
) -> tuple[pp.Sequence, float, float]:
    """Adiabatic T1rho prep block using pairs of hyperbolic secant (HS) full passage pulses

    The block consists of `n_pairs` pairs of hyperbolic secant (HS) adiabatic full passage pulses, the second pulse
    of each pair being phase-inverted (180° phase offset). Each pair returns the magnetization to +z. No tip-down /
    tip-up pulses are used: the magnetization follows the effective field during each adiabatic full passage and
    relaxes with adiabatic T1rho. The implementation and default parameters follow [Si25]_. Optionally, spoiler gradients are added at the end of
    the block.

    The HS pulse is defined by
        B1(t)   = rf_amplitude * sech(beta * t)
        w_RF(t) = -mu * beta * tanh(beta * t)
    with beta = 2 * pi * rf_bandwidth / (2 * mu) and t in [-rf_duration / 2, rf_duration / 2]. The phase modulation
    is the analytic integral of the frequency modulation: phi(t) = -mu * ln(cosh(beta * t)).

    Parameters
    ----------
    seq
        PyPulseq Sequence object.
    system
        PyPulseq system limit object.
    n_pairs
        Number of HS pulse pairs. The total adiabatic spin-lock time is 2 * n_pairs * rf_duration.
    rf_duration
        Duration of each HS pulse (in seconds).
    rf_amplitude
        Peak B1 amplitude of the HS pulses (in T).
    rf_bandwidth
        Frequency sweep bandwidth of the HS pulses (in Hz).
    rf_mu
        Dimensionless HS parameter mu, determining the amplitude of the frequency sweep.
    rf_dwell
        Dwell time of the HS waveform samples (in seconds). Must be a multiple of system.rf_raster_time.
    pulse_gap
        Delay between the two HS pulses of each pair (in seconds).
    add_spoiler
        Toggles addition of spoiler gradients after the last HS pulse.
    spoiler_ramp_time
        Duration of gradient spoiler ramps (in seconds).
    spoiler_flat_time
        Duration of gradient spoiler plateau (in seconds).

    Returns
    -------
    seq
        PyPulseq Sequence object.
    block_duration
        Total duration of the adiabatic T1rho preparation block (in seconds).
    spin_lock_time
        Total duration of all HS pulses (= effective adiabatic spin-lock time) (in seconds).

    Raises
    ------
    ValueError
        If n_pairs is smaller than 1.

    References
    ----------
     Si D, et al. (2025) Magn Reson Med. https://doi.org/10.1002/mrm.30582
    """
    if n_pairs < 1:
        raise ValueError(f'Number of HS pulse pairs must be at least 1, got {n_pairs}.')

    # set system to default if not provided
    if system is None:
        system = sys_defaults

    # create new sequence if not provided
    if seq is None:
        seq = pp.Sequence(system=system)

    # get current duration of sequence before adding T1rho preparation block
    time_start = sum(seq.block_durations.values())

    # calculate HS waveform (amplitude in Hz) sampled at the center of each dwell interval
    n_samples = round(rf_duration / rf_dwell)
    time = (np.arange(n_samples) + 0.5) * rf_dwell - rf_duration / 2
    beta = 2 * np.pi * rf_bandwidth / (2 * rf_mu)  # in rad/s
    amplitude = GYROMAGNETIC_RATIO_PROTON * rf_amplitude / np.cosh(beta * time)  # in Hz
    phase = -rf_mu * np.log(np.cosh(beta * time))  # in rad
    signal = amplitude * np.exp(1j * phase)

    # create HS pulses without amplitude rescaling, second pulse of each pair is phase-inverted
    hs_pulses = [
        pp.make_arbitrary_rf(
            signal=signal,
            flip_angle=2 * np.pi,  # ignored because of no_signal_scaling=True
            dwell=rf_dwell,
            delay=system.rf_dead_time,
            phase_offset=phase_offset,
            no_signal_scaling=True,
            system=system,
            use='preparation',
        )
        for phase_offset in (0, np.pi)
    ]

    # add HS pulse pairs to sequence
    for _ in range(n_pairs):
        seq.add_block(hs_pulses[0])
        if pulse_gap > 0:
            seq.add_block(pp.make_delay(pulse_gap))
        seq.add_block(hs_pulses[1])

    # add spoiler gradient if requested
    if add_spoiler:
        gz_spoil = pp.make_trapezoid(
            channel='z',
            amplitude=0.4 * system.max_grad,
            flat_time=spoiler_flat_time,
            rise_time=spoiler_ramp_time,
            system=system,
        )
        seq.add_block(gz_spoil)

    # calculate total duration of adiabatic T1rho preparation block
    block_duration = sum(seq.block_durations.values()) - time_start

    # effective adiabatic spin-lock time
    spin_lock_time = 2 * n_pairs * n_samples * rf_dwell

    return (seq, block_duration, spin_lock_time)
