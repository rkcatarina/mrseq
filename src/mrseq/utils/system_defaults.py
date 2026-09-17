"""Define default system limitations."""

from pypulseq.opts import Opts

# sys_defaults = Opts(
#     max_grad=30,
#     grad_unit='mT/m',
#     max_slew=120,
#     slew_unit='T/m/s',
#     rf_ringdown_time=30e-6,
#     rf_dead_time=100e-6,
#     adc_dead_time=10e-6,
#     B0 = 2.89
# )

#System settingg for OSI2one v1.0
GAMMA = 42.576e6
sys_defaults = Opts(
    max_grad=629e3 / GAMMA * 1e3,
    grad_unit='mT/m',
    max_slew=25,
    slew_unit='T/m/s',
    rf_ringdown_time=30e-6,
    rf_dead_time=20e-6,
    grad_raster_time=10e-6,
    adc_raster_time=1e-6,
    rf_raster_time=1e-6,
    block_duration_raster=1e-6,
    gamma=GAMMA,
    B0 = 50e-3
)

# System settings for low-field 0.55T scanner in Chile
# sys_defaults = Opts(
#     max_grad=24,
#     grad_unit='mT/m',
#     max_slew=38,
#     slew_unit='T/m/s',
#     rf_ringdown_time=20e-6,
#     rf_dead_time=100e-6,
#     adc_dead_time=10e-6,
#     B0=0.55,
# )
