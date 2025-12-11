# TODO Add documentation

from dataclasses import dataclass


@dataclass
class BaseConfig:
    channel: int = 0
    sample_rate: int = int(20e6)
    centre_frequency: int = int(1e9)
    bandwidth: int = int(10e6)
    gain: int = 0


@dataclass
class RxConfig(BaseConfig):
    time_duration: float = 0.01
    buffer_size: int = 2000


@dataclass
class TxConfig(BaseConfig):
    cw_tone_frequency: int = int(1e6)
    number_samples: int = int(1e6)
