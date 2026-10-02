import math

from ..hardware.models import SensorSample


def calculate_motion_score(sample: SensorSample) -> float:
    acceleration = math.sqrt(
        sample.ax**2 +
        sample.ay**2 +
        sample.az**2
    )

    gyro = math.sqrt(
        sample.gx**2 +
        sample.gy**2 +
        sample.gz**2
    )

    acceleration_motion = abs(acceleration - 1.0)

    return acceleration_motion + gyro * 0.05