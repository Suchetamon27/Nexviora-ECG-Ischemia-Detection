from io import BytesIO

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def render_ecg_window(signal: list[float]) -> bytes:
    figure, axis = plt.subplots(
        figsize=(10, 4),
        dpi=100,
    )

    axis.plot(signal, linewidth=1)
    axis.set_axis_off()

    buffer = BytesIO()

    figure.savefig(
        buffer,
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )

    plt.close(figure)

    return buffer.getvalue()