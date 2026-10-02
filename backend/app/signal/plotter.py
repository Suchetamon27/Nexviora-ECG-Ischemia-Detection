import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def generate_ecg_strip_image(samples: list, fs: int = 250, title: str = "") -> str:
    """
    Renders a 5-second ECG waveform strip on a medical-style dark grid
    and returns a base64-encoded PNG image string.
    """
    if not samples or len(samples) < 10:
        return ""

    arr = np.array(samples, dtype=float)
    t = np.arange(len(arr)) / float(fs)

    fig, ax = plt.subplots(figsize=(8.0, 2.2), dpi=120, facecolor="#0b0f19")
    ax.set_facecolor("#090d14")

    # ECG trace
    ax.plot(t, arr, color="#4ade80", linewidth=1.2, alpha=0.95)

    # Styling and grid
    ax.set_xlim(0, max(t[-1], 0.1))
    
    # Calculate Y limits with 12% padding
    ymin, ymax = float(np.min(arr)), float(np.max(arr))
    if ymax - ymin < 50:
        ymin -= 25
        ymax += 25
    pad = (ymax - ymin) * 0.12
    ax.set_ylim(ymin - pad, ymax + pad)

    # Minor & major medical grid
    ax.grid(True, which="both", color="#1e293b", linestyle="--", linewidth=0.6, alpha=0.8)
    ax.tick_params(colors="#64748b", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#1e293b")

    ax.set_xlabel("Time (s)", color="#64748b", fontsize=8, labelpad=2)
    ax.set_ylabel("ADC (counts)", color="#64748b", fontsize=8, labelpad=2)

    if title:
        ax.set_title(title, color="#cbd5e1", fontsize=9, fontweight="bold", pad=4)

    plt.tight_layout(pad=0.5)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    
    return base64.b64encode(buf.getvalue()).decode("utf-8")
