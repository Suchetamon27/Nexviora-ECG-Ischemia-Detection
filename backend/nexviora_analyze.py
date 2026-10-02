#!/usr/bin/env python3
"""
Nexviora ECG Analysis — Interactive Zoom Edition
-------------------------------------------------
5-panel figure with a TIME SLIDER + WINDOW SLIDER at the bottom.
Drag the sliders to scrub through the recording at 1s / 2s / 5s windows
and see every P-wave, QRS complex, T-wave in full detail.

Usage:
    python3 nexviora_analyze.py                       # most recent file
    python3 nexviora_analyze.py nexviora_rec_xxx.csv  # specific file
"""

import sys, os, glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
import neurokit2 as nk
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- Config -----
RECORDINGS_DIR = os.path.dirname(os.path.abspath(__file__))
FS             = 250
MOTION_THRESH  = 0.15
BEAT_PRE       = 75
BEAT_POST      = 187
NOTCH_HZ       = 50

# ----------------------------------------------------------------- Load ------
if len(sys.argv) > 1:
    csv_path = sys.argv[1]
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(RECORDINGS_DIR, csv_path)
else:
    files = sorted(glob.glob(os.path.join(RECORDINGS_DIR, "nexviora_rec_*.csv")))
    if not files:
        sys.exit("No nexviora_rec_*.csv found.")
    csv_path = files[-1]

print(f"[Nexviora] Loading: {csv_path}")
df = pd.read_csv(csv_path)
df = df.drop_duplicates("timestamp_ms").sort_values("timestamp_ms").reset_index(drop=True)

t   = df["timestamp_ms"].to_numpy() / 1000.0
ecg = df["ecg_filtered"].to_numpy(float)
ppg = df["ppg_dc_ir"].to_numpy(float)

has_imu = all(c in df.columns for c in ["pitch_deg", "roll_deg", "motion_g"])
if has_imu:
    pitch_raw  = df["pitch_deg"].to_numpy(float)
    roll_raw   = df["roll_deg"].to_numpy(float)
    motion_raw = df["motion_g"].to_numpy(float)

duration = t[-1] - t[0]
print(f"[Nexviora] Duration: {duration:.1f} s  |  Samples: {len(t)}")

# ----------------------------------------------- Resample to exact 250 Hz ----
tu    = np.arange(t[0], t[-1], 1.0/FS)
ecg_u = np.interp(tu, t, ecg)
ppg_u = np.interp(tu, t, ppg)
if has_imu:
    pitch_u  = np.interp(tu, t, pitch_raw)
    roll_u   = np.interp(tu, t, roll_raw)
    motion_u = np.interp(tu, t, motion_raw)

# ------------------------------------------------------------- Filtering -----
bp, ba = butter(3, [0.5, 40.0], fs=FS, btype="band")
ecg_f  = filtfilt(bp, ba, ecg_u)
nb, na = iirnotch(NOTCH_HZ, 30, FS)
ecg_f  = filtfilt(nb, na, ecg_f)

bp2, ba2 = butter(3, [0.5, 8.0], fs=FS, btype="band")
ppg_f    = filtfilt(bp2, ba2, ppg_u)

# --------------------------------------------------------- Motion mask -------
if has_imu:
    bad = np.abs(motion_u - 1.0) > MOTION_THRESH
else:
    bad = np.zeros(len(tu), dtype=bool)
n_bad = bad.sum()
print(f"[Nexviora] Motion artefact: {n_bad}/{len(tu)} ({n_bad/len(tu)*100:.1f}%)")

# ---------------------------------------------------- R-peak detection -------
print("[Nexviora] NeuroKit2 processing ...")
_, info = nk.ecg_process(ecg_f, sampling_rate=FS)
r_peaks = np.array(info["ECG_R_Peaks"])
r_clean = r_peaks[~bad[r_peaks]]
rr_ms   = np.diff(tu[r_clean]) * 1000.0
hr_bpm  = 60_000.0 / rr_ms
print(f"[Nexviora] R-peaks: {len(r_clean)} clean / {len(r_peaks)} total")
if len(hr_bpm):
    print(f"[Nexviora] HR -> mean {hr_bpm.mean():.1f}  median {np.median(hr_bpm):.1f}  "
          f"min {hr_bpm.min():.1f}  max {hr_bpm.max():.1f} BPM")

# ---------------------------------------------------- Averaged PQRST --------
beats = []
for i in r_clean:
    lo, hi = i - BEAT_PRE, i + BEAT_POST
    if lo < 0 or hi > len(ecg_f): continue
    if bad[lo:hi].any(): continue
    seg = ecg_f[lo:hi].copy()
    seg -= np.median(seg[:50])
    beats.append(seg)
avg_beat = np.mean(beats, axis=0) if beats else None
t_beat   = (np.arange(BEAT_PRE + BEAT_POST) - BEAT_PRE) / FS * 1000.0
print(f"[Nexviora] PQRST template: {len(beats)} clean beats")

# --------------------------------------------------------- Noise floor -------
nm = np.ones(len(ecg_f), bool)
for i in r_clean: nm[max(0,i-75):i+75] = False
calm_sigma = ecg_f[nm & ~bad].std() if (nm & ~bad).any() else float("nan")

# ============================================================== PLOT =========
tu_rel = tu - tu[0]
total_dur = tu_rel[-1]

C = dict(
    ecg="#4ade80", filt="#22d3ee", peak="#f43f5e", avg="#f59e0b",
    ppg="#a78bfa", ppg2="#d946ef", imuP="#fb923c", imuR="#60a5fa",
    imuG="#facc15", shade="#7f1d1d", grid="#1e293b", label="#94a3b8",
    title="#e2e8f0", bg="#0e1117", ax_bg="#090d14"
)

N_ROWS = 5 if has_imu else 4
fig = plt.figure(figsize=(20, 14), facecolor=C["bg"])
fig.canvas.manager.set_window_title("Nexviora ECG Analysis — Interactive Zoom")

# Main grid: rows for plots + bottom strip for sliders
outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[12, 1],
                          hspace=0.25, top=0.93, bottom=0.04, left=0.055, right=0.97)

# Inner grid for data panels
inner = gridspec.GridSpecFromSubplotSpec(N_ROWS, 2, subplot_spec=outer[0],
                                         width_ratios=[3, 1],
                                         hspace=0.55, wspace=0.22)

# Slider strip
slider_gs = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[1],
                                             width_ratios=[3, 2, 2, 0.8], wspace=0.3)

def sax(ax, title, xlabel="Time (s)", ylabel=""):
    ax.set_facecolor(C["ax_bg"])
    ax.tick_params(colors=C["label"], labelsize=8)
    ax.xaxis.label.set_color(C["label"]); ax.yaxis.label.set_color(C["label"])
    ax.set_xlabel(xlabel, fontsize=8); ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, color=C["title"], fontsize=9, fontweight="bold", pad=4)
    for sp in ax.spines.values(): sp.set_color(C["grid"])
    ax.grid(True, color=C["grid"], lw=0.6, ls="--")

# ---- 1. Raw ECG -------------------------------------------------------------
ax1 = fig.add_subplot(inner[0, 0])
sax(ax1, "(1)  Raw ECG  [AD8232]", ylabel="ADC counts")
line_raw, = ax1.plot([], [], color=C["ecg"], lw=0.9, alpha=0.9)

# ---- 2. Filtered ECG + R-peaks ----------------------------------------------
ax2 = fig.add_subplot(inner[1, 0], sharex=ax1)
sax(ax2, "(2)  Filtered ECG + R-peaks", ylabel="Filtered counts")
line_filt, = ax2.plot([], [], color=C["filt"], lw=1.0, alpha=0.9)
scat_peaks = ax2.scatter([], [], color=C["peak"], s=40, zorder=5)

# ---- 3. Instantaneous HR ----------------------------------------------------
ax3 = fig.add_subplot(inner[2, 0], sharex=ax1)
sax(ax3, "(3)  Heart Rate  [BPM]", ylabel="BPM")
line_hr, = ax3.plot([], [], color=C["avg"], lw=1.3, marker="o", ms=4)
if len(hr_bpm):
    hr_mean_line = ax3.axhline(hr_bpm.mean(), color=C["peak"], lw=1.0, ls="--")

# ---- 4. PPG -----------------------------------------------------------------
ax4 = fig.add_subplot(inner[3, 0], sharex=ax1)
sax(ax4, "(4)  PPG  [MAX30102 IR]", ylabel="IR counts")
line_ppg_raw, = ax4.plot([], [], color=C["ppg"], lw=0.6, alpha=0.7)
line_ppg_flt, = ax4.plot([], [], color=C["ppg2"], lw=1.1, alpha=0.9)

# ---- 5. IMU -----------------------------------------------------------------
ax5, axg = None, None
line_pitch, line_roll, line_g = None, None, None
if has_imu:
    ax5 = fig.add_subplot(inner[4, 0], sharex=ax1)
    sax(ax5, "(5)  IMU  [MPU-6500]", ylabel="degrees")
    line_pitch, = ax5.plot([], [], color=C["imuP"], lw=0.9, alpha=0.9, label="Pitch")
    line_roll,  = ax5.plot([], [], color=C["imuR"], lw=0.9, alpha=0.9, label="Roll")
    ax5.legend(fontsize=7, facecolor=C["bg"], labelcolor=C["label"], loc="upper left")
    axg = ax5.twinx()
    axg.set_ylabel("|g|", color=C["imuG"], fontsize=8)
    axg.tick_params(colors=C["imuG"], labelsize=8)
    line_g, = axg.plot([], [], color=C["imuG"], lw=0.7, alpha=0.5)

# ---- Right: averaged PQRST --------------------------------------------------
ax_a = fig.add_subplot(inner[:3, 1])
sax(ax_a, "Averaged PQRST Template", xlabel="Time (ms)", ylabel="Filtered counts")
if avg_beat is not None:
    for seg in beats:
        ax_a.plot(t_beat, seg, color=C["filt"], lw=0.35, alpha=0.04)
    ax_a.plot(t_beat, avg_beat, color=C["avg"], lw=2.2, label=f"Avg (n={len(beats)})")
    ax_a.axvline(0, color=C["peak"], lw=1.2, ls="--", alpha=0.8)
    ax_a.text(4, avg_beat.max()*0.95, "R", color=C["peak"], fontsize=10, fontweight="bold")
    ax_a.legend(fontsize=7, facecolor=C["bg"], labelcolor=C["label"])

# ---- Right: metrics ---------------------------------------------------------
ax_m = fig.add_subplot(inner[3:, 1])
ax_m.set_facecolor(C["ax_bg"])
ax_m.axis("off")

sdnn  = rr_ms.std() if len(rr_ms) > 1 else float("nan")
rmssd = np.sqrt(np.mean(np.diff(rr_ms)**2)) if len(rr_ms) > 2 else float("nan")

mlines = [
    f"File: {os.path.basename(csv_path)}",
    f"Duration:  {duration:.1f} s",
    f"Samples:   {len(tu):,} @ 250 Hz",
    "",
    f"R-peaks:   {len(r_clean)} clean",
    f"Beats:     {len(beats)} templates",
    "",
]
if len(hr_bpm):
    mlines += [
        f"HR mean:   {hr_bpm.mean():.1f} BPM",
        f"HR median: {np.median(hr_bpm):.1f} BPM",
        f"HR range:  {hr_bpm.min():.0f}–{hr_bpm.max():.0f}",
        f"SDNN:      {sdnn:.1f} ms",
        f"RMSSD:     {rmssd:.1f} ms",
    ]
mlines += [f"", f"Noise sig: {calm_sigma:.3f}", f"Motion:    {n_bad/len(tu)*100:.1f}%"]

ax_m.text(0.05, 0.97, "\n".join(mlines), transform=ax_m.transAxes,
          fontsize=7.5, color=C["label"], va="top", family="monospace", linespacing=1.6)

fig.suptitle(f"Nexviora ECG Analysis   |   {os.path.basename(csv_path)}",
             color=C["title"], fontsize=12, fontweight="bold")

# ============================= INTERACTIVE SLIDERS ============================

# Time position slider
ax_slider_t = fig.add_subplot(slider_gs[0])
ax_slider_t.set_facecolor(C["ax_bg"])
slider_time = Slider(ax_slider_t, 'Start (s)', 0, max(0.01, total_dur - 1),
                     valinit=0, valstep=0.1,
                     color=C["filt"], track_color=C["grid"])
slider_time.label.set_color(C["label"])
slider_time.valtext.set_color(C["label"])

# Window width slider
ax_slider_w = fig.add_subplot(slider_gs[1])
ax_slider_w.set_facecolor(C["ax_bg"])
slider_win = Slider(ax_slider_w, 'Window (s)', 0.5, min(30, total_dur),
                    valinit=min(2.0, total_dur), valstep=0.5,
                    color=C["avg"], track_color=C["grid"])
slider_win.label.set_color(C["label"])
slider_win.valtext.set_color(C["label"])

# Y-axis height slider (controls the visible ADC range for ECG plots)
ax_slider_h = fig.add_subplot(slider_gs[2])
ax_slider_h.set_facecolor(C["ax_bg"])
slider_height = Slider(ax_slider_h, 'Y Range', 200, 4000,
                       valinit=3000, valstep=100,
                       color=C["peak"], track_color=C["grid"])
slider_height.label.set_color(C["label"])
slider_height.valtext.set_color(C["label"])

# "Show All" button
ax_btn = fig.add_subplot(slider_gs[3])
ax_btn.set_facecolor(C["ax_bg"])
btn_all = Button(ax_btn, 'Show All', color=C["grid"], hovercolor="#334155")
btn_all.label.set_color(C["title"])
btn_all.label.set_fontsize(10)
btn_all.label.set_fontweight("bold")

# Precompute HR time values
t_hr = tu_rel[r_clean[1:]] if len(r_clean) > 1 else np.array([])

def update_view(t_start, t_end):
    """Redraw all plots for the given time window."""
    mask_s = (tu_rel >= t_start) & (tu_rel <= t_end)
    idx = np.where(mask_s)[0]
    if len(idx) < 2:
        return

    t_w = tu_rel[idx]

    # Y-axis range from height slider (centered on ADC midpoint 2048)
    y_range = slider_height.val
    ecg_mid = 2048
    y_lo = ecg_mid - y_range / 2
    y_hi = ecg_mid + y_range / 2

    # 1. Raw ECG — fixed Y
    line_raw.set_data(t_w, ecg_u[idx])
    ax1.set_xlim(t_start, t_end)
    ax1.set_ylim(y_lo, y_hi)

    # 2. Filtered ECG + peaks — fixed Y (same range, centered on filtered mean)
    line_filt.set_data(t_w, ecg_f[idx])
    pk_mask = (r_clean >= idx[0]) & (r_clean <= idx[-1])
    pk_in = r_clean[pk_mask]
    if len(pk_in):
        scat_peaks.set_offsets(np.column_stack([tu_rel[pk_in], ecg_f[pk_in]]))
    else:
        scat_peaks.set_offsets(np.empty((0, 2)))
    # For filtered, center on the median of the visible data
    filt_med = np.median(ecg_f[idx])
    ax2.set_ylim(filt_med - y_range / 2, filt_med + y_range / 2)

    # 3. HR
    hr_mask = (t_hr >= t_start) & (t_hr <= t_end)
    if hr_mask.any():
        line_hr.set_data(t_hr[hr_mask], hr_bpm[hr_mask])
        hr_vis = hr_bpm[hr_mask]
        ax3.set_ylim(max(30, hr_vis.min()-8), min(220, hr_vis.max()+8))
    else:
        line_hr.set_data([], [])

    # 4. PPG
    line_ppg_raw.set_data(t_w, ppg_u[idx])
    line_ppg_flt.set_data(t_w, ppg_f[idx])
    y_ppg = ppg_u[idx]
    ppad = (y_ppg.max() - y_ppg.min()) * 0.08 + 5
    ax4.set_ylim(y_ppg.min() - ppad, y_ppg.max() + ppad)

    # 5. IMU
    if has_imu:
        line_pitch.set_data(t_w, pitch_u[idx])
        line_roll.set_data(t_w, roll_u[idx])
        all_deg = np.concatenate([pitch_u[idx], roll_u[idx]])
        dpad = (all_deg.max() - all_deg.min()) * 0.1 + 1
        ax5.set_ylim(all_deg.min() - dpad, all_deg.max() + dpad)
        line_g.set_data(t_w, motion_u[idx])
        gv = motion_u[idx]
        gpad = (gv.max() - gv.min()) * 0.1 + 0.02
        axg.set_ylim(gv.min() - gpad, gv.max() + gpad)

    fig.canvas.draw_idle()

def on_slider_change(val):
    t_start = slider_time.val
    win     = slider_win.val
    t_end   = min(t_start + win, total_dur)
    slider_time.valmax = max(0.01, total_dur - win)
    if t_start > slider_time.valmax:
        slider_time.set_val(slider_time.valmax)
        return
    update_view(t_start, t_end)

def on_show_all(event):
    slider_time.set_val(0)
    slider_win.set_val(min(30, total_dur))
    update_view(0, total_dur)

slider_time.on_changed(on_slider_change)
slider_win.on_changed(on_slider_change)
slider_height.on_changed(on_slider_change)
btn_all.on_clicked(on_show_all)

# Initial view: first 2 seconds (or full if <2s)
init_win = min(2.0, total_dur)
update_view(0, init_win)

# Save full overview PNG
update_view(0, total_dur)
out_png = csv_path.replace(".csv", "_analysis.png")
plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"[Nexviora] Saved overview: {out_png}")

# Reset to zoomed-in 2s view for interactive exploration
update_view(0, init_win)

print(f"\n[Nexviora] ✓ Interactive window open!")
print(f"  • Drag 'Start' slider to scrub through the recording")
print(f"  • Drag 'Window' slider to set zoom (0.5s to {min(30, total_dur):.0f}s)")
print(f"  • Click 'Show All' to see the full recording")
print(f"  • Close the window to exit\n")

plt.show()
