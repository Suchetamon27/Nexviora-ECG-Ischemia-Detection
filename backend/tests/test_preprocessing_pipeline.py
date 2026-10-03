import os
import unittest
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.signal.filters import ECGFilter
from app.signal.sampling import SamplingValidator
from app.signal.quality import SignalQualityAssessor
from app.signal.features import extract_ecg_clinical_features


def generate_synthetic_ecg(
    duration_s: float = 5.0,
    fs: int = 250,
    hr_bpm: float = 75.0,
    baseline_adc: float = 2048.0,
    r_amplitude_adc: float = 1200.0,
    st_elevation_mm: float = 0.0,
) -> Tuple_Signal:
    """
    Generates a realistic synthetic single-lead ECG signal with known fiducial morphology.
    100 ADC counts ~= 1 mm (0.1 mV standard calibration).
    """
    t = np.linspace(0, duration_s, int(duration_s * fs), endpoint=False)
    ecg = np.full_like(t, baseline_adc)

    beat_interval_s = 60.0 / hr_bpm
    beat_times = np.arange(0.3, duration_s - 0.2, beat_interval_s)

    r_peaks = []
    st_level_counts = st_elevation_mm * 100.0

    for bt in beat_times:
        r_idx = int(round(bt * fs))
        r_peaks.append(r_idx)

        # P-wave: -160ms to -80ms (width 60ms, amp +150 counts)
        p_t = bt - 0.16
        ecg += 150.0 * np.exp(-((t - p_t) ** 2) / (2 * (0.025 ** 2)))

        # Q-wave: -30ms (amp -120 counts)
        q_t = bt - 0.03
        ecg -= 120.0 * np.exp(-((t - q_t) ** 2) / (2 * (0.008 ** 2)))

        # R-peak: 0ms (amp +1200 counts)
        ecg += r_amplitude_adc * np.exp(-((t - bt) ** 2) / (2 * (0.012 ** 2)))

        # S-wave: +30ms (amp -250 counts)
        s_t = bt + 0.03
        ecg -= 250.0 * np.exp(-((t - s_t) ** 2) / (2 * (0.010 ** 2)))

        # ST-segment offset: +60ms to +140ms
        st_t = bt + 0.09
        if abs(st_level_counts) > 1.0:
            ecg += st_level_counts * np.exp(-((t - st_t) ** 2) / (2 * (0.040 ** 2)))

        # T-wave: +240ms (width 80ms, amp +300 counts)
        t_t = bt + 0.24
        ecg += 300.0 * np.exp(-((t - t_t) ** 2) / (2 * (0.045 ** 2)))

    return t, ecg, r_peaks


class Tuple_Signal(tuple):
    pass


class TestECGPreprocessingPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fs = 250
        cls.filter = ECGFilter(
            sampling_rate=cls.fs,
            highpass_hz=0.5,
            lowpass_hz=35.0,
            notch_hz=50.0,
            notch_enabled=True,
            notch_adaptive=True,
            baseline_method="dual_median",
        )
        cls.sampling_val = SamplingValidator(nominal_fs=cls.fs)
        cls.quality_assessor = SignalQualityAssessor(sampling_rate=cls.fs)
        cls.plot_results = []

    def test_01_clean_signal_morphology_preservation(self):
        """Scenario 1: Clean ECG - Verify SNR and morphology preservation."""
        t, raw, r_peaks = generate_synthetic_ecg(fs=self.fs, st_elevation_mm=0.0)
        clean = np.array(self.filter.clean_5s_chunk(raw.tolist()))

        # Evaluate quality
        qa = self.quality_assessor.evaluate_chunk(raw.tolist(), clean.tolist())
        self.assertEqual(qa.status, "CLEAN", f"Expected CLEAN, got {qa.status}")
        self.assertGreaterEqual(qa.quality_score, 0.80)

        # Morphology preservation: R-peak amplitude preservation (> 90%)
        orig_r_amps = [raw[p] - 2048.0 for p in r_peaks if p < len(raw)]
        clean_r_amps = [clean[p] - 2048.0 for p in r_peaks if p < len(clean)]
        amp_ratio = np.mean(clean_r_amps) / np.mean(orig_r_amps)
        self.assertGreater(amp_ratio, 0.88, f"R amplitude attenuated too much ({amp_ratio:.2f})")

        # Feature extraction ST-deviation should remain ~0.0
        features = extract_ecg_clinical_features(clean.tolist(), hr=75.0, fs=self.fs)
        self.assertLess(abs(features["st_elevation_mm"]), 0.15, "Spurious ST elevation introduced!")

        self.plot_results.append(("1. Clean ECG Benchmark", t, raw, clean, f"Status: {qa.status} | Quality: {qa.quality_score:.2f} | R-Amp: {amp_ratio * 100:.1f}%"))

    def test_02_baseline_wander_removal(self):
        """Scenario 2: Severe Baseline Wander (0.15Hz drift + polynomial swing)."""
        t, clean_base, r_peaks = generate_synthetic_ecg(fs=self.fs, st_elevation_mm=0.0)
        drift = 350.0 * np.sin(2 * np.pi * 0.18 * t) + 120.0 * ((t - 2.5) ** 2)
        raw_noisy = clean_base + drift

        clean = np.array(self.filter.clean_5s_chunk(raw_noisy.tolist()))
        qa = self.quality_assessor.evaluate_chunk(raw_noisy.tolist(), clean.tolist())

        # Check that baseline drift is eliminated (median deviation close to 2048)
        std_drift_raw = np.std(raw_noisy - np.mean(raw_noisy))
        std_clean = np.std(clean - 2048.0)
        self.assertLess(std_clean, std_drift_raw * 0.75, "Baseline wander was not sufficiently attenuated")

        # Verify ST level is preserved without distortion
        features = extract_ecg_clinical_features(clean.tolist(), hr=75.0, fs=self.fs)
        self.assertLess(abs(features["st_elevation_mm"]), 0.20, "Baseline removal corrupted ST level!")

        self.plot_results.append(("2. Baseline Wander (0.18 Hz Drift)", t, raw_noisy, clean, f"Status: {qa.status} | Quality: {qa.quality_score:.2f} | Drift Suppressed"))

    def test_03_powerline_50hz_noise_removal(self):
        """Scenario 3: 50 Hz Powerline Mains Hum + Harmonic."""
        t, clean_base, _ = generate_synthetic_ecg(fs=self.fs, st_elevation_mm=0.0)
        mains_noise = 180.0 * np.sin(2 * np.pi * 50.0 * t) + 45.0 * np.sin(2 * np.pi * 100.0 * t)
        raw_noisy = clean_base + mains_noise

        clean = np.array(self.filter.clean_5s_chunk(raw_noisy.tolist()))
        qa = self.quality_assessor.evaluate_chunk(raw_noisy.tolist(), clean.tolist())

        # Spectral power reduction at 50Hz
        fft_raw = np.abs(np.fft.rfft(raw_noisy - np.mean(raw_noisy)))
        fft_clean = np.abs(np.fft.rfft(clean - np.mean(clean)))
        freqs = np.fft.rfftfreq(len(raw_noisy), 1.0 / self.fs)
        idx_50 = np.argmin(np.abs(freqs - 50.0))

        notch_attenuation_db = 20 * np.log10(max(1e-6, fft_raw[idx_50]) / max(1e-6, fft_clean[idx_50]))
        self.assertGreater(notch_attenuation_db, 18.0, f"50 Hz hum attenuation ({notch_attenuation_db:.1f} dB) below target")

        self.plot_results.append(("3. 50 Hz Powerline Hum", t, raw_noisy, clean, f"Status: {qa.status} | 50Hz Attenuation: {notch_attenuation_db:.1f} dB"))

    def test_04_high_frequency_emg_noise(self):
        """Scenario 4: High-Frequency EMG Muscle Tremor Noise (35-100 Hz)."""
        np.random.seed(42)
        t, clean_base, _ = generate_synthetic_ecg(fs=self.fs, st_elevation_mm=0.0)
        emg_noise = np.random.normal(0, 110.0, len(t))
        raw_noisy = clean_base + emg_noise

        clean = np.array(self.filter.clean_5s_chunk(raw_noisy.tolist()))
        qa = self.quality_assessor.evaluate_chunk(raw_noisy.tolist(), clean.tolist())

        # Noise variance reduction
        raw_hf_std = np.std(np.diff(raw_noisy))
        clean_hf_std = np.std(np.diff(clean))
        hf_reduction = 1.0 - (clean_hf_std / raw_hf_std)
        self.assertGreater(hf_reduction, 0.65, f"HF EMG noise reduction ({hf_reduction*100:.1f}%) too low")

        self.plot_results.append(("4. High-Frequency Muscle Noise", t, raw_noisy, clean, f"Status: {qa.status} | Noise Variance Reduced: {hf_reduction*100:.1f}%"))

    def test_05_motion_artifact_detection(self):
        """Scenario 5: Body Movement & Accelerometer Artifact."""
        t, clean_base, _ = generate_synthetic_ecg(fs=self.fs)
        # Abrupt motion transient step in middle of trace
        step = np.zeros_like(t)
        step[int(1.5 * self.fs):int(2.8 * self.fs)] = 600.0
        raw_noisy = clean_base + step
        motion_samples = [1.45 if (1.5 <= ti <= 2.8) else 1.0 for ti in t]

        clean = np.array(self.filter.clean_5s_chunk(raw_noisy.tolist()))
        qa = self.quality_assessor.evaluate_chunk(raw_noisy.tolist(), clean.tolist(), motion_samples=motion_samples)

        self.assertIn("EXCESSIVE_MOTION", qa.detected_artifacts, "Failed to flag EXCESSIVE_MOTION artifact")
        self.plot_results.append(("5. Accelerometer Motion Artifact", t, raw_noisy, clean, f"Status: {qa.status} | Detected: {','.join(qa.detected_artifacts)}"))

    def test_06_sampling_gap_detection(self):
        """Scenario 6: Missing Samples & Serial Gap Detection."""
        t, clean_base, _ = generate_synthetic_ecg(fs=self.fs)
        # Simulate 70ms gap in timestamps
        timestamps_ms = list(t * 1000.0)
        # Inject jump at index 300
        for i in range(300, len(timestamps_ms)):
            timestamps_ms[i] += 70.0  # 70ms jump

        aligned, report = self.sampling_val.validate_and_align(clean_base.tolist(), timestamps_ms)
        self.assertTrue(report.has_critical_gap, "SamplingValidator failed to catch 70ms gap")
        self.assertGreaterEqual(report.max_gap_ms, 70.0)

        qa = self.quality_assessor.evaluate_chunk(clean_base.tolist(), clean_base.tolist(), gap_detected=report.has_critical_gap, gap_max_ms=report.max_gap_ms)
        self.assertEqual(qa.status, "REJECTED", "Critical gap should reject the segment")
        self.assertIn("DATA_GAP", qa.detected_artifacts)

        self.plot_results.append(("6. Data Gap (70ms Drop)", t, clean_base, clean_base, f"Status: {qa.status} | Max Gap: {report.max_gap_ms:.1f}ms | REJECTED"))

    def test_07_clipping_saturation_detection(self):
        """Scenario 7: ADC Saturation / Railing."""
        t, clean_base, _ = generate_synthetic_ecg(fs=self.fs)
        railed = np.clip(clean_base + 800.0, 0, 4095)
        # Force 20% to clip at 4095
        railed[100:400] = 4095.0

        clean = np.array(self.filter.clean_5s_chunk(railed.tolist()))
        qa = self.quality_assessor.evaluate_chunk(railed.tolist(), clean.tolist())

        self.assertIn("CLIPPING_SATURATION", qa.detected_artifacts, "Failed to flag CLIPPING_SATURATION")
        self.assertEqual(qa.status, "REJECTED", "Railed segment must be REJECTED")

        self.plot_results.append(("7. ADC Clipping (4095 Counts)", t, railed, clean, f"Status: {qa.status} | Artifact: {','.join(qa.detected_artifacts)} | REJECTED"))

    @classmethod
    def tearDownClass(cls):
        """Generates comprehensive visual verification plot artifact."""
        if not cls.plot_results:
            return

        plots_dir = os.path.dirname(__file__)
        plot_path = os.path.join(plots_dir, "preprocessing_verification_plots.png")

        fig, axes = plt.subplots(len(cls.plot_results), 1, figsize=(14, 2.8 * len(cls.plot_results)), sharex=True)
        fig.suptitle("Nexviora Robust Preprocessing Pipeline — Synthetic Noise Verification Suite", fontsize=14, fontweight="bold", y=0.995)

        for i, (title, t, raw, clean, meta) in enumerate(cls.plot_results):
            ax = axes[i]
            ax.plot(t, raw, color="#ef4444", alpha=0.6, linewidth=1.0, label="Raw Unfiltered Signal")
            ax.plot(t, clean, color="#10b981", linewidth=1.4, label="Cleaned Signal (Robust Preprocessing)")
            ax.set_title(f"{title}  [{meta}]", fontsize=10, fontweight="bold", pad=4, loc="left", color="#1e293b")
            ax.set_ylabel("ADC Counts", fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.5)
            if i == 0:
                ax.legend(loc="upper right", fontsize=8)

        axes[-1].set_xlabel("Time (seconds)", fontsize=10)
        plt.tight_layout()
        plt.subplots_adjust(top=0.96)
        plt.savefig(plot_path, dpi=160, bbox_inches="tight")
        plt.close()
        print(f"\n[Test Suite] Saved verification plots to: {plot_path}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
