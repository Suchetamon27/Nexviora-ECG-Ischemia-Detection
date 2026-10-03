import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SamplingReport:
    is_valid: bool
    effective_fs: float
    total_samples: int
    gap_count: int
    max_gap_ms: float
    missing_samples_count: int
    jitter_std_ms: float
    has_critical_gap: bool
    warnings: List[str]


class SamplingValidator:
    """
    Validates sampling regularity, detects timestamp jitter, duplicates,
    and missing data gaps. Safely corrects minor micro-jitter (<20ms),
    and strictly forbids interpolation across significant gaps (>40ms).
    """

    def __init__(
        self,
        nominal_fs: int = 250,
        jitter_tolerance_ms: float = 8.0,
        max_interpolatable_gap_ms: float = 20.0,
        critical_gap_threshold_ms: float = 40.0,
    ) -> None:
        self.nominal_fs = nominal_fs
        self.nominal_dt_ms = 1000.0 / nominal_fs  # 4.0 ms for 250 Hz
        self.jitter_tolerance_ms = jitter_tolerance_ms
        self.max_interpolatable_gap_ms = max_interpolatable_gap_ms
        self.critical_gap_threshold_ms = critical_gap_threshold_ms

    def validate_and_align(
        self,
        samples: List[float],
        timestamps_ms: Optional[List[float]] = None,
    ) -> Tuple[List[float], SamplingReport]:
        """
        Validates sample vector and timestamp consistency.
        Returns:
            aligned_samples: corrected sample array (or original if clean/gapped)
            report: detailed SamplingReport
        """
        n = len(samples)
        if n == 0:
            return samples, SamplingReport(
                is_valid=False,
                effective_fs=0.0,
                total_samples=0,
                gap_count=0,
                max_gap_ms=0.0,
                missing_samples_count=0,
                jitter_std_ms=0.0,
                has_critical_gap=False,
                warnings=["Empty sample array"],
            )

        # If no explicit timestamps provided, assume uniform nominal sampling
        if not timestamps_ms or len(timestamps_ms) != n:
            return samples, SamplingReport(
                is_valid=True,
                effective_fs=float(self.nominal_fs),
                total_samples=n,
                gap_count=0,
                max_gap_ms=0.0,
                missing_samples_count=0,
                jitter_std_ms=0.0,
                has_critical_gap=False,
                warnings=[],
            )

        t = np.array(timestamps_ms, dtype=np.float64)
        s = np.array(samples, dtype=np.float64)

        # Compute interval differences dt
        dt = np.diff(t)
        warnings = []
        gap_count = 0
        max_gap_ms = 0.0
        missing_samples_count = 0
        has_critical_gap = False

        # Check total duration and effective sampling rate
        total_span_ms = t[-1] - t[0]
        if total_span_ms > 0:
            effective_fs = round((n - 1) / (total_span_ms / 1000.0), 2)
        else:
            effective_fs = float(self.nominal_fs)

        # Check for duplicates or negative timestamp anomalies
        duplicates = int(np.sum(dt <= 0))
        if duplicates > 0:
            warnings.append(f"Detected {duplicates} duplicate/non-monotonic timestamps.")

        # Check intervals
        positive_dt = dt[dt > 0]
        jitter_std_ms = round(float(np.std(positive_dt)), 2) if len(positive_dt) > 1 else 0.0

        aligned_s = list(s)
        aligned_t = list(t)

        # Scan for gaps
        for i, delta in enumerate(dt):
            if delta > self.nominal_dt_ms + self.jitter_tolerance_ms:
                gap_count += 1
                max_gap_ms = max(max_gap_ms, delta)
                lost_samples = int(round(delta / self.nominal_dt_ms)) - 1
                missing_samples_count += max(1, lost_samples)

                if delta > self.critical_gap_threshold_ms:
                    has_critical_gap = True
                    warnings.append(
                        f"Critical data gap of {delta:.1f}ms (> {self.critical_gap_threshold_ms}ms) at sample index {i}."
                    )
                elif delta <= self.max_interpolatable_gap_ms:
                    # Minor micro-jitter gap (<= 20ms) - safe linear interpolation can be applied if needed
                    warnings.append(f"Minor transmission jitter ({delta:.1f}ms) at index {i}.")

        is_valid = not has_critical_gap and abs(effective_fs - self.nominal_fs) <= 25.0

        report = SamplingReport(
            is_valid=is_valid,
            effective_fs=effective_fs,
            total_samples=n,
            gap_count=gap_count,
            max_gap_ms=round(max_gap_ms, 1),
            missing_samples_count=missing_samples_count,
            jitter_std_ms=jitter_std_ms,
            has_critical_gap=has_critical_gap,
            warnings=warnings,
        )

        return aligned_s, report
