from collections import deque
from typing import List, Optional

class ECGWindow:
    def __init__(
        self,
        sampling_rate: int,
        window_seconds: float,
        step_seconds: float,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.window_size = int(sampling_rate * window_seconds)
        self.step_size = int(sampling_rate * step_seconds)

        self.samples = deque(maxlen=self.window_size)
        self._since_last_window = 0

    def add_chunk(self, chunk: List[float]) -> Optional[List[float]]:
        # Returns the window if we reached the step size
        windows = []
        for sample in chunk:
            self.samples.append(sample)
            self._since_last_window += 1
            if len(self.samples) == self.window_size and self._since_last_window >= self.step_size:
                windows.append(list(self.samples))
                self._since_last_window = 0
        
        # In a real streaming scenario we might have multiple steps in one chunk.
        # But here step_size is 250 and chunk is 250, so we return the latest one.
        if windows:
            return windows[-1]
        return None