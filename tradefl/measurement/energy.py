"""Best-effort accelerator energy measurement using ``nvidia-smi``.

The meter integrates sampled board power over wall-clock time.  It deliberately
returns ``None`` when NVIDIA telemetry is unavailable rather than inventing an
energy estimate.
"""
from __future__ import annotations

import subprocess
import threading
import time
from collections.abc import Callable


class EnergyMeter:
    """Integrate GPU board power samples and return joules."""

    def __init__(
        self,
        sample_interval_seconds: float = 0.5,
        power_reader: Callable[[], float | None] | None = None,
    ) -> None:
        self.sample_interval_seconds = sample_interval_seconds
        self.power_reader = power_reader or _read_nvidia_power_watts
        self._samples: list[tuple[float, float]] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start sampling; repeated use of one meter starts a fresh interval."""

        self._samples = []
        self._stop_event.clear()
        self._sample()
        self._thread = threading.Thread(target=self._run, name="tradefl-energy-meter", daemon=True)
        self._thread.start()

    def stop(self) -> float | None:
        """Stop sampling and integrate watts over seconds using trapezoids."""

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.sample_interval_seconds * 2))
        self._sample()
        if len(self._samples) < 2:
            return None
        return sum(
            (right_time - left_time) * (left_watts + right_watts) / 2
            for (left_time, left_watts), (right_time, right_watts) in zip(self._samples, self._samples[1:])
        )

    def _run(self) -> None:
        while not self._stop_event.wait(self.sample_interval_seconds):
            self._sample()

    def _sample(self) -> None:
        watts = self.power_reader()
        if watts is not None:
            self._samples.append((time.monotonic(), watts))


def _read_nvidia_power_watts() -> float | None:
    """Return aggregate board power for visible NVIDIA GPUs, if available."""

    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if completed.returncode:
        return None
    try:
        values = [float(line.strip()) for line in completed.stdout.splitlines() if line.strip() not in {"", "[N/A]"}]
    except ValueError:
        return None
    return sum(values) if values else None
