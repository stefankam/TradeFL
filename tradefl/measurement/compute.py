"""Compute timing placeholders for measured device time."""
from time import perf_counter
class Timer:
    def __enter__(self): self.start=perf_counter(); return self
    def __exit__(self,*_): self.seconds=perf_counter()-self.start
