"""Memory measurement helpers."""
from __future__ import annotations
import resource
def peak_rss_bytes()->int: return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
