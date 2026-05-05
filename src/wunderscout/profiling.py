from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
import time
from typing import Dict, Optional, List


@dataclass
class Profiler:
    section_times: Dict[str, List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    num_detections: int = 0
    confidence_count: float = 0.0

    def add_section(self, section_name: str, duration: float):
        self.section_times[section_name].append(duration)

    def add_detections(self, num_detections: int, condifence: float):
        self.num_detections += num_detections
        self.confidence_count += condifence

    def report(self):
        return {
            "section_times": self.section_times,
            "num_detections": self.num_detections,
            "confidence_count": self.confidence_count,
        }


@contextmanager
def timed_section(stats: Optional[Profiler], name: str):
    if stats is None:
        yield
        return

    start = time.perf_counter()
    try:
        yield
    finally:
        duration = round((time.perf_counter() - start) * 1000, 2)  # Convert to ms
        stats.add_section(name, duration)
