from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SaveResult:
    successful_paths: list[Path] = field(default_factory=list)
    failed_paths: list[Path] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
