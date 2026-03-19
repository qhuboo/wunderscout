class WunderscoutError(Exception):
    """Base exception for wunderscout errors."""

    pass


class CalibrationError(WunderscoutError):
    """Raised when calibration fails (no detections, bad crops, etc)."""

    pass


class VideoProcessingError(WunderscoutError):
    """Raised when video cannot be processed."""

    pass
