import logging
from .models import Models
from .geometry import PitchMapper
from .teams import TeamClassifier
from .core import Detector
from .heatmaps import HeatmapGenerator, Heatmap, HeatmapKey

logging.getLogger("wunderscout").addHandler(logging.NullHandler())

__all__ = [
    "Models",
    "PitchMapper",
    "TeamClassifier",
    "Detector",
    "HeatmapGenerator",
    "Heatmap",
    "HeatmapKey",
]


def set_stream_logger(name="wunderscout", level=logging.DEBUG, format_string=None):
    """
    Add a stream handler for the given name and level to the logging module.
    By default, this logs all wunderscout messages to ``stdout``.

        >>> import wunderscout
        >>> wunderscout.set_stream_logger('wunderscout', logging.INFO)

    For debugging purposes a good choice is to set the stream logger to ``''``
    which is equivalent to saying "log everything".

    :type name: string
    :param name: Log name
    :type level: int
    :param level: Logging level, e.g. ``logging.INFO``
    :type format_string: str
    :param format_string: Log message format
    """
    if format_string is None:
        format_string = "[%(asctime)s] [%(levelname)s] [%(module)s] [%(funcName)s] [%(lineno)d] %(message)s"

    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
