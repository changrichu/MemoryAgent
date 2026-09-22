"""
日志工具
"""
import sys
from loguru import logger as _logger

from ..config import settings


def get_logger(name: str = "ragagent"):
    """配置并返回 logger"""
    _logger.remove()
    _logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )
    return _logger
