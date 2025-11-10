# llm_engineering/application/log_setup.py
import sys
from loguru import logger
from llm_engineering.application.settings import get_settings

def setup_logging() -> None:
    """Configure Loguru once, based on Settings.debug."""
    settings = get_settings()

    logger.remove()  # remove default handler(s) to avoid duplicates on reload
    logger.add(
        sys.stdout,
        level="DEBUG" if settings.debug else "INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        backtrace=False,
        diagnose=False,
    )
