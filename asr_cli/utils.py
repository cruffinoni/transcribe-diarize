import logging
import sys
from shutil import which as _which


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def fatal(msg: str, code: int = 1) -> None:
    logging.error(msg)
    sys.exit(code)


def which(cmd: str) -> str:
    path = _which(cmd)
    if not path:
        fatal(f"'{cmd}' not found on PATH. Install it (e.g. brew install {cmd}).")
    return path
