import contextlib
import logging
import sys
import time
from shutil import which as _which

from rich.console import Console

_console = Console()


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


@contextlib.contextmanager
def step_status(message: str, spinner: bool = True):
    """
    Show a spinner (if rich is available) and log start/end with duration.
    Disable spinner when it would clash with other progress bars (e.g., tqdm).
    """
    console = _console
    start = time.time()
    status = None
    if console.is_interactive and spinner:
        status = console.status(f"[bold cyan]{message}...", spinner="dots")
        status.__enter__()
    else:
        logging.info("%s...", message)
    try:
        yield
    finally:
        elapsed = time.time() - start
        if status:
            status.__exit__(None, None, None)
            console.log(f"{message} done in {elapsed:.1f}s")
        else:
            logging.info("%s done in %.1fs", message, elapsed)
