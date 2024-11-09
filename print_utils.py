from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

CONSOLE = Console()

logging.getLogger("httpx").setLevel(logging.WARNING)


# Configure logging with Rich, using the shared console
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=CONSOLE)],
)
LOGGER = logging.getLogger("rich")
