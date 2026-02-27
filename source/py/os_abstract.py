"""OS abstractions."""

import os
from pathlib import Path
from subprocess import run
from sys import platform


def open_with_default_app(target: Path | str):
    """Open a directory, file or URL using system's default app.

    Args:
        target: Directory, file or URL to open.

    May raise an exception.
    """
    match platform:
        case "linux":
            run(["xdg-open", target], check=True)
        case "darwin":  # macOS
            run(["open", target], check=True)
        case "win32":
            os.startfile(target)
        case _:
            raise OSError(f"Unsupported platform: {platform}")
