"""Small helpers for what is safe to show a user and safe file writes."""
import os
import re
import shutil
import threading
import time
from pathlib import Path

_PATH = re.compile(r"(?:[A-Za-z]:)?[\\/][^\s'\"]*[\\/][^\s'\"]+")


def friendly(e, limit=600) -> str:
    """An error message for the user: trimmed, with any server file paths removed."""
    return _PATH.sub("[file]", str(e)[-limit:])


def atomic_write(target_path, text: str, encoding="utf-8", chmod=None):
    """Atomically write text to a file, resilient against Windows WinError 5 file locking."""
    target = Path(target_path)
    # Unique temp file per thread/process to avoid cross-thread collisions on the temp file itself
    tmp = target.with_name(f".{target.name}.{os.getpid()}_{threading.get_ident()}_{time.time_ns()}.tmp")
    try:
        tmp.write_text(text, encoding=encoding)
        if chmod is not None:
            try:
                tmp.chmod(chmod)
            except OSError:
                pass
        for attempt in range(10):
            try:
                tmp.replace(target)
                return
            except OSError:
                if attempt == 9:
                    try:
                        shutil.copyfile(tmp, target)
                        return
                    except Exception:
                        pass
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        tmp.unlink(missing_ok=True)


