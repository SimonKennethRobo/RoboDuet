"""Best-effort training output: storage failures must not stop optimization."""
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path


_last_warning = {}


@contextmanager
def optional_output(label):
    """Skip a failed output operation, retry naturally on its next invocation."""
    try:
        yield
    except (OSError, RuntimeError) as error:
        # PyTorch's C++ zip writer reports filesystem errors as RuntimeError.
        # Do not hide unrelated tensor, TorchScript or serialization failures.
        if not isinstance(error, OSError) and not any(
            marker in str(error) for marker in (
                "PytorchStreamWriter failed", "unexpected pos",
                "cannot be opened",
            )
        ):
            raise
        now = time.monotonic()
        if now - _last_warning.get(str(label), float('-inf')) >= 60:
            _last_warning[str(label)] = now
            try:
                print(f"[WARNING] Skipping output {label}: {error}. "
                      "Training continues; output will be attempted next time.", file=sys.stderr)
            except OSError:
                pass  # stderr itself may be redirected to the failed filesystem.


def atomic_output(path, write):
    """Publish a completed file without overwriting the last good file on failure.

    The caller controls whether an error is fatal via optional_output().
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.stem}.', suffix=path.suffix, dir=path.parent)
    os.close(fd)
    try:
        write(temporary)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except OSError:
            pass
