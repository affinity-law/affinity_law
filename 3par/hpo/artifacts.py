"""Artifact storage: immutable inputs, atomic results, and a single-process lock."""

import json
import os
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    temporary.write_bytes(content)
    os.replace(temporary, path)


def save_json(path, value):
    atomic_bytes(path, json_text(value).encode("utf-8"))


def update_json(path, **sections):
    """Merge record sections atomically while holding the run lock."""
    path = Path(path)
    value = read_json(path) if path.exists() else {}
    value.update(sections)
    save_json(path, value)


def freeze_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Frozen file differs from the current input; use a new run name: {path}")
    else:
        save_json(path, value)


def copy_verified(source, destination):
    """Copy only new files and never overwrite existing experimental evidence."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        if destination.read_bytes() != source.read_bytes():
            raise ValueError(f"Copy destination has different content and was not overwritten: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    if destination.read_bytes() != source.read_bytes():
        raise IOError(f"Copy verification failed: {destination}")


def next_attempt(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    ids = [int(p.name[8:]) for p in folder.glob("attempt_*") if p.name[8:].isdigit()]
    attempt = folder / f"attempt_{max(ids, default=0) + 1:03d}"
    attempt.mkdir(exist_ok=False)
    return attempt


@contextmanager
def run_lock(folder):
    """OS lock that is released on crashes without deleting the lock file."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".run.lock").open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            lock = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            unlock = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            lock = lambda: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            unlock = lambda: fcntl.flock(handle, fcntl.LOCK_UN)
        try:
            lock()
        except OSError as exc:
            raise RuntimeError(f"Another process is already using this run directory: {folder}") from exc
        try:
            yield
        finally:
            handle.seek(0)
            unlock()
