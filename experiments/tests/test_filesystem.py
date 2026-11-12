import multiprocessing
import time

import pytest

from experiments.filesystem import exclusive_lock
from experiments.store import publish


def _hold_lock(path, ready):
    with exclusive_lock(path):
        ready.set()
        time.sleep(.3)


def test_lock_serializes_spawned_processes(tmp_path):
    ctx = multiprocessing.get_context('spawn')
    ready = ctx.Event()
    path = str(tmp_path / '.lock')
    process = ctx.Process(target=_hold_lock, args=(path, ready))
    process.start()
    try:
        assert ready.wait(10)
        start = time.monotonic()
        with exclusive_lock(path):
            assert time.monotonic() - start > .1
    finally:
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join()
    assert process.exitcode == 0


def test_publication_preserves_existing_file(tmp_path):
    path = tmp_path / 'record'
    publish(path, lambda stream: stream.write(b'original'))
    with pytest.raises(FileExistsError):
        publish(path, lambda stream: stream.write(b'replacement'))
    assert path.read_bytes() == b'original'
    assert not list(tmp_path.glob('.pending-*'))
