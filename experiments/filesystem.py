"""Local filesystem primitives for POSIX and Windows experiment stores."""
import contextlib
import os


def sync_directory(path):
    # Windows does not support opening directories through os.open. File contents
    # are still fsynced; directory-entry crash durability is POSIX-only.
    if os.name == 'nt':
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextlib.contextmanager
def exclusive_lock(path):
    with open(path, 'a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            # msvcrt locks a byte range starting at the current position.
            if stream.seek(0, os.SEEK_END) == 0:
                stream.write(b'\0')
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)
