"""Bound local asset inventory before hashing; never acquire or deserialize assets."""
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat


@dataclass(frozen=True)
class AssetLimits:
    max_entries: int = 10000
    max_files: int = 10000
    max_depth: int = 32
    max_path_bytes: int = 4096
    max_file_bytes: int = 16 * 1024**3
    max_total_bytes: int = 32 * 1024**3
    max_total_hash_bytes: int = 32 * 1024**3
    small_hash_bytes: int = 8 * 1024**2

    def __post_init__(self):
        if any(type(value) is not int or value < 1 for value in self.__dict__.values()):
            raise ValueError('Asset budgets must be explicit positive integers')


DEFAULT_ASSET_LIMITS = AssetLimits()


def _fingerprint(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns)


def _hash_file(path, expected):
    """Read at most the admitted bytes plus a growth sentinel, on one regular fd."""
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or _fingerprint(before) != _fingerprint(expected):
            raise ValueError(f'Asset changed after size admission: {path}')
        digest = hashlib.sha256()
        consumed = 0
        while True:
            chunk = stream.read(min(1024**2, expected.st_size - consumed + 1))
            if not chunk:
                break
            consumed += len(chunk)
            if consumed > expected.st_size:
                raise ValueError(f'Asset grew beyond admitted size budget: {path}')
            digest.update(chunk)
        if consumed != expected.st_size or _fingerprint(os.fstat(stream.fileno())) != _fingerprint(before):
            raise ValueError(f'Asset changed while hashing: {path}')
    return {'sha256': digest.hexdigest(), 'status': 'HASHED', 'size_bytes': consumed}


def inventories(sources, *, hash_large=False, limits=DEFAULT_ASSET_LIMITS):
    """Share count, traversal, source-byte and hashing budgets across all sources.

    Enumerate with streaming scandir, admit the complete operation, then hash.
    Directory symlinks are not traversed; file symlinks must stay inside a source
    directory. A directly selected file is an explicit source, not a directory.
    Missing sources retain empty inventories for preflight diagnostics.
    """
    if type(hash_large) is not bool or not isinstance(limits, AssetLimits):
        raise ValueError('Boolean hash policy and typed AssetLimits required')
    entries = file_count = source_bytes = hash_bytes = 0
    admitted = []
    for source in sources:
        selected = Path(source)
        root = selected.resolve()
        files = []

        def admit_file(path, key, info):
            nonlocal file_count, source_bytes, hash_bytes
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(f'Only regular asset files are supported: {path}')
            file_count += 1
            source_bytes += info.st_size
            if file_count > limits.max_files:
                raise ValueError('Asset file count exceeds budget')
            if len(key.encode('utf-8', errors='strict')) > limits.max_path_bytes:
                raise ValueError('Asset relative path exceeds byte budget')
            if info.st_size > limits.max_file_bytes or source_bytes > limits.max_total_bytes:
                raise ValueError('Asset file/total size exceeds byte budget')
            should_hash = hash_large or info.st_size <= limits.small_hash_bytes
            hash_bytes += info.st_size if should_hash else 0
            if hash_bytes > limits.max_total_hash_bytes:
                raise ValueError('Asset hashing exceeds aggregate byte budget')
            files.append((path, key, info, should_hash))

        if root.is_file():
            entries += 1
            if entries > limits.max_entries:
                raise ValueError('Asset discovery exceeds entry budget')
            admit_file(root, selected.name, root.stat())
        elif root.is_dir():
            pending = [(root, 0)]
            while pending:
                directory, depth = pending.pop()
                if not directory.resolve().is_relative_to(root):
                    raise ValueError('Asset directory symlink escapes its root')
                with os.scandir(directory) as children:
                    for child in children:
                        entries += 1
                        if entries > limits.max_entries:
                            raise ValueError('Asset discovery exceeds entry budget')
                        path = Path(child.path)
                        target = path.resolve()
                        if not target.is_relative_to(root):
                            raise ValueError('Asset symlink escapes its root')
                        if child.is_dir(follow_symlinks=False):
                            if depth + 1 > limits.max_depth:
                                raise ValueError('Asset traversal exceeds depth budget')
                            pending.append((path, depth + 1))
                        elif child.is_symlink() and target.is_dir():
                            continue
                        else:
                            admit_file(path, str(path.relative_to(root)), child.stat())
        elif root.exists():
            raise ValueError('Only regular asset files/directories are supported')
        admitted.append(files)

    # No payload reads before every region has passed admission.
    return [{key: _hash_file(path, info) if should_hash else
             {'sha256': None, 'status': 'SKIPPED_SIZE_LIMIT', 'size_bytes': info.st_size}
             for path, key, info, should_hash in sorted(files, key=lambda item: item[1])}
            for files in admitted]
