"""Optional live scheduling settings, separate from model sampling parameters."""
import json
import os
from contextlib import contextmanager
import hashlib
from pathlib import Path
import threading
import time

MAX_CASE_WORKERS = 512
_lock = threading.Lock()
_path = None
_checked_at = 0.0
_options = {}


def runtime_options():
    """Read an atomically replaced local JSON control file at most once a second."""
    global _path, _checked_at, _options
    path = os.environ.get('CHRONOJURY_RUNTIME_CONFIG')
    if not path:
        return {}
    with _lock:
        now = time.monotonic()
        if path == _path and now - _checked_at < 1:
            return _options
        options = json.loads(Path(path).read_text(encoding='utf-8'))
        workers = options.get('reason_workers', 1)
        if not isinstance(workers, int) or not 1 <= workers <= MAX_CASE_WORKERS:
            raise ValueError(f'reason_workers must be between 1 and {MAX_CASE_WORKERS}')
        if options.get('prefetch_cases', 0) < 0 or options.get('request_timeout_floor', 0) < 0:
            raise ValueError('Prefetch count and request timeout must be nonnegative')
        if any(not isinstance(value, int) or value < 1
               for value in options.get('api_capacities', {}).values()):
            raise ValueError('API capacities must be positive integers')
        bases = options.get('api_bases')
        if bases is not None and (not isinstance(bases, list) or not bases
                                  or any(not isinstance(base, str) or not base.strip() for base in bases)):
            raise ValueError('api_bases must be a nonempty list of endpoint URLs')
        _path, _checked_at, _options = path, now, options
        return options


@contextmanager
def case_lock(output_dir, case_id):
    """Coordinate resumed workers that share case outputs on one machine."""
    root = runtime_options().get('case_lock_dir')
    if not root:
        yield
        return
    import fcntl
    group = hashlib.sha256(str(Path(output_dir).resolve()).encode()).hexdigest()[:16]
    directory = Path(root) / group
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f'{case_id}.lock').open('a') as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        yield
