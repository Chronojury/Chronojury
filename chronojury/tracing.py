"""Optional per-case JSONL records, independent of the published result schema."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import hashlib
import inspect
import json
import os
from pathlib import Path
import threading
import time

_context = ContextVar("chronojury_trace", default={})
_write_lock = threading.Lock()


@contextmanager
def trace_scope(**fields):
    token = _context.set({**_context.get(), **fields})
    try:
        yield
    finally:
        _context.reset(token)


def _recordable(value):
    if isinstance(value, dict):
        return {str(k): _recordable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_recordable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str) and value.startswith("data:"):
        return {"media_type": value.split(";", 1)[0][5:],
                "sha256": hashlib.sha256(value.encode()).hexdigest(),
                "encoded_characters": len(value)}
    return value


def record_event(event, **data):
    root = os.environ.get("CHRONOJURY_TRACE_DIR")
    if not root:
        return
    context = _context.get()
    case_id = str(context.get("case_id", "requests"))
    if Path(case_id).name != case_id:
        raise ValueError("Trace case ID must be a filename component")
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_recordable({"time": time.time(), **context, "event": event, **data}),
                      ensure_ascii=False) + "\n"
    with _write_lock, (directory / f"{case_id}.jsonl").open("a", encoding="utf-8") as file:
        file.write(line)


def trace_case(operation, case_arg=0):
    def decorate(function):
        signature = inspect.signature(function)
        argument = list(signature.parameters)[case_arg]
        @wraps(function)
        def wrapped(*args, **kwargs):
            if not os.environ.get("CHRONOJURY_TRACE_DIR"):
                return function(*args, **kwargs)
            case = signature.bind(*args, **kwargs).arguments[argument]
            with trace_scope(case_id=Path(case).name, operation=operation):
                record_event("case_start")
                result = function(*args, **kwargs)
                record_event("case_end", result=result)
                return result
        return wrapped
    return decorate


def trace_operation(operation):
    def decorate(function):
        signature = inspect.signature(function)
        @wraps(function)
        def wrapped(*args, **kwargs):
            if not os.environ.get("CHRONOJURY_TRACE_DIR"):
                return function(*args, **kwargs)
            bound = signature.bind(*args, **kwargs).arguments
            fields = {k: bound[k] for k in ("phase", "current_iteration", "case_id") if k in bound}
            if args and hasattr(args[0], "name"):
                fields["agent"] = args[0].name
            with trace_scope(operation=operation, **fields):
                result = function(*args, **kwargs)
                record_event("operation_result", result=result)
                return result
        return wrapped
    return decorate
