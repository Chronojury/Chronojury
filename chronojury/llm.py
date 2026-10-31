"""OpenAI-compatible requests to a hosted API or replicated model endpoints."""
import itertools
import os
import threading
import time
from uuid import uuid4

import openai
from .swer import config
from .tracing import record_event
from .runtime import runtime_options

_sequence = itertools.count()
_lock = threading.Lock()
_available = threading.Condition(_lock)
_routes = {}


def _acquire(endpoints):
    """Balance active requests and temporarily exclude failed replicas."""
    while True:
        capacities = runtime_options().get('api_capacities', {})
        with _available:
            now = time.monotonic()
            for endpoint in endpoints:
                _routes.setdefault(endpoint, {"active": 0, "retry_at": 0, "failures": 0})
            available = [e for e in endpoints if _routes[e]["retry_at"] <= now
                         and _routes[e]['active'] < capacities.get(e, float('inf'))]
            if available:
                offset = next(_sequence) % len(available)
                available = available[offset:] + available[:offset]
                endpoint = min(available, key=lambda e: _routes[e]["active"] / capacities.get(e, 1))
                _routes[endpoint]["active"] += 1
                return endpoint
            _available.wait(timeout=1)


def _release(endpoint, failed):
    with _lock:
        state = _routes[endpoint]
        state["active"] -= 1
        if failed:
            state["failures"] += 1
            state["retry_at"] = time.monotonic() + min(120, 5 * 2 ** min(state["failures"], 5))
        else:
            state["failures"] = 0
            state["retry_at"] = 0
        _available.notify_all()


def _retryable(error):
    return isinstance(error, (openai.error.APIConnectionError, openai.error.Timeout,
                              openai.error.RateLimitError, openai.error.ServiceUnavailableError)) or (
        isinstance(error, openai.error.OpenAIError)
        and getattr(error, "http_status", None) in (404, 408, 429, 500, 502, 503, 504)
    )


def chat_completion(**kwargs):
    """Route requests without changing model messages or sampling parameters.

    CHRONOJURY_API_BASES is a comma-separated list of equivalent replicas.
    Without it, OPENAI_BASE_URL selects a single OpenAI-compatible endpoint.
    Credentials and routing are request-local, including under concurrency.
    """
    bases = os.environ.get("CHRONOJURY_API_BASES") or os.environ.get(
        "OPENAI_BASE_URL", "http://localhost:8000/v1"
    )
    endpoints = [base.strip().rstrip("/") for base in bases.split(",") if base.strip()]
    if runtime_options().get('api_bases'):
        endpoints = [base.strip().rstrip('/') for base in runtime_options()['api_bases']]
    if not endpoints:
        raise ValueError("Configure at least one OpenAI-compatible API endpoint.")
    explicit_endpoint = kwargs.pop("api_base", None)
    if explicit_endpoint:
        endpoints = [explicit_endpoint.rstrip("/")]
    kwargs.setdefault("api_key", os.environ.get("OPENAI_API_KEY", "EMPTY"))
    if kwargs.get("model", "").rsplit("/", 1)[-1].lower().startswith("qwen"):
        kwargs["chat_template_kwargs"] = {
            **config.QWEN_CHAT_TEMPLATE_KWARGS,
            **(kwargs.get("chat_template_kwargs") or {}),
        }
    # In openai 0.28, request_timeout controls transport; timeout is a body field.
    timeout = kwargs.pop("timeout", None)
    kwargs.setdefault("request_timeout", timeout or float(os.environ.get("CHRONOJURY_REQUEST_TIMEOUT", "60")))
    floor = runtime_options().get('request_timeout_floor', 0)
    if floor and isinstance(kwargs['request_timeout'], (int, float)):
        kwargs['request_timeout'] = max(floor, kwargs['request_timeout'])
    request_id = uuid4().hex
    record_event("llm_request", request_id=request_id, request={
        k: v for k, v in kwargs.items()
        if k not in ("api_key", "api_base", "headers", "organization", "api_type", "api_version")
    })
    while True:
        endpoint = _acquire(endpoints)
        started = time.monotonic()
        failed = False
        try:
            response = openai.ChatCompletion.create(api_base=endpoint, **kwargs)
        except Exception as error:
            # Replicas can expose different context limits. Retry the unchanged
            # request on the remaining replicas, but surface it if none can fit.
            context_limit = getattr(error, "http_status", None) == 400 and any(
                text in str(error).lower() for text in
                ("maximum context length", "max_model_len", "context length", "context window")
            )
            if context_limit and len(endpoints) > 1:
                endpoints = [e for e in endpoints if e != endpoint]
            failed = _retryable(error)
            failed = failed or (context_limit and endpoint not in endpoints)
            record_event("llm_error", request_id=request_id, endpoint=endpoint,
                         error_type=type(error).__name__, http_status=getattr(error, "http_status", None),
                         retrying=failed)
            if not failed:
                raise
            if isinstance(error, openai.error.Timeout) and isinstance(kwargs['request_timeout'], (int, float)):
                maximum = runtime_options().get('request_timeout_max', kwargs['request_timeout'])
                kwargs['request_timeout'] = min(maximum, 2 * kwargs['request_timeout'])
        finally:
            _release(endpoint, failed)
        if failed:
            continue
        record_event("llm_response", request_id=request_id, endpoint=endpoint,
                     seconds=time.monotonic() - started, response=response)
        return response
