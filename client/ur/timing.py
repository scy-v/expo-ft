"""Opt-in host timing. Bounded asynchronous JSONL; never records sensor payloads."""
import atexit
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import itertools
import json
import logging
import os
from pathlib import Path
import queue
import threading
import time

_context = ContextVar('timing_context', default={})
_writer = None
_ids = itertools.count()


class TraceWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('a', buffering=1)
        self.queue = queue.Queue(maxsize=16384)
        self.dropped = 0
        self.error = None
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._run, name='timing-jsonl', daemon=True)
        self.thread.start()

    def emit(self, row):
        if self.error or self.stopping.is_set():
            return
        try:
            self.queue.put_nowait(row)
        except queue.Full:
            self.dropped += 1

    def _run(self):
        try:
            while not self.stopping.is_set() or not self.queue.empty():
                try:
                    row = self.queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                row['dropped_events'] = self.dropped
                self.stream.write(json.dumps(row, ensure_ascii=False)+'\n')
        except Exception as error:
            self.error = str(error)
            logging.exception('Timing diagnostic writer failed; training continues')
        finally:
            self.stream.close()

    def close(self):
        self.stopping.set()
        self.thread.join(timeout=3)


def configure(enabled, directory, role):
    global _writer
    if type(enabled) is not bool:
        raise ValueError('timing_diagnostics must be a YAML boolean')
    if not enabled or _writer is not None:
        return
    path = Path(directory)/f'{role}-{time.time_ns()}-{os.getpid()}.jsonl'
    _writer = TraceWriter(path)
    atexit.register(_writer.close)
    logging.info('Detailed timing diagnostics: %s', path)


def enabled():
    return _writer is not None


def set_context(**fields):
    _context.set(fields)


@contextmanager
def span(name, **fields):
    writer = _writer
    if writer is None:
        yield
        return
    token = _context.set({**_context.get(), **fields})
    start = time.perf_counter_ns()
    cpu_start = time.thread_time_ns()
    common = dict(_context.get(), span=name, id=next(_ids), pid=os.getpid(),
                  thread=threading.current_thread().name)
    writer.emit(dict(common, event='begin', wall_ns=time.time_ns(), monotonic_ns=start))
    error = None
    try:
        yield
    except BaseException as exc:
        error = type(exc).__name__
        raise
    finally:
        end = time.perf_counter_ns()
        writer.emit(dict(common, event='end', wall_ns=time.time_ns(), monotonic_ns=end,
                         duration_ms=(end-start)/1e6,
                         thread_cpu_ms=(time.thread_time_ns()-cpu_start)/1e6, error=error))
        _context.reset(token)


def traced(name):
    def decorate(fn):
        @wraps(fn)
        def call(*args, **kwargs):
            if _writer is None:
                return fn(*args, **kwargs)
            with span(name):
                return fn(*args, **kwargs)
        return call
    return decorate


def transform_pipeline(pipeline, data, name):
    """Trace individual OpenPI CompositeTransform stages without changing their order."""
    if not enabled():
        return pipeline(data)
    with span(name, transform=type(pipeline).__name__):
        stages = getattr(pipeline, 'transforms', None)
        if stages is None:
            return pipeline(data)
        for index, transform in enumerate(stages):
            data = transform_pipeline(transform, data, f'{name}.{index}.{type(transform).__name__}')
        return data
