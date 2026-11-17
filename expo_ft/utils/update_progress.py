"""Live learner status on the rollout tqdm bar; never waits on GPU work."""
import threading
import time


# A zero-width bar keeps tqdm's ncols clipping active (without {bar}, tqdm
# returns an unbounded string even when dynamic_ncols=True).
BAR_FORMAT = 'step {n_fmt}/{total_fmt} | {postfix}{bar:0}'


class UpdateProgress:
    def __init__(self):
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._completed = 0
        self._stage = 'waiting_buffer'
        self._started = None
        self._last = None

    def stage(self, name):
        with self._lock:
            self._stage = name

    def begin(self):
        with self._lock:
            self._stage = 'updating'
            self._started = time.monotonic()

    def complete(self, count):
        with self._lock:
            self._last = time.monotonic() - self._started
            self._started = None
            self._completed = count
            self._stage = 'idle'

    def snapshot(self):
        with self._lock:
            elapsed = time.monotonic() - self._started if self._started is not None else None
            return dict(net_updates=self._completed, learner=self._stage,
                        update_s=f'{elapsed:.1f}' if elapsed is not None else '-',
                        last_s=f'{self._last:.1f}' if self._last is not None else '-')

    def start(self, bar):
        if bar.disable:
            return

        def refresh():
            while not self._stop.is_set():
                status = self.snapshot()
                state = status['learner']
                current = (f"run={status['update_s']}s" if state == 'updating'
                           else state.replace('preparing_batch', 'batch').replace('waiting_buffer', 'buffer'))
                text = f"updates={status['net_updates']} {current} last={status['last_s']}s"
                bar.set_postfix_str(text, refresh=True)
                self._stop.wait(0.5)

        self._thread = threading.Thread(target=refresh, name='learner-progress', daemon=True)
        self._thread.start()

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
