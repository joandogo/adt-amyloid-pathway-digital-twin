"""Periodic progress display for manual study and archive-review scripts."""

import shutil
import sys
import threading
import time
from typing import TextIO


class ProgressBar:
    """Report completed items and the active phase during long operations."""

    def __init__(self, total: int, *, label: str, unit: str,
                 interval: float = 10., stream: TextIO | None = None) -> None:
        if total < 1 or not 0 < interval < float('inf'):
            raise ValueError('Progress total and interval must be positive and finite.')
        self.total = total
        self.label = label
        self.unit = unit
        self.interval = interval
        self.stream = stream if stream is not None else sys.stderr
        self.done = 0
        self.phase = 'Preparando'
        self.started = time.monotonic()
        self.last_width = 0
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None

    def _render(self) -> None:
        with self.lock:
            elapsed = int(time.monotonic() - self.started)
            duration = f'{elapsed // 3600:02d}:{elapsed // 60 % 60:02d}:{elapsed % 60:02d}'
            filled = 16 * self.done // self.total
            prefix = f'{self.label} [{"#" * filled}{"-" * (16 - filled)}] {self.done}/{self.total} {self.unit} | '
            suffix = f' | {duration}'
            if self.stream.isatty():
                width = shutil.get_terminal_size(fallback=(120, 24)).columns
                available = max(0, width - len(prefix) - len(suffix) - 1)
                phase = self.phase[:available]
                line = prefix + phase + suffix
                self.stream.write('\r' + line.ljust(self.last_width))
                self.last_width = max(self.last_width, len(line))
            else:
                self.stream.write(prefix + self.phase + suffix + '\n')
            self.stream.flush()

    def _heartbeat(self) -> None:
        while not self.stop.wait(self.interval):
            self._render()

    def start(self) -> None:
        self._render()
        self.thread = threading.Thread(target=self._heartbeat, name='adt-progress', daemon=True)
        self.thread.start()

    def stage(self, phase: str) -> None:
        with self.lock:
            self.phase = phase
        self._render()

    def advance(self, phase: str) -> None:
        with self.lock:
            self.done += 1
            if self.done > self.total:
                raise ValueError('Progress advanced beyond the declared total.')
            self.phase = phase
        self._render()
        if self.stream.isatty():
            with self.lock:
                self.stream.write('\n')
                self.stream.flush()
                self.last_width = 0

    def close(self) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=1.)
        if self.stream.isatty():
            with self.lock:
                self.stream.write('\n')
                self.stream.flush()
