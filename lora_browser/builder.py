"""Build visual-only and audio-only copies, and baked groups, one at a time.

A copy is seconds of reading and writing; a bake can be a minute of linear
algebra on a large LoRA.  Neither belongs in a Gradio event, so a worker thread
takes them in the order asked for and the panel polls :meth:`Builder.status`.
Nothing here knows about Gradio: :meth:`Builder.run_pending` drives the same
loop synchronously, which is how the tests run it.

A build writes to a temporary name and appears under its own only once it is
complete, so WanGP never loads half a file.  If WanGP is asked to generate
before a build finishes, the file is simply not there yet and WanGP says so.
"""

from __future__ import annotations

import os
import threading
import traceback
from dataclasses import dataclass, field
from typing import Callable

from . import bake as bk
from . import lora_files as lf
from .derived import Store


@dataclass
class Job:
    #: The built file's id; at most one job per id is queued or running.
    key: str
    #: What the progress bar says it is building.
    label: str
    #: Does the work.  Called with a progress callback taking 0..1.
    work: Callable[[Callable[[float], None]], None]
    store: Store | None = None


@dataclass
class _State:
    queue: list[Job] = field(default_factory=list)
    current: Job | None = None
    progress: float = 0.0
    failures: dict[str, str] = field(default_factory=dict)
    #: Files finished since the queue last ran dry.
    finished: int = 0
    #: Bumped on every completion, so a poller can tell something new is on disk.
    generation: int = 0


class Builder:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = _State()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------ queue

    def submit(self, job: Job, start: bool = True) -> bool:
        """Queue ``job`` unless the same file is already queued or building.

        A file that failed is not retried until asked for again
        (:meth:`retry`): a missing original would otherwise fail on every
        payload.
        """
        with self._lock:
            state = self._state
            if job.key in state.failures:
                return False
            if (state.current and state.current.key == job.key) or any(item.key == job.key for item in state.queue):
                return False
            if state.current is None and not state.queue:
                state.finished = 0            # a new batch
            state.queue.append(job)
        if start:
            self._start()
        return True

    def retry(self, key: str) -> None:
        with self._lock:
            self._state.failures.pop(key, None)

    def cancel(self, key: str) -> None:
        """Drop a queued build that is no longer wanted.  One already running
        finishes: a half-written file is never left behind."""
        with self._lock:
            self._state.queue = [job for job in self._state.queue if job.key != key]

    def busy(self, key: str) -> str:
        """``building``, ``queued``, ``failed`` or ``""``."""
        with self._lock:
            state = self._state
            if state.current and state.current.key == key:
                return "building"
            if any(job.key == key for job in state.queue):
                return "queued"
            if key in state.failures:
                return "failed"
            return ""

    def failure(self, key: str) -> str:
        with self._lock:
            return self._state.failures.get(key, "")

    def progress(self, key: str) -> float:
        with self._lock:
            state = self._state
            return state.progress if state.current and state.current.key == key else 0.0

    def status(self) -> dict:
        with self._lock:
            state = self._state
            current = state.current
            return {
                "running": current is not None or bool(state.queue),
                "current": {"key": current.key, "label": current.label, "pct": round(state.progress * 100)}
                if current else None,
                "queued": [{"key": job.key, "label": job.label} for job in state.queue],
                "failed": dict(state.failures),
                "finished": state.finished,
                "generation": state.generation,
            }

    # ------------------------------------------------------------- loop

    def _start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self.run_pending, name="lora-browser-build", daemon=True)
            self._thread.start()

    def run_pending(self) -> None:
        """Build everything queued, in order, then return."""
        while True:
            with self._lock:
                state = self._state
                if not state.queue:
                    state.current = None
                    return
                job = state.queue.pop(0)
                state.current = job
                state.progress = 0.0

            def report(fraction: float, key=job.key) -> None:
                with self._lock:
                    if self._state.current and self._state.current.key == key:
                        self._state.progress = max(0.0, min(1.0, float(fraction)))

            error = ""
            try:
                job.work(report)
            except Exception as exc:          # a bad file must not stop the queue
                traceback.print_exc()
                error = str(exc) or exc.__class__.__name__
            with self._lock:
                state = self._state
                state.current = None
                if error:
                    state.failures[job.key] = error
                else:
                    state.finished += 1
                    state.generation += 1


# ------------------------------------------------------------------ jobs


def split_job(store: Store, derived_id: str, source_id: str, source_path: str, mode: str, label: str) -> Job:
    """A visual-only or audio-only copy of one LTX-2 LoRA."""

    def work(report) -> None:
        lora = lf.read_header(source_path)
        keep = [
            tensor.name for tensor in lora.tensors
            if lf.keeps(lf.ltx2_side(lf.split_key(tensor.name)[0]), mode)
        ]
        if not any(lf.ltx2_side(lf.split_key(name)[0]) != lf.SHARED for name in keep):
            raise ValueError(f"{label}: nothing to keep in {mode} only")
        stamp = store.source_stamp(source_id)
        metadata = dict(lora.metadata)
        metadata.update({"lora_browser": mode, "lora_browser_source": source_id})
        digest = lf.copy_subset(
            lora, keep, store.write_path(derived_id), metadata,
            progress=lambda done, total: report(done / total if total else 1.0),
        )
        store.record(derived_id, {
            "kind": "split", "mode": mode, "sources": [dict(stamp, sha256=digest)],
            "bytes": os.path.getsize(store.write_path(derived_id)), "label": label,
        })

    return Job(key=derived_id, label=f"{label}, {mode} only", work=work, store=store)


def bake_job(store: Store, derived_id: str, members: list[tuple[str, str, str, str]], rank: int,
             sided: bool, label: str) -> Job:
    """A group baked into one file.  ``members`` is
    ``(original id, path, mode, display name)`` per LoRA."""

    def work(report) -> None:
        plan = bk.plan([bk.Member(path, mode, name) for _, path, mode, name in members], sided=sided)
        stamps = [store.source_stamp(lora_id) for lora_id, _, _, _ in members]
        result = bk.bake(
            plan, rank, store.write_path(derived_id),
            {"lora_browser_members": ", ".join(name for _, _, _, name in members)},
            progress=lambda done, total: report(done / total if total else 1.0),
        )
        store.record(derived_id, {
            "kind": "bake", "rank": result.rank, "kept": result.kept, "exact": result.exact,
            "sources": stamps,
            "members": [{"id": lora_id, "mode": mode, "name": name} for lora_id, _, mode, name in members],
            "bytes": os.path.getsize(store.write_path(derived_id)), "label": label,
        })

    return Job(key=derived_id, label=f"{label}, baked at rank {int(rank)}", work=work, store=store)
