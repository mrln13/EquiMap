"""Run several P-T grids concurrently, each in its own process.

Why this exists
---------------
The MAGEMin backend talks to Julia through ``juliacall``.  MAGEMin's
``multi_point_minimization`` uses ``@threads :static``, which only works on
Julia's *main* thread.  That makes it impossible to run a MAGEMin grid from a
Qt ``QThread`` (it deadlocks), so the viewer historically ran grids
synchronously on the GUI main thread — freezing the UI, and forcing the
composition-vs-scale sweep (often a dozen grids) to run strictly serially.

Running each grid in a separate **process** sidesteps the constraint entirely:
inside a worker process Julia *is* on its own main thread, so ``@threads``
works.  A :class:`concurrent.futures.ProcessPoolExecutor` lets several grids
run at once, and — crucially — each worker boots Julia + MAGEMin only **once**
(in the pool initialiser) and then reuses that session for every grid it is
handed, amortising the ~20 s startup across the whole sweep.

This module is deliberately free of any Qt / viewer imports so it can be unit
tested (and pickled into worker processes) on its own.

Public API
----------
``iter_grid_results(jobs, backend_name, backend_kwargs, ...)`` — a generator
yielding ``(job_index, result_dicts, error)`` tuples as grids finish.  Results
are :func:`thermo.result.result_to_portable_dict` dicts (no unpicklable Julia
objects).  Raises :class:`SweepPoolBroken` if the worker pool itself dies
(e.g. Julia cannot initialise in a subprocess) so the caller can fall back to a
serial in-process run.
"""
from __future__ import annotations

import os
import tempfile
import time
from contextlib import contextmanager
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

__all__ = [
    "SweepPoolBroken",
    "default_worker_count",
    "iter_grid_results",
]


class SweepPoolBroken(RuntimeError):
    """Raised when the worker pool dies (typically a failed backend init)."""


# ---------------------------------------------------------------------------
# Worker-side state (lives in each subprocess, set by the pool initialiser)
# ---------------------------------------------------------------------------
_WORKER_BACKEND = None  # type: ignore[var-annotated]

# When several workers boot Julia at the same time they race on juliapkg's
# deps file (~/.julia/.../pythoncall/juliapkg.json) — one rewrites it while
# another reads, the reader sees an empty file and dies with JSONDecodeError,
# the pool goes BROKEN. Serialise that one resolve step with a cross-process
# lock file in TEMP. The lock is only held during _activate(); grid runs go
# fully parallel afterwards.
_JULIAPKG_LOCK_FILE = Path(tempfile.gettempdir()) / "thermod_juliapkg_init.lock"
_JULIAPKG_LOCK_STALE_S = 600.0  # if a lock file is older than this, ignore it


@contextmanager
def _juliapkg_init_lock(timeout_s: float = 600.0, poll_s: float = 0.5):
    """Cross-process exclusive lock around juliapkg deps-file resolution."""
    deadline = time.monotonic() + timeout_s
    fd: Optional[int] = None
    while True:
        try:
            fd = os.open(str(_JULIAPKG_LOCK_FILE),
                         os.O_CREAT | os.O_EXCL | os.O_RDWR)
            break
        except FileExistsError:
            try:
                age = time.time() - _JULIAPKG_LOCK_FILE.stat().st_mtime
                if age > _JULIAPKG_LOCK_STALE_S:
                    try:
                        _JULIAPKG_LOCK_FILE.unlink()
                    except OSError:
                        pass
                    continue
            except OSError:
                pass
            if time.monotonic() >= deadline:
                # Don't hard-fail: proceed without the lock so a stuck lockfile
                # never permanently breaks the sweep.
                fd = None
                break
            time.sleep(poll_s)
    try:
        yield
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                _JULIAPKG_LOCK_FILE.unlink()
            except OSError:
                pass


def _init_worker(backend_name: str,
                 backend_kwargs: Optional[Dict[str, Any]],
                 threads_per_worker: int) -> None:
    """Pool initialiser: build + activate the backend ONCE per worker process."""
    global _WORKER_BACKEND
    # JULIA_NUM_THREADS must be set before juliacall starts Julia.  Capping it
    # per worker keeps total threads ≈ CPU count when several workers run grids
    # concurrently (otherwise every worker would grab every core).
    if threads_per_worker and threads_per_worker > 0:
        os.environ["JULIA_NUM_THREADS"] = str(int(threads_per_worker))

    from thermo.calc import ThermoCalc

    calc = ThermoCalc(backend=backend_name, **(backend_kwargs or {}))
    backend = calc.backend
    # Eagerly pay the (expensive) activation cost here, in the worker's main
    # thread, so the first grid isn't penalised and so a broken Julia env fails
    # fast (-> BrokenProcessPool -> caller falls back to serial).
    if hasattr(backend, "_activate"):
        if backend_name == "magemin":
            with _juliapkg_init_lock():
                backend._activate()
        else:
            backend._activate()
    _WORKER_BACKEND = backend


def _run_one_job(job: Dict[str, Any]) -> List[Dict]:
    """Run a single grid in the worker process; return portable result dicts."""
    from thermo.result import result_to_portable_dict

    backend = _WORKER_BACKEND
    if backend is None:  # pragma: no cover - defensive; initialiser sets it
        raise RuntimeError("worker backend was not initialised")

    results = backend.run_grid(
        job["comp"],
        p_range=tuple(job["p_range"]),
        t_range=tuple(job["t_range"]),
        n_p=int(job["n_p"]),
        n_t=int(job["n_t"]),
    )
    return [result_to_portable_dict(r) for r in results]


# ---------------------------------------------------------------------------
# Driver (runs in the parent / orchestrating process)
# ---------------------------------------------------------------------------

def default_worker_count(n_jobs: int, cap: int = 4) -> int:
    """A conservative default: ~half the cores, capped, never more than jobs."""
    cores = os.cpu_count() or 2
    return max(1, min(n_jobs, cap, cores // 2 or 1))


def iter_grid_results(
    jobs: List[Dict[str, Any]],
    backend_name: str,
    backend_kwargs: Optional[Dict[str, Any]] = None,
    *,
    max_workers: Optional[int] = None,
    threads_per_worker: Optional[int] = None,
    cancel_flag: Optional[Callable[[], bool]] = None,
    use_cache: bool = True,
    cache: Optional[Any] = None,
    _executor_class: type = ProcessPoolExecutor,
) -> Iterator[Tuple[int, Optional[List[Dict]], Optional[str]]]:
    """Run *jobs* across a process pool, yielding results as they complete.

    Parameters
    ----------
    jobs :
        List of dicts, each with keys ``comp`` (a picklable
        :class:`thermo.composition.Composition`), ``p_range``, ``t_range``,
        ``n_p``, ``n_t``.
    backend_name, backend_kwargs :
        Forwarded to :class:`thermo.calc.ThermoCalc` inside each worker.
    max_workers :
        Number of concurrent worker processes.  Defaults to
        :func:`default_worker_count`.
    threads_per_worker :
        ``JULIA_NUM_THREADS`` for each worker.  Defaults to spreading the
        machine's cores evenly across the workers.
    cancel_flag :
        Optional callable polled as each grid completes; when it returns True
        no further results are yielded and pending grids are cancelled.
    use_cache, cache :
        Disk memoisation of grid results (see :mod:`thermo.cache`).  Cached
        grids are yielded immediately and never submitted to the pool; if
        *every* job is a cache hit the worker pool (and Julia) is never
        started at all.  Pass ``use_cache=False`` to force fresh computation
        (results are still stored), or a :class:`~thermo.cache.GridCache`
        instance via ``cache`` to use a non-default location.

    Yields
    ------
    ``(job_index, result_dicts, error)`` — exactly one of *result_dicts* /
    *error* is non-None.  ``job_index`` indexes into *jobs* (completion order
    is not submission order).

    Raises
    ------
    SweepPoolBroken
        If a worker process dies (commonly because the backend could not be
        initialised in a subprocess).  Already-yielded results stand; the
        caller should run the remaining jobs another way.
    """
    if not jobs:
        return

    # --- Cache pass: serve hits immediately, only submit the misses ----------
    gc = None
    if cache is not None:
        gc = cache
    elif use_cache:
        try:
            from thermo.cache import GridCache
            gc = GridCache()
        except Exception:
            gc = None

    # ``use_cache`` toggles reading (False forces fresh grids that are still
    # stored); a hit is yielded immediately and never submitted to the pool.
    keys: Dict[int, Optional[str]] = {}
    misses: List[int] = []
    for i, job in enumerate(jobs):
        key = None
        if gc is not None and getattr(gc, "enabled", False):
            try:
                key = gc.key_for(
                    backend_name, backend_kwargs, job["comp"],
                    job["p_range"], job["t_range"], job["n_p"], job["n_t"],
                )
            except Exception:
                key = None
            if use_cache and key is not None:
                try:
                    hit = gc.get(key)
                except Exception:
                    hit = None
                if hit is not None:
                    if cancel_flag is not None and cancel_flag():
                        return
                    yield (i, hit, None)
                    keys[i] = key
                    continue
        keys[i] = key
        misses.append(i)

    if not misses:
        return  # fully cached — no executor, no Julia

    n_workers = max_workers if max_workers and max_workers > 0 else default_worker_count(len(misses))
    n_workers = max(1, min(n_workers, len(misses)))
    if threads_per_worker is None or threads_per_worker <= 0:
        cores = os.cpu_count() or 2
        threads_per_worker = max(1, cores // n_workers)

    executor = _executor_class(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(backend_name, backend_kwargs, threads_per_worker),
    )
    try:
        fut_to_idx = {
            executor.submit(_run_one_job, jobs[i]): i
            for i in misses
        }
        for fut in as_completed(fut_to_idx):
            if cancel_flag is not None and cancel_flag():
                for f in fut_to_idx:
                    f.cancel()
                break
            idx = fut_to_idx[fut]
            try:
                dicts = fut.result()
            except BrokenExecutor as exc:
                raise SweepPoolBroken(str(exc) or "worker process pool died") from exc
            except Exception as exc:  # one grid failed; report and keep going
                yield (idx, None, f"{type(exc).__name__}: {exc}")
                continue
            if gc is not None and getattr(gc, "enabled", False) and keys.get(idx):
                try:
                    gc.put(keys[idx], dicts, meta={
                        "backend": str(backend_name),
                        "n_p": int(jobs[idx]["n_p"]), "n_t": int(jobs[idx]["n_t"]),
                        "p_range": [float(jobs[idx]["p_range"][0]), float(jobs[idx]["p_range"][1])],
                        "t_range": [float(jobs[idx]["t_range"][0]), float(jobs[idx]["t_range"][1])],
                    })
                except Exception:
                    pass
            yield (idx, dicts, None)
    finally:
        # Don't block the caller waiting on in-flight grids when we bail out
        # (cancel or exception); pending jobs are cancelled.
        executor.shutdown(wait=False, cancel_futures=True)
