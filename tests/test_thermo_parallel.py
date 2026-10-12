"""Orchestration tests for thermo.parallel (no Julia needed).

Drives the pool with a thread executor + a fake backend, monkeypatching the
worker initialiser so the success/error/cancel/broken-pool paths are exercised
without a real backend. The real ProcessPoolExecutor + MAGEMin path is checked
by the user in the GUI; what's testable here is the orchestration + transport.
"""
import concurrent.futures as cf

import pytest

from thermo import parallel
from thermo.composition import Composition
from thermo.result import (
    ThermoResult, PhaseResult, result_from_portable_dict,
)

SEED_EL = {"Si": 21.0, "Al": 6.4, "Ca": 6.7, "Mg": 4.6, "Fe": 6.2}


def _jobs(n):
    return [{
        "comp": Composition.from_element_wt(SEED_EL),
        "p_range": (0.5, 1.5), "t_range": (600.0, 800.0), "n_p": 2, "n_t": 2,
    } for _ in range(n)]


class _FakeBackend:
    def run_grid(self, comp, p_range, t_range, n_p, n_t):
        assert isinstance(comp, Composition)
        assert "Si" in comp.to_element_wt()
        out = []
        for ip in range(n_p):
            for it in range(n_t):
                out.append(ThermoResult(
                    p_gpa=float(p_range[0] + ip), t_c=float(t_range[0] + it),
                    stable_phases=[PhaseResult("q", 1.0, 1.0)],
                    bulk_properties={"rho_kgm3": 3000.0 + ip + it},
                    backend="fake", converged=True))
        return out


class _BoomBackend(_FakeBackend):
    def run_grid(self, *a, **k):
        raise RuntimeError("kaboom in one grid")


def _install(monkeypatch, backend_cls):
    def _fake_init(name, kwargs, threads):
        parallel._WORKER_BACKEND = backend_cls()
    monkeypatch.setattr(parallel, "_init_worker", _fake_init)


def test_default_worker_count():
    assert parallel.default_worker_count(1) == 1
    assert parallel.default_worker_count(100, cap=4) <= 4
    assert parallel.default_worker_count(2) <= 2


def test_success(monkeypatch):
    _install(monkeypatch, _FakeBackend)
    out = list(parallel.iter_grid_results(
        _jobs(5), "fake", {}, max_workers=3, use_cache=False,
        _executor_class=cf.ThreadPoolExecutor))
    assert len(out) == 5
    seen = set()
    for idx, dicts, err in out:
        assert err is None and dicts is not None
        seen.add(idx)
        assert len(dicts) == 4                      # 2x2 grid
        rr = [result_from_portable_dict(d) for d in dicts]
        assert all(r.backend == "fake" for r in rr)
    assert seen == set(range(5))                    # each job once, no dupes


def test_per_grid_error_continues(monkeypatch):
    _install(monkeypatch, _BoomBackend)
    out = list(parallel.iter_grid_results(
        _jobs(3), "fake", {}, max_workers=2, use_cache=False,
        _executor_class=cf.ThreadPoolExecutor))
    assert len(out) == 3
    assert all(d is None and e is not None for _, d, e in out)
    assert all("kaboom" in e for _, _, e in out)


def test_cancel_stops_early(monkeypatch):
    _install(monkeypatch, _FakeBackend)
    seen = []
    flag = {"v": False}

    def cancel_flag():
        if seen:
            flag["v"] = True
        return flag["v"]

    for idx, dicts, err in parallel.iter_grid_results(
            _jobs(8), "fake", {}, max_workers=2, cancel_flag=cancel_flag,
            use_cache=False, _executor_class=cf.ThreadPoolExecutor):
        seen.append(idx)
    assert 0 < len(seen) < 8


def test_broken_pool_raises(monkeypatch):
    def _boom_init(name, kwargs, threads):
        raise RuntimeError("backend init failed (e.g. Julia missing)")
    monkeypatch.setattr(parallel, "_init_worker", _boom_init)
    with pytest.raises(parallel.SweepPoolBroken):
        list(parallel.iter_grid_results(
            _jobs(3), "fake", {}, max_workers=2, use_cache=False,
            _executor_class=cf.ThreadPoolExecutor))


def test_empty_jobs(monkeypatch):
    _install(monkeypatch, _FakeBackend)
    assert list(parallel.iter_grid_results([], "fake", {})) == []
