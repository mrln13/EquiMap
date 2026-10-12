"""Tests for the on-disk grid result cache (thermo.cache) and its wiring into
ThermoCalc.run_grid + thermo.parallel.iter_grid_results.

No Julia / real backend needed: a fake backend (subclass of AbstractBackend)
supplies deterministic ThermoResults, and the parallel path is driven through a
ThreadPoolExecutor seam exactly like test_thermo_parallel.py. Caches are always
pointed at a tmp_path so the user's real ~/.thermod/grid_cache is never touched.
"""
import concurrent.futures as cf
import json

import pytest

from thermo import cache as cache_mod
from thermo.cache import GridCache, CACHE_VERSION
from thermo.composition import Composition
from thermo.backends.base import AbstractBackend
from thermo.result import (
    ThermoResult, PhaseResult, result_to_portable_dict,
)
from thermo import parallel

SEED_EL = {"Si": 21.0, "Al": 6.4, "Ca": 6.7, "Mg": 4.6, "Fe": 6.2}


def _comp(**over):
    d = dict(SEED_EL)
    d.update(over)
    return Composition.from_element_wt(d)


def _grid(comp, n_p=2, n_t=2, p_range=(0.5, 1.5), t_range=(600.0, 800.0)):
    out = []
    for ip in range(n_p):
        for it in range(n_t):
            out.append(ThermoResult(
                p_gpa=float(p_range[0] + ip), t_c=float(t_range[0] + it),
                stable_phases=[PhaseResult("q", 1.0, 1.0)],
                bulk_properties={"rho_kgm3": 3000.0 + ip + it},
                backend="fake", converged=True))
    return out


# ---------------------------------------------------------------------------
# Key behaviour
# ---------------------------------------------------------------------------

def test_key_is_stable_and_sensitive(tmp_path):
    gc = GridCache(tmp_path)
    base = dict(backend_name="magemin", backend_kwargs={"db": "ig"},
                composition=_comp(), p_range=(0.5, 1.5), t_range=(600, 800),
                n_p=4, n_t=4)
    k1 = gc.key_for(**base)
    k2 = gc.key_for(**base)
    assert k1 == k2                                   # deterministic
    assert len(k1) == 64                              # sha-256 hex

    # Each varied dimension must change the key.
    assert gc.key_for(**{**base, "backend_name": "perplex"}) != k1
    assert gc.key_for(**{**base, "backend_kwargs": {"db": "mp"}}) != k1
    assert gc.key_for(**{**base, "composition": _comp(Fe=9.9)}) != k1
    assert gc.key_for(**{**base, "p_range": (0.6, 1.5)}) != k1
    assert gc.key_for(**{**base, "t_range": (600, 850)}) != k1
    assert gc.key_for(**{**base, "n_p": 5}) != k1
    assert gc.key_for(**{**base, "n_t": 5}) != k1
    assert gc.key_for(**{**base, "extra": {"fe3_fraction": 0.1}}) != k1


def test_key_accepts_dict_composition(tmp_path):
    gc = GridCache(tmp_path)
    common = dict(backend_name="magemin", backend_kwargs={}, p_range=(0.5, 1.5),
                  t_range=(600, 800), n_p=2, n_t=2)
    # Composition object and its element-wt dict must hash identically.
    comp = _comp()
    k_obj = gc.key_for(composition=comp, **common)
    k_dict = gc.key_for(composition=comp.to_element_wt(), **common)
    assert k_obj == k_dict


def test_key_backend_name_case_insensitive(tmp_path):
    gc = GridCache(tmp_path)
    common = dict(backend_kwargs={}, composition=_comp(), p_range=(0.5, 1.5),
                  t_range=(600, 800), n_p=2, n_t=2)
    assert gc.key_for(backend_name="MAGEMin", **common) == \
           gc.key_for(backend_name="magemin", **common)


# ---------------------------------------------------------------------------
# put / get round trip
# ---------------------------------------------------------------------------

def test_put_get_roundtrip(tmp_path):
    gc = GridCache(tmp_path)
    key = gc.key_for("fake", {}, _comp(), (0.5, 1.5), (600, 800), 2, 2)
    assert gc.get(key) is None                        # cold miss

    results = _grid(_comp())
    assert gc.put_results(key, results, meta={"backend": "fake"}) is True

    dicts = gc.get(key)
    assert dicts is not None and len(dicts) == 4
    objs = gc.get_results(key)
    assert len(objs) == 4
    assert objs[0].backend == "fake"
    assert objs[0].bulk_properties["rho_kgm3"] == pytest.approx(3000.0)
    assert objs[-1].stable_phases[0].name == "q"


def test_put_writes_one_json_file(tmp_path):
    gc = GridCache(tmp_path)
    key = gc.key_for("fake", {}, _comp(), (0.5, 1.5), (600, 800), 2, 2)
    gc.put(key, [result_to_portable_dict(r) for r in _grid(_comp())])
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    assert files[0].stem == key
    # No leftover temp files.
    assert not list(tmp_path.glob("*.tmp"))


def test_version_gate(tmp_path, monkeypatch):
    gc = GridCache(tmp_path)
    key = gc.key_for("fake", {}, _comp(), (0.5, 1.5), (600, 800), 2, 2)
    gc.put(key, [result_to_portable_dict(r) for r in _grid(_comp())])
    assert gc.get(key) is not None

    # Bump the running version: the on-disk entry (still tagged old version)
    # must now be ignored.
    monkeypatch.setattr(cache_mod, "CACHE_VERSION", CACHE_VERSION + 1)
    gc2 = GridCache(tmp_path)
    assert gc2.get(key) is None


def test_corrupt_file_is_a_miss(tmp_path):
    gc = GridCache(tmp_path)
    key = gc.key_for("fake", {}, _comp(), (0.5, 1.5), (600, 800), 2, 2)
    (tmp_path / f"{key}.json").write_text("{ this is not valid json", encoding="utf-8")
    assert gc.get(key) is None                        # tolerated, not raised


def test_wrong_shape_file_is_a_miss(tmp_path):
    gc = GridCache(tmp_path)
    key = gc.key_for("fake", {}, _comp(), (0.5, 1.5), (600, 800), 2, 2)
    (tmp_path / f"{key}.json").write_text(
        json.dumps({"version": CACHE_VERSION, "results": "not-a-list"}),
        encoding="utf-8")
    assert gc.get(key) is None


# ---------------------------------------------------------------------------
# enable / disable
# ---------------------------------------------------------------------------

def test_disabled_instance_is_noop(tmp_path):
    gc = GridCache(tmp_path, enabled=False)
    key = gc.key_for("fake", {}, _comp(), (0.5, 1.5), (600, 800), 2, 2)
    assert gc.put(key, [result_to_portable_dict(r) for r in _grid(_comp())]) is False
    assert gc.get(key) is None
    assert not list(tmp_path.glob("*.json"))


def test_env_disable(tmp_path, monkeypatch):
    monkeypatch.setenv("THERMOD_GRID_CACHE", "off")
    gc = GridCache(tmp_path)                           # enabled=True ignored
    assert gc.enabled is False
    assert cache_mod.cache_disabled_by_env() is True


def test_env_dir_override(tmp_path, monkeypatch):
    monkeypatch.setenv("THERMOD_GRID_CACHE_DIR", str(tmp_path / "custom"))
    assert cache_mod.default_cache_dir() == tmp_path / "custom"


# ---------------------------------------------------------------------------
# maintenance
# ---------------------------------------------------------------------------

def test_clear_and_stats(tmp_path):
    gc = GridCache(tmp_path)
    for fe in (6.0, 7.0, 8.0):
        key = gc.key_for("fake", {}, _comp(Fe=fe), (0.5, 1.5), (600, 800), 2, 2)
        gc.put(key, [result_to_portable_dict(r) for r in _grid(_comp(Fe=fe))])
    s = gc.stats()
    assert s["n_entries"] == 3
    assert s["total_bytes"] > 0
    assert s["enabled"] is True
    assert gc.clear() == 3
    assert gc.stats()["n_entries"] == 0


# ---------------------------------------------------------------------------
# ThermoCalc.run_grid integration (fake backend instance)
# ---------------------------------------------------------------------------

class _CountingBackend(AbstractBackend):
    name = "fake"

    def __init__(self):
        self.calls = 0

    def run(self, composition, p_gpa, t_c):           # pragma: no cover - unused
        raise NotImplementedError

    def run_grid(self, composition, p_range, t_range, n_p=20, n_t=20, **kw):
        self.calls += 1
        return _grid(composition, n_p=n_p, n_t=n_t,
                     p_range=p_range, t_range=t_range)


def test_thermocalc_run_grid_caches(tmp_path):
    from thermo import ThermoCalc

    backend = _CountingBackend()
    calc = ThermoCalc(backend=backend)
    gc = GridCache(tmp_path)

    r1 = calc.run_grid(_comp(), p_range=(0.5, 1.5), t_range=(600, 800),
                       n_p=3, n_t=3, cache=gc)
    assert backend.calls == 1 and len(r1) == 9

    # Identical grid → served from cache, backend not called again.
    r2 = calc.run_grid(_comp(), p_range=(0.5, 1.5), t_range=(600, 800),
                       n_p=3, n_t=3, cache=gc)
    assert backend.calls == 1                          # still 1 — cache hit
    assert [r.p_gpa for r in r2] == [r.p_gpa for r in r1]

    # Different grid → miss → backend runs again.
    calc.run_grid(_comp(), p_range=(0.5, 2.0), t_range=(600, 800),
                  n_p=3, n_t=3, cache=gc)
    assert backend.calls == 2


def test_thermocalc_use_cache_false_skips_read(tmp_path):
    from thermo import ThermoCalc

    backend = _CountingBackend()
    calc = ThermoCalc(backend=backend)
    gc = GridCache(tmp_path)
    key = gc.key_for("fake", {}, _comp(), (0.5, 1.5), (600, 800), 2, 2)
    gc.put(key, [result_to_portable_dict(r) for r in _grid(_comp())])

    # use_cache=False must ignore the populated cache and recompute.
    calc.run_grid(_comp(), p_range=(0.5, 1.5), t_range=(600, 800),
                  n_p=2, n_t=2, cache=gc, use_cache=False)
    assert backend.calls == 1


# ---------------------------------------------------------------------------
# parallel.iter_grid_results integration
# ---------------------------------------------------------------------------

class _ExplodingExecutor:
    """Stand-in executor that fails if anyone tries to construct it."""
    def __init__(self, *a, **k):
        raise AssertionError("executor must not be created when fully cached")


def _install_fake(monkeypatch):
    class _FakeBackend:
        def run_grid(self, comp, p_range, t_range, n_p, n_t):
            return _grid(comp, n_p=n_p, n_t=n_t, p_range=p_range, t_range=t_range)

    def _fake_init(name, kwargs, threads):
        parallel._WORKER_BACKEND = _FakeBackend()
    monkeypatch.setattr(parallel, "_init_worker", _fake_init)


def _jobs_distinct(n):
    return [{
        "comp": _comp(Fe=6.0 + i),
        "p_range": (0.5, 1.5), "t_range": (600.0, 800.0), "n_p": 2, "n_t": 2,
    } for i in range(n)]


def test_parallel_populates_then_serves_from_cache(tmp_path, monkeypatch):
    _install_fake(monkeypatch)
    gc = GridCache(tmp_path)
    jobs = _jobs_distinct(3)

    # First pass: cold cache → executor runs all 3, results get stored.
    out1 = list(parallel.iter_grid_results(
        jobs, "fake", {}, max_workers=2, cache=gc,
        _executor_class=cf.ThreadPoolExecutor))
    assert len(out1) == 3
    assert all(e is None and d is not None for _, d, e in out1)
    assert gc.stats()["n_entries"] == 3

    # Second pass: every job is a cache hit, so the executor must NEVER be
    # constructed (the exploding stand-in proves it).
    out2 = list(parallel.iter_grid_results(
        jobs, "fake", {}, max_workers=2, cache=gc,
        _executor_class=_ExplodingExecutor))
    assert len(out2) == 3
    assert all(e is None and d is not None for _, d, e in out2)
    assert {i for i, _, _ in out2} == {0, 1, 2}


def test_parallel_mixed_hit_and_miss(tmp_path, monkeypatch):
    _install_fake(monkeypatch)
    gc = GridCache(tmp_path)
    jobs = _jobs_distinct(3)

    # Pre-seed only job 1's key.
    j = jobs[1]
    key = gc.key_for("fake", {}, j["comp"], j["p_range"], j["t_range"],
                     j["n_p"], j["n_t"])
    gc.put(key, [result_to_portable_dict(r) for r in
                 _grid(j["comp"], n_p=2, n_t=2)])

    out = list(parallel.iter_grid_results(
        jobs, "fake", {}, max_workers=2, cache=gc,
        _executor_class=cf.ThreadPoolExecutor))
    assert {i for i, _, _ in out} == {0, 1, 2}        # all delivered
    assert gc.stats()["n_entries"] == 3                # misses now stored too
