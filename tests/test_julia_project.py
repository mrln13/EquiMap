"""juliapkg uses the conda environment's Julia project however Python is started."""
from thermo.backends.magemin import _use_env_julia_project


def _env(tmp_path, julia_env=True):
    (tmp_path / "conda-meta").mkdir()
    if julia_env:
        (tmp_path / "julia_env").mkdir()
    return tmp_path


def test_not_activated_points_at_env_project(tmp_path):
    prefix = _env(tmp_path)
    environ = {}
    assert _use_env_julia_project(environ, prefix) == str(prefix / "julia_env")
    assert environ["PYTHON_JULIAPKG_PROJECT"] == str(prefix / "julia_env")


def test_other_env_activated_still_points_at_ours(tmp_path):
    prefix = _env(tmp_path)
    environ = {"CONDA_PREFIX": str(tmp_path.parent)}      # e.g. base activated
    assert _use_env_julia_project(environ, prefix) == str(prefix / "julia_env")


def test_activated_or_explicit_left_alone(tmp_path):
    prefix = _env(tmp_path)
    environ = {"CONDA_PREFIX": str(prefix)}
    assert _use_env_julia_project(environ, prefix) is None and "PYTHON_JULIAPKG_PROJECT" not in environ
    environ = {"PYTHON_JULIAPKG_PROJECT": "X:/mine"}
    assert _use_env_julia_project(environ, prefix) is None
    assert environ["PYTHON_JULIAPKG_PROJECT"] == "X:/mine"


def test_no_env_project_yet_changes_nothing(tmp_path):
    prefix = _env(tmp_path, julia_env=False)
    environ = {}
    assert _use_env_julia_project(environ, prefix) is None and environ == {}
