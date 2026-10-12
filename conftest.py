"""Pytest configuration — make the repo root importable from tests/.

A repo-root ``conftest.py`` causes pytest to add this directory to ``sys.path``
so ``import thermo`` / ``eds_cache`` / ``oxford_loader`` / ``viewer.*`` resolve
regardless of the working directory tests are launched from.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
