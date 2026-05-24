"""Tests for stable path resolution (paths.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dispatcher_watts.paths import HOME_ENV_VAR, home_dir


def test_env_var_overrides_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(HOME_ENV_VAR, str(tmp_path))
    assert home_dir() == tmp_path.resolve()


def test_env_var_expands_user(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(HOME_ENV_VAR, "~/dw-home")
    assert home_dir() == (Path.home() / "dw-home").resolve()


def test_falls_back_to_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    # With no override, home_dir() discovers the repo root by walking up to the
    # nearest pyproject.toml -- so the result must contain one.
    monkeypatch.delenv(HOME_ENV_VAR, raising=False)
    resolved = home_dir()
    assert (resolved / "pyproject.toml").is_file()


def test_home_is_independent_of_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The whole point: changing the working directory must not move the home.
    monkeypatch.delenv(HOME_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    assert (home_dir() / "pyproject.toml").is_file()
