"""Scaffolding smoke tests — verify the package layout in Spec.html §I.

These run with zero optional deps (numpy is the only runtime dep) so CI green
on the bare wheel proves the skeleton is healthy before any module is built.
"""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest


def test_package_version() -> None:
    import pokerbot

    assert pokerbot.__version__ == "0.1.0"


@pytest.mark.parametrize(
    "module",
    [
        "pokerbot",
        "pokerbot.abstraction",
        "pokerbot.strategy_db",
        "pokerbot.runtime",
        "pokerbot.training",
        "pokerbot.tournament",
        "pokerbot.cli",
    ],
)
def test_subpackages_importable(module: str) -> None:
    importlib.import_module(module)


def test_cli_help_exits_clean(capsys: pytest.CaptureFixture[str]) -> None:
    from pokerbot.cli import _build_parser

    parser = _build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for cmd in (
        "build-abstraction",
        "train",
        "evaluate",
        "export-strategy",
        "serve",
        "build-pushfold",
    ):
        assert cmd in out, f"{cmd!r} missing from CLI --help"


def test_cli_unknown_command_errors() -> None:
    from pokerbot.cli import _build_parser

    parser = _build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["nonsense"])
    assert exc.value.code == 2


def test_cli_pending_command_returns_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    """Subcommands that are still stubbed return rc=1 with a pointer message.

    `build-abstraction` was wired up in GAP 2, so we exercise one that's still
    pending — `serve` (step 6's HTTP server).
    """
    from pokerbot.cli import main

    rc = main(["serve", "--db", "sqlite:///:memory:", "--abstraction", "/tmp/x"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "not yet implemented" in err
    assert "step 6" in err


def test_cli_module_entrypoint() -> None:
    """`python -m pokerbot.cli --version` should print the package version."""
    result = subprocess.run(
        [sys.executable, "-m", "pokerbot.cli", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "0.1.0" in result.stdout
