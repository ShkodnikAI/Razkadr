"""CLI skeleton tests: stubs, exit codes, --version, --verbose (naryad 036)."""

import json

import pytest

from razkadr import __version__
from razkadr.cli import EXIT_NOT_IMPLEMENTED, STUB_COMMANDS, main


@pytest.mark.parametrize("cmd", sorted(STUB_COMMANDS))
def test_stub_commands_exit_3_with_message(cmd, capsys):
    assert main([cmd]) == EXIT_NOT_IMPLEMENTED
    err = capsys.readouterr().err
    assert "not implemented" in err
    assert f"No{STUB_COMMANDS[cmd]}" in err


def test_stub_accepts_common_interface(tmp_path):
    """Unified interface: --config/--force/--runs-dir/--resume parse and are accepted.

    Index became a real command (naryad 005), select (naryad 014) and
    text (naryad 017) became real, so the stub check uses build.
    """
    code = main([
        "build", "--config", str(tmp_path / "missing.toml"), "--force",
        "--runs-dir", str(tmp_path / "runs"), "--resume", "20260101T000000Z_deadbeef",
    ])
    assert code == EXIT_NOT_IMPLEMENTED


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_command_prints_help_and_exits_zero(capsys):
    assert main([]) == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_verbose_flag_enables_debug_json_logging(capsys):
    from razkadr.log import get_logger

    assert main(["--verbose", "build"]) == EXIT_NOT_IMPLEMENTED
    err = capsys.readouterr().err
    assert '"level": "DEBUG"' not in err  # nothing debug-logged yet, but handler is DEBUG
    get_logger("probe").debug("verbose probe works")
    err = capsys.readouterr().err
    assert '"level": "DEBUG"' in err and "verbose probe works" in err


def test_validate_still_works_alongside_stubs(tmp_path, capsys):
    doc = {"schema_version": "1.0", "project": {"id": "p", "title": "t", "preset": "recipe", "output_lang": "ru"},
           "sources": [], "episodes": []}
    path = tmp_path / "sb.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert main(["validate", str(path)]) == 0
    assert capsys.readouterr().out.strip() == "ok"
