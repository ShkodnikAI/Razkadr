from razkadr import __version__
from razkadr.cli import main


def test_version():
    assert __version__ == "0.0.1"


def test_cli_runs():
    assert main([]) == 0
