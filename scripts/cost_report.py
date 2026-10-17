"""Command-line wrapper for the cost report aggregator (naryad 030).

The implementation lives in ``razkadr.cost_report`` (importable, tested
in ``tests/test_cost_report.py``); this file only exposes the CLI:

    python scripts/cost_report.py runs/<id> [runs/<id> ...] --label ... [--out PATH]
"""

import sys

from razkadr.cost_report import main

if __name__ == "__main__":
    sys.exit(main())
