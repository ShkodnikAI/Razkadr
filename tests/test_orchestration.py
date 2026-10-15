import importlib.util
import sys
from pathlib import Path

CI_DIR = Path(__file__).resolve().parents[1] / "scripts" / "ci"
sys.path.insert(0, str(CI_DIR))


def load(name):
    spec = importlib.util.spec_from_file_location(name, CI_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dispatch = load("dispatch")
automerge = load("automerge")


def issue(n, labels, body="", state="open"):
    return {"number": n, "state": state, "body": body, "labels": [{"name": x} for x in labels]}


def test_parse_deps():
    assert dispatch.parse_deps("**Зависит от:** #2, #34  \nx") == [2, 34]
    assert dispatch.parse_deps("**Зависит от:** —  ") == []
    assert dispatch.parse_deps("**Нарядов в волне:** #1, #2") == [1, 2]


def test_ready_only_when_deps_closed():
    issues = [
        issue(1, ["naryad", "owner:agent"], state="closed"),
        issue(2, ["naryad", "owner:agent"], "**Зависит от:** #1"),
        issue(3, ["naryad", "owner:agent"], "**Зависит от:** #2"),
        issue(4, ["naryad", "owner:agent", "draft"], "**Зависит от:** #1"),
    ]
    acts = dispatch.plan(issues)
    assert ("label", 2, "ready") in acts
    assert not any(a[1] in (3, 4) for a in acts)


def test_independent_issue_not_blocked_by_owner_issue():
    issues = [
        issue(40, ["naryad", "owner:human"], "**Зависит от:** —"),
        issue(5, ["naryad", "owner:agent"], "**Зависит от:** —"),
    ]
    acts = dispatch.plan(issues)
    assert ("label", 40, "needs-owner") in acts
    assert ("label", 5, "ready") in acts


def test_in_progress_and_ping_once():
    issues = [
        issue(5, ["naryad", "owner:agent", "in-progress"]),
        issue(40, ["naryad", "owner:human", "needs-owner", "owner-pinged"]),
    ]
    assert dispatch.plan(issues) == []


def test_gate_pings_owner_and_blocked_unready_removed():
    issues = [
        issue(2, ["naryad"], state="closed"),
        issue(42, ["gate"], "**Нарядов в волне:** #2"),
        issue(7, ["naryad", "owner:agent", "ready"], "**Зависит от:** #99"),
    ]
    acts = dispatch.plan(issues)
    assert ("label", 42, "needs-owner") in acts
    assert ("unlabel", 7, "ready") in acts


def pr(**kw):
    base = {"head": {"ref": "naryad/2-schema"}, "draft": False, "labels": [], "mergeable": True,
            "body": "Closes #2\n## Assumptions\n-\n## Unresolved\n-"}
    base.update(kw)
    return base


OPEN_ISSUE = issue(2, ["naryad"])


def test_verdict_merge():
    assert automerge.verdict(pr(), ["src/a.py", "schemas/x.json"], OPEN_ISSUE, 0, True)[0] == "merge"


def test_verdict_requires_ci_and_branch():
    assert automerge.verdict(pr(), [], OPEN_ISSUE, 0, False)[0] == "silent"
    assert automerge.verdict(pr(head={"ref": "feature/x"}), [], OPEN_ISSUE, 0, True)[0] == "silent"


def test_verdict_body_rules():
    assert automerge.verdict(pr(body="нет"), [], OPEN_ISSUE, 0, True)[0] == "comment"
    assert automerge.verdict(pr(body="Closes #2"), [], OPEN_ISSUE, 0, True)[0] == "comment"
    assert automerge.verdict(pr(), [], None, 0, True)[0] == "comment"


def test_verdict_protected_and_owner_hold():
    assert automerge.verdict(pr(), ["docs/adr/0006.md"], OPEN_ISSUE, 0, True)[0] == "owner"
    assert automerge.verdict(pr(), ["AGENTS.md"], OPEN_ISSUE, 0, True)[0] == "owner"
    held = pr(labels=[{"name": "needs-owner"}])
    assert automerge.verdict(held, [], OPEN_ISSUE, 0, True)[0] == "silent"


def test_verdict_behind_and_conflict():
    assert automerge.verdict(pr(), [], OPEN_ISSUE, 2, True)[0] == "comment"
    assert automerge.verdict(pr(mergeable=False), [], OPEN_ISSUE, 0, True)[0] == "comment"
