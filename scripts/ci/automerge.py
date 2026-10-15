"""Автомёрж PR наряда после зелёного CI и запуск раздачи следующих нарядов.

Мёржится PR, если: ветка `naryad/*`, не draft, CI по текущему head успешен, в теле `Closes #N`
(N — открытый наряд), есть разделы Assumptions и Unresolved, нет метки needs-owner, ветка не отстаёт
от main, нет конфликтов, не тронуты защищённые пути. Тронуты защищённые пути — PR получает
needs-owner и ждёт владельца.
"""
import re
import sys
import time

import dispatch
import ghapi

PROTECTED = ("docs/adr/", ".github/", "scripts/ci/", "AGENTS.md", "LICENSE", "docs/AUDIT.md", "PROJECT_SCHEME.md")
CLOSES = re.compile(r"(?i)\bcloses\s+#(\d+)")


def verdict(pr, files, issue, behind_by, ci_ok):
    """-> (kind, message); kind: silent | comment | owner | merge."""
    labels = {lab["name"] for lab in pr["labels"]}
    if not pr["head"]["ref"].startswith("naryad/") or pr["draft"] or not ci_ok:
        return "silent", ""
    if "needs-owner" in labels:
        return "silent", ""
    body = pr.get("body") or ""
    m = CLOSES.search(body)
    if not m:
        return "comment", "В теле PR нет `Closes #N`."
    if issue is None or issue["state"] != "open" or "naryad" not in {lab["name"] for lab in issue["labels"]}:
        return "comment", f"#{m.group(1)} не открытый наряд."
    if "needs-owner" in {lab["name"] for lab in issue["labels"]}:
        return "silent", ""
    low = body.lower()
    if "assumptions" not in low or "unresolved" not in low:
        return "comment", "В теле PR должны быть разделы Assumptions и Unresolved."
    touched = [f for f in files if f.startswith(PROTECTED)]
    if touched:
        return "owner", "PR меняет защищённые пути, нужно решение владельца: " + ", ".join(sorted(touched)[:8])
    if behind_by > 0:
        return "comment", f"Ветка отстаёт от main на {behind_by} коммит(ов): сделай rebase и запушь."
    if pr.get("mergeable") is False:
        return "comment", "Конфликт с main: разреши и запушь."
    return "merge", ""


def ci_green(sha):
    runs = [r for r in ghapi.paged(f"/actions/runs?head_sha={sha}") if r["name"] == "ci"]
    return bool(runs) and all(r["status"] == "completed" and r["conclusion"] == "success" for r in runs)


def comment_once(number, sha, key, text):
    marker = f"<!-- razkadr-bot:{key}:{sha[:7]} -->"
    for c in ghapi.paged(f"/issues/{number}/comments"):
        if marker in c["body"]:
            return
    ghapi.must("POST", f"/issues/{number}/comments", {"body": f"{marker}\n{text}"})


def sweep():
    for item in ghapi.paged("/pulls?state=open"):
        n, sha = item["number"], item["head"]["sha"]
        pr = ghapi.must("GET", f"/pulls/{n}")
        for _ in range(5):
            if pr.get("mergeable") is not None:
                break
            time.sleep(2)
            pr = ghapi.must("GET", f"/pulls/{n}")
        if not pr["head"]["ref"].startswith("naryad/") or pr["draft"]:
            continue
        ok = ci_green(sha)
        m = CLOSES.search(pr.get("body") or "")
        issue = None
        if m:
            status, issue = ghapi.request("GET", f"/issues/{m.group(1)}")
            issue = issue if status == 200 else None
        files = [f["filename"] for f in ghapi.paged(f"/pulls/{n}/files")]
        cmp_ = ghapi.must("GET", f"/compare/{pr['base']['ref']}...{pr['head']['label']}")
        kind, msg = verdict(pr, files, issue, cmp_["behind_by"], ok)
        print(f"PR #{n}: {kind} {msg}")
        if kind == "comment":
            comment_once(n, sha, "fix", msg)
        elif kind == "owner":
            ghapi.must("POST", f"/issues/{n}/labels", {"labels": ["needs-owner"]})
            comment_once(n, sha, "owner", f"@{ghapi.repo().split('/')[0]} {msg}. Снимите метку `needs-owner`, когда решите.")
        elif kind == "merge":
            title = f"{pr['title']} (#{n})"
            ghapi.must("PUT", f"/pulls/{n}/merge", {"merge_method": "squash", "sha": sha, "commit_title": title, "commit_message": pr.get("body") or ""})
            num = int(m.group(1))
            ghapi.must("PATCH", f"/issues/{num}", {"state": "closed", "state_reason": "completed"})
            for lab in ("in-progress", "ready"):
                ghapi.request("DELETE", f"/issues/{num}/labels/{lab}")
            if pr["head"]["repo"] and pr["head"]["repo"]["full_name"] == ghapi.repo():
                ghapi.request("DELETE", f"/git/refs/heads/{pr['head']['ref']}")
            print(f"merged PR #{n}, closed #{num}")


if __name__ == "__main__":
    sweep()
    dispatch.run()
    sys.exit(0)
