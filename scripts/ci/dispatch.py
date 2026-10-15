"""Раздача нарядов: ставит метки `ready` / `needs-owner` по состоянию зависимостей.

Правила:
- наряд готов (`ready`), если он открыт, исполнитель agent|both, не draft, все зависимости закрыты,
  нет меток needs-owner / in-progress;
- наряды владельца (owner:human) и шлюзы с закрытыми зависимостями получают `needs-owner` и один раз
  упоминают владельца;
- наряд, зависящий от незакрытого, пропускается; независимые идут дальше.
Агент берёт `ready` с наименьшим номером.
"""
import os
import re

import ghapi

DEP_RE = re.compile(r"\*\*(?:Зависит от|Нарядов в волне):\*\*([^\n]*)")


def parse_deps(body):
    m = DEP_RE.search(body or "")
    return [int(x) for x in re.findall(r"#(\d+)", m.group(1))] if m else []


def names(issue):
    return {lab["name"] for lab in issue["labels"]}


def owner_of(labels):
    for lab in labels:
        if lab.startswith("owner:"):
            return lab.split(":", 1)[1]
    return "human" if "gate" in labels else "agent"


def plan(issues):
    """Возвращает список действий: (op, номер, аргумент)."""
    closed = {i["number"] for i in issues if i["state"] == "closed"}
    actions = []
    for issue in sorted(issues, key=lambda i: i["number"]):
        if issue["state"] != "open":
            continue
        labels = names(issue)
        if not labels & {"naryad", "gate"}:
            continue
        n = issue["number"]
        blocked = [d for d in parse_deps(issue["body"]) if d not in closed]
        if blocked or "draft" in labels:
            if "ready" in labels:
                actions.append(("unlabel", n, "ready"))
            continue
        owner = owner_of(labels)
        is_gate = "gate" in labels
        can_start = owner in ("agent", "both") and not is_gate
        if can_start and not labels & {"needs-owner", "in-progress", "ready"}:
            actions.append(("label", n, "ready"))
        if (owner in ("human", "both") or is_gate) and "owner-pinged" not in labels:
            if owner == "both":
                text = "Агент может начинать. От вас потребуется действие, описанное в контракте (ключи, ссылки, утверждение)."
                actions.append(("label", n, "owner-pinged"))
            elif is_gate:
                text = "Все наряды волны закрыты. Нужна приёмка шлюза: аудитор готовит протокол, решение go / no-go за вами."
                actions += [("label", n, "needs-owner"), ("label", n, "owner-pinged")]
            else:
                text = "Зависимости закрыты, наряд для вас можно выполнять."
                actions += [("label", n, "needs-owner"), ("label", n, "owner-pinged")]
            actions.append(("comment", n, text))
    return actions


def run():
    issues = [i for i in ghapi.paged("/issues?state=all") if "pull_request" not in i]
    owner = os.environ.get("OWNER_LOGIN", ghapi.repo().split("/")[0])
    for op, n, arg in plan(issues):
        if op == "label":
            ghapi.must("POST", f"/issues/{n}/labels", {"labels": [arg]})
        elif op == "unlabel":
            ghapi.request("DELETE", f"/issues/{n}/labels/{arg}")
        elif op == "comment":
            ghapi.must("POST", f"/issues/{n}/comments", {"body": f"@{owner} {arg}"})
        print(op, n, arg)
    fresh = [i for i in ghapi.paged("/issues?state=open&labels=ready") if "pull_request" not in i]
    queue = sorted(i["number"] for i in fresh)
    line = "Очередь агента (ready): " + (", ".join(f"#{n}" for n in queue) or "пусто")
    print(line)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


if __name__ == "__main__":
    run()
