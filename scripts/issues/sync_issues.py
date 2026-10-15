#!/usr/bin/env python3
"""Генерация docs/naryady/*.md и создание GitHub-меток, вех и issues из seed-данных.

Номер issue == номер наряда (1..41), шлюзы A–F == 42..47. Скрипт идемпотентен:
существующие issues (по номеру) пропускаются. Если номер issue не совпал с ожидаемым —
останавливается. Авторизация берётся из окружения прокси; заголовок Authorization не отправляется,
если не задана явно переменная GITHUB_TOKEN.

Использование: python scripts/issues/sync_issues.py [--render-only] [--repo owner/name]
"""
import argparse, importlib.util, json, os, sys, time, urllib.request, urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
WAVES = {0: "Фундамент", 1: "Индексатор", 2: "Отбор", 3: "Текст", 4: "Раскладка и экспорт", 5: "Приёмка ядра MVP"}
GATE_OF = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F"}
SLUG = {}


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def slugify(s):
    tr = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя", ["a","b","v","g","d","e","e","zh","z","i","y","k","l","m","n","o","p","r","s","t","u","f","h","c","ch","sh","sch","","y","","e","yu","ya"]))
    out = "".join(tr.get(c, c) for c in s.lower())
    out = "".join(c if c.isalnum() else "-" for c in out)
    return "-".join(p for p in out.split("-") if p)[:48]


def refs(ids):
    return ", ".join(f"#{i}" for i in ids) if ids else "—"


def render(n):
    adr = ", ".join(f"ADR-{a:04d}" for a in n["adr"]) or "—"
    head = [
        f"# Наряд №{n['id']:03d} — {n['title']}", "",
        f"**Волна:** {n['wave']} ({WAVES[n['wave']]})  ",
        f"**Стадия:** {n['stage']}  ",
        f"**Размер:** {n['size']}  ",
        f"**Исполнитель:** {n['owner']}  ",
        f"**Зависит от:** {refs(n['depends'])}  ",
        f"**Связанные ADR:** {adr}  ",
        f"**Статус:** {'closed' if n['state']=='closed' else 'open'}" + ("  \n**Черновик:** контракт и критерии уточняются аудитором перед выдачей агенту" if n["draft"] else ""),
        "",
    ]
    L = lambda xs: "\n".join(x if x.startswith("[") is False else f"- {x}" for x in xs)
    body = head + [
        "## Контекст", n["ctx"], "",
        "## Контракт", n["contract"], "",
        "## Scope", "\n".join(f"- {x}" for x in n["scope"]), "",
        "## Non-goals", "\n".join(f"- {x}" for x in n["non"]), "",
        "## Acceptance criteria", "\n".join(f"- {x}" for x in n["accept"]), "",
        "## Assumptions / Open questions", "\n".join(f"- {x}" for x in n["assume"]), "",
        "## Проверка аудитора", "\n".join(f"- {x}" for x in n["audit"]), "",
        "---", "Агент: прочитай `AGENTS.md`. PR содержит `Closes #" + str(n["id"]) + "`, раздел Assumptions и Unresolved; числа только из реальных запусков.", "",
    ]
    return "\n".join(body)


def render_gate(g):
    crit = "\n".join(f"- [ ] {c}" for c in g["criteria"])
    nums = refs(g["depends"])
    return f"""# Шлюз {g['gate']} — приёмка волны {g['wave']} ({WAVES[g['wave']]})

**Нарядов в волне:** {nums}  
**Протокол:** `docs/acceptance/wave-{g['wave']}.md` (по `docs/acceptance/TEMPLATE.md`)  
**Решает:** аудитор готовит протокол, владелец принимает решение go / go с замечаниями / no-go.

## Критерии
{crit}

## No-go
{g['nogo']}

## Порядок
1. Все наряды волны закрыты, PR прошли аудит.
2. Аудитор запускает проверки сам (не со слов агента) и заполняет протокол числами из запусков.
3. Демонстрация владельцу, решение записывается в протокол.
4. При go: аудитор уточняет черновики нарядов следующей волны (снимает `draft`).
"""


def api(repo, method, path, data=None):
    url = f"https://api.github.com/repos/{repo}{path}"
    hdr = {"Accept": "application/vnd.github+json", "User-Agent": "razkadr-sync"}
    if os.environ.get("GITHUB_TOKEN"):
        hdr["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]
    req = urllib.request.Request(url, method=method, headers=hdr,
                                 data=json.dumps(data).encode() if data is not None else None)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-only", action="store_true")
    ap.add_argument("--repo", default="ShkodnikAI/Razkadr")
    a = ap.parse_args()
    nary = load("seed_w0_w1").A + load("seed_w2_w5").B
    gates = load("seed_w2_w5").GATES
    items = {}
    outdir = ROOT / "docs" / "naryady"; outdir.mkdir(exist_ok=True, parents=True)
    for n in nary:
        text = render(n)
        (outdir / f"{n['id']:03d}-{slugify(n['title'])}.md").write_text(text, encoding="utf-8")
        labels = ["naryad", f"wave:{n['wave']}", f"stage:{n['stage']}", f"size:{n['size']}", f"owner:{n['owner']}"] + (["draft"] if n["draft"] else [])
        items[n["id"]] = dict(title=f"Наряд №{n['id']:03d} — {n['title']}", body=text, labels=labels, wave=n["wave"], closed=n["state"] == "closed")
    for g in gates:
        text = render_gate(g)
        (outdir / f"{g['id']:03d}-gate-{g['gate'].lower()}.md").write_text(text, encoding="utf-8")
        items[g["id"]] = dict(title=f"Шлюз {g['gate']} — приёмка волны {g['wave']}", body=text, labels=["gate", f"wave:{g['wave']}"], wave=g["wave"], closed=False)
    assert sorted(items) == list(range(1, 48)), sorted(items)
    print(f"отрендерено файлов: {len(items)}")
    if a.render_only:
        return

    # метки
    colors = {"naryad": "6B0F1A", "gate": "D4AF7F", "draft": "BFBFBF"}
    wanted = {}
    for it in items.values():
        for l in it["labels"]:
            wanted[l] = colors.get(l, "2B2B2B" if l.startswith("owner") else "F8F4E9")
    for name, col in wanted.items():
        st, r = api(a.repo, "POST", "/labels", {"name": name, "color": col})
        if st not in (201, 422):
            sys.exit(f"метка {name}: {st} {r}")
    # вехи
    ms = {}
    st, existing = api(a.repo, "GET", "/milestones?state=all&per_page=100")
    for m in existing or []:
        ms[m["title"]] = m["number"]
    for w, nm in WAVES.items():
        title = f"Волна {w} — {nm} (шлюз {GATE_OF[w]})"
        if title not in ms:
            st, r = api(a.repo, "POST", "/milestones", {"title": title})
            if st != 201:
                sys.exit(f"веха {title}: {st} {r}")
            ms[title] = r["number"]
        ms[w] = ms[title]
    # issues по порядку
    st, ex = api(a.repo, "GET", "/issues?state=all&per_page=100")
    have = {i["number"] for i in ex or [] if "pull_request" not in i}
    for num in range(1, 48):
        it = items[num]
        if num in have:
            print(f"#{num} уже есть, пропуск"); continue
        st, r = api(a.repo, "POST", "/issues", {"title": it["title"], "body": it["body"], "labels": it["labels"], "milestone": ms[it["wave"]]})
        if st != 201:
            sys.exit(f"issue {num}: {st} {r}")
        if r["number"] != num:
            sys.exit(f"номер issue {r['number']} != {num}: остановка")
        if it["closed"]:
            api(a.repo, "PATCH", f"/issues/{num}", {"state": "closed", "state_reason": "completed"})
        print(f"#{num} создан")
        time.sleep(1.2)


if __name__ == "__main__":
    main()
