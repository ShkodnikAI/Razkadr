"""Минимальный клиент GitHub REST API (stdlib). Токен берётся из GITHUB_TOKEN, если задан."""
import json
import os
import urllib.error
import urllib.request


def repo():
    return os.environ.get("GITHUB_REPOSITORY", "ShkodnikAI/Razkadr")


def request(method, path, data=None):
    url = path if path.startswith("http") else f"https://api.github.com/repos/{repo()}{path}"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "razkadr-bot",
        "Content-Type": "application/json",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, method=method, headers=headers, data=body)
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as err:
        raw = err.read()
        return err.code, (json.loads(raw) if raw else None)


def paged(path):
    sep = "&" if "?" in path else "?"
    page, out = 1, []
    while True:
        status, data = request("GET", f"{path}{sep}per_page=100&page={page}")
        if status != 200:
            raise RuntimeError(f"GET {path}: {status} {data}")
        items = data["workflow_runs"] if isinstance(data, dict) and "workflow_runs" in data else data
        out.extend(items)
        if len(items) < 100:
            return out
        page += 1


def must(method, path, data=None, ok=(200, 201, 204)):
    status, resp = request(method, path, data)
    if status not in ok:
        raise RuntimeError(f"{method} {path}: {status} {resp}")
    return resp
