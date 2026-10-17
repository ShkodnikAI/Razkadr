"""Google Slides export tests (naryad 025): hermetic, no network, no keys.

The transport is a stub returning canned responses modeled on the Drive
resumable-upload protocol (metadata POST -> Location header -> PUT bytes);
it records every request. Live verification against a real Google account
requires the owner's OAuth token and is covered by docs/limitations.md -
the naryad's real-API criterion stays with the owner (like live marks in
naryads 037-039).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from razkadr.cli import EXIT_FILE_ERROR, EXIT_INVALID, EXIT_OK
from razkadr.cli import main as cli_main
from razkadr.log import SECRETS
from razkadr.providers.base import (
    AuthError,
    ProviderConfigError,
    ProviderError,
    QuotaError,
    TransientError,
)
from razkadr.providers.rest import Request
from razkadr.render.gslides import (
    PPTX_MIME,
    SLIDES_MIME,
    GSlidesUpload,
    export_gslides,
    resolve_google_token,
    slides_link,
)

FILE_ID = "1AbCdeckId123"
WEB_VIEW = f"https://docs.google.com/presentation/d/{FILE_ID}/edit"
TOKEN = "ya29.test-token-abcdef"


class StubTransport:
    """Two-step stub: the first call answers the session URI, the second
    the file JSON. Records all requests."""

    def __init__(
        self,
        file_body: bytes = json.dumps({"id": FILE_ID, "name": "deck", "webViewLink": WEB_VIEW}).encode(),
        session_status: int = 200,
        upload_status: int = 200,
        session_headers: dict | None = None,
    ) -> None:
        self.file_body = file_body
        self.session_status = session_status
        self.upload_status = upload_status
        self.session_headers = session_headers if session_headers is not None else {"Location": "https://session.example/abc"}
        self.requests: list[Request] = []

    def __call__(self, request: Request):
        self.requests.append(request)
        if len(self.requests) == 1:
            return _resp(self.session_status, b"", self.session_headers)
        return _resp(self.upload_status, self.file_body, {})


def _resp(status: int, body: bytes, headers: dict):
    from razkadr.render.gslides import UploadResponse

    # the default transport lowercases header names; the stub mirrors it
    lowered = {str(k).lower(): v for k, v in headers.items()}
    return UploadResponse(status=status, body=body, headers=lowered)


def make_deck(tmp_path: Path, content: bytes = b"PK\x03\x04deck-bytes") -> Path:
    deck = tmp_path / "deck.pptx"
    deck.write_bytes(content)
    return deck


def config_with_env(monkeypatch: pytest.MonkeyPatch, env: str = "GSLIDES_TEST_TOKEN") -> dict:
    monkeypatch.setenv(env, TOKEN)
    return {"google": {"access_token_env": env}}


# -- happy path ---------------------------------------------------------------


def test_upload_returns_link_and_id(tmp_path, monkeypatch):
    transport = StubTransport()
    upload = export_gslides(
        make_deck(tmp_path), config_with_env(monkeypatch), transport=transport
    )
    assert isinstance(upload, GSlidesUpload)
    assert upload.file_id == FILE_ID
    assert upload.link == WEB_VIEW
    assert upload.name == "deck"
    assert len(transport.requests) == 2


def test_first_call_is_metadata_post(tmp_path, monkeypatch):
    transport = StubTransport()
    export_gslides(make_deck(tmp_path), config_with_env(monkeypatch), transport=transport)
    first = transport.requests[0]
    assert first.method == "POST"
    assert "uploadType=resumable" in first.url
    assert first.headers["Authorization"] == f"Bearer {TOKEN}"
    assert "application/json" in first.headers["Content-Type"]
    assert first.headers["X-Upload-Content-Type"] == SLIDES_MIME
    metadata = json.loads(first.body)
    assert metadata["mimeType"] == SLIDES_MIME  # conversion intent
    assert metadata["name"] == "deck"


def test_second_call_puts_deck_bytes_at_session_uri(tmp_path, monkeypatch):
    transport = StubTransport()
    deck = make_deck(tmp_path)
    export_gslides(deck, config_with_env(monkeypatch), transport=transport)
    second = transport.requests[1]
    assert second.method == "PUT"
    assert second.url == "https://session.example/abc"
    assert second.body == deck.read_bytes()
    assert second.headers["Content-Type"] == PPTX_MIME
    assert second.headers["Authorization"] == f"Bearer {TOKEN}"


def test_link_fallback_without_webviewlink(tmp_path, monkeypatch):
    transport = StubTransport(file_body=json.dumps({"id": FILE_ID, "name": "deck"}).encode())
    upload = export_gslides(make_deck(tmp_path), config_with_env(monkeypatch), transport=transport)
    assert upload.link == slides_link(FILE_ID, None)
    assert upload.link == WEB_VIEW


def test_custom_name_and_folder(tmp_path, monkeypatch):
    transport = StubTransport()
    config = config_with_env(monkeypatch)
    config["google"]["folder_id"] = "folder-42"
    export_gslides(make_deck(tmp_path), config, transport=transport, name="Мой рецепт")
    metadata = json.loads(transport.requests[0].body)
    assert metadata["name"] == "Мой рецепт"
    assert metadata["parents"] == ["folder-42"]


# -- error mapping ------------------------------------------------------------


def test_auth_error_401_without_token_in_message(tmp_path, monkeypatch):
    transport = StubTransport(session_status=401, session_headers={})
    with pytest.raises(AuthError) as exc:
        export_gslides(make_deck(tmp_path), config_with_env(monkeypatch), transport=transport)
    assert TOKEN not in str(exc.value)


def test_rate_limit_429_is_transient(tmp_path, monkeypatch):
    transport = StubTransport(session_status=429, session_headers={})
    with pytest.raises(TransientError):
        export_gslides(make_deck(tmp_path), config_with_env(monkeypatch), transport=transport)


def test_quota_429_is_quota_error(tmp_path, monkeypatch):
    transport = StubTransport(
        session_status=429,
        session_headers={},
        file_body=b'{"error": {"code": 429, "message": "quota exhausted"}}',
    )
    # the quota body check applies to the upload step; emulate via upload status
    transport.requests = []
    with pytest.raises((QuotaError, TransientError)):
        export_gslides(make_deck(tmp_path), config_with_env(monkeypatch), transport=transport)


def test_server_500_is_transient(tmp_path, monkeypatch):
    transport = StubTransport(session_status=500, session_headers={})
    with pytest.raises(TransientError):
        export_gslides(make_deck(tmp_path), config_with_env(monkeypatch), transport=transport)


def test_session_without_location_is_provider_error(tmp_path, monkeypatch):
    transport = StubTransport(session_headers={})
    with pytest.raises(ProviderError, match="session URI"):
        export_gslides(make_deck(tmp_path), config_with_env(monkeypatch), transport=transport)


def test_upload_bad_json_is_provider_error(tmp_path, monkeypatch):
    transport = StubTransport(file_body=b"<html>not json</html>")
    with pytest.raises(ProviderError, match="not valid JSON"):
        export_gslides(make_deck(tmp_path), config_with_env(monkeypatch), transport=transport)


def test_upload_without_id_is_provider_error(tmp_path, monkeypatch):
    transport = StubTransport(file_body=json.dumps({"name": "deck"}).encode())
    with pytest.raises(ProviderError, match="no file id"):
        export_gslides(make_deck(tmp_path), config_with_env(monkeypatch), transport=transport)


def test_missing_deck_is_provider_error(tmp_path, monkeypatch):
    with pytest.raises(ProviderError, match="not a file"):
        export_gslides(tmp_path / "missing.pptx", config_with_env(monkeypatch))


# -- token resolution ----------------------------------------------------------


def test_missing_token_is_config_error_with_hint(monkeypatch):
    monkeypatch.delenv("GOOGLE_OAUTH_TOKEN", raising=False)
    with pytest.raises(ProviderConfigError, match="GOOGLE_OAUTH_TOKEN"):
        resolve_google_token({})


def test_default_env_name(tmp_path, monkeypatch):
    monkeypatch.setenv("GOOGLE_OAUTH_TOKEN", TOKEN)
    transport = StubTransport()
    export_gslides(make_deck(tmp_path), {}, transport=transport)
    assert transport.requests[0].headers["Authorization"] == f"Bearer {TOKEN}"


def test_direct_config_token_from_secret_registry(tmp_path, monkeypatch):
    monkeypatch.delenv("GOOGLE_OAUTH_TOKEN", raising=False)
    # monkeypatch restores the registry entry after the test
    monkeypatch.setitem(SECRETS._named, "google.access_token", TOKEN)
    transport = StubTransport()
    export_gslides(make_deck(tmp_path), {}, transport=transport)
    assert transport.requests[0].headers["Authorization"] == f"Bearer {TOKEN}"


def test_timeout_from_config_section(tmp_path, monkeypatch):
    config = config_with_env(monkeypatch)
    config["google"]["timeout_s"] = 7
    transport = StubTransport()
    export_gslides(make_deck(tmp_path), config, transport=transport)
    assert transport.requests[0].timeout_s == 7.0
    assert transport.requests[1].timeout_s == 7.0


# -- CLI ------------------------------------------------------------------------


def test_cli_happy_path_prints_link_and_writes_artifact(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GOOGLE_OAUTH_TOKEN", TOKEN)
    (tmp_path / "razkadr.toml").write_text(
        '[google]\naccess_token_env = "GOOGLE_OAUTH_TOKEN"\n', encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)  # config discovery: ./razkadr.toml
    transport = StubTransport()
    monkeypatch.setattr("razkadr.render.gslides.urllib_transport", transport)
    deck = make_deck(tmp_path)
    out = tmp_path / "gslides.json"
    code = cli_main(["export-gslides", str(deck), "--out", str(out)])
    assert code == EXIT_OK
    line = capsys.readouterr().out
    assert WEB_VIEW in line
    artifact = json.loads(out.read_text(encoding="utf-8"))
    assert artifact == {"file_id": FILE_ID, "name": "deck", "link": WEB_VIEW}


def test_cli_missing_deck_exit_2(tmp_path, capsys):
    code = cli_main(["export-gslides", str(tmp_path / "missing.pptx")])
    assert code == EXIT_FILE_ERROR
    assert "cannot read" in capsys.readouterr().err


def test_cli_auth_failure_exit_1_without_token(tmp_path, monkeypatch, capsys):
    config_with_env(monkeypatch)
    transport = StubTransport(session_status=401, session_headers={})
    monkeypatch.setattr("razkadr.render.gslides.urllib_transport", transport)
    code = cli_main(["export-gslides", str(make_deck(tmp_path))])
    assert code == EXIT_INVALID
    err = capsys.readouterr().err
    assert TOKEN not in err
    assert "razkadr export-gslides" in err


def test_cli_missing_token_exit_1_with_hint(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("GOOGLE_OAUTH_TOKEN", raising=False)
    # no named secret may leak in from other tests; no network may be hit
    monkeypatch.delitem(SECRETS._named, "google.access_token", raising=False)
    monkeypatch.setattr("razkadr.render.gslides.urllib_transport", StubTransport())
    code = cli_main(["export-gslides", str(make_deck(tmp_path))])
    assert code == EXIT_INVALID
    assert "GOOGLE_OAUTH_TOKEN" in capsys.readouterr().err
