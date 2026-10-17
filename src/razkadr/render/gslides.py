"""Google Slides export: deck.pptx uploaded to Drive as a converted
presentation (naryad 025). Contract: ``deck.pptx -> link to Google Slides``.

The deck is uploaded with the Drive resumable protocol and the metadata
mimeType ``application/vnd.google-apps.presentation`` - Drive converts the
PPTX into a native Google Slides presentation; the result is the
``webViewLink`` of the created file (or the canonical link built from the
file id when Drive omits it).

OAuth: BYOK-style, the access token never lives in the config value -
``[google] access_token_env`` names an environment variable (default
``GOOGLE_OAUTH_TOKEN``); a direct ``access_token`` in the toml is read from
the secret registry (discouraged, masked like any provider key). The token
travels in the ``Authorization`` header only; error messages contain no
token, no URL and no request body. The scope needed is
``https://www.googleapis.com/auth/drive.file`` (create files the app
uploaded; nothing else).

Transport is injectable (same seam as provider adapters, naryad 037): the
default transport hits the network, tests run hermetically against stub
responses. HTTP status mapping is shared with the provider layer
(``rest.raise_for_status``): 401/403 -> AuthError, 429 -> Quota/Transient,
5xx -> TransientError. No retry loop here - the export is a user-invoked
command (like export-pdf), the caller re-runs it on a transient failure.
Refresh-token flows are out of scope: the owner provides a ready access
token (see docs/limitations.md).
"""

from __future__ import annotations

import http.client
import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from razkadr.log import SECRETS
from razkadr.providers.base import ProviderConfigError, ProviderError, TransientError
from razkadr.providers.rest import DEFAULT_TIMEOUT_S, Request, raise_for_status

DRIVE_FILES_URL = "https://www.googleapis.com/upload/drive/v3/files"
DRIVE_UPLOAD_FIELDS = "id,name,webViewLink"
SLIDES_MIME = "application/vnd.google-apps.presentation"
PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
DEFAULT_TOKEN_ENV = "GOOGLE_OAUTH_TOKEN"

PROVIDER = "gdrive"


@dataclass(frozen=True)
class UploadResponse:
    """HTTP response with headers - the resumable session URI arrives in
    the ``Location`` header of the first call (rest.Response has no headers,
    and modifying shared plumbing is outside this naryad's scope)."""

    status: int
    body: bytes
    headers: dict


Transport = Callable[[Request], UploadResponse]


def urllib_transport(request: Request) -> UploadResponse:
    """Default transport: one urllib call; failures are mapped in _send."""

    req = urllib.request.Request(
        request.url,
        data=request.body if request.method != "GET" else None,
        method=request.method,
    )
    for name, value in request.headers.items():
        req.add_header(name, value)
    try:
        with urllib.request.urlopen(req, timeout=request.timeout_s) as resp:
            return UploadResponse(
                status=resp.status,
                body=resp.read(),
                headers={k.lower(): v for k, v in resp.getheaders()},
            )
    except urllib.error.HTTPError as exc:
        return UploadResponse(
            status=exc.code, body=exc.read(), headers={k.lower(): v for k, v in exc.headers.items()}
        )


def _send(transport: Transport, request: Request) -> UploadResponse:
    """One call: transport -> network-failure mapping -> status mapping.

    Any non-provider exception from the transport (timeout, DNS, TLS,
    connection reset) is transient by definition; only the exception type
    name enters the message - no URL, no body, no token.
    """
    try:
        response = transport(request)
    except (OSError, TimeoutError, http.client.HTTPException) as exc:
        raise TransientError(f"{PROVIDER}: transport failure: {type(exc).__name__}") from None
    raise_for_status(PROVIDER, response)  # duck-typed: uses .status/.body only
    return response


def resolve_google_token(config: dict) -> str:
    """OAuth access token: env var named by ``[google] access_token_env``
    (default GOOGLE_OAUTH_TOKEN), or a direct (discouraged) ``access_token``
    value stored in the secret registry at load time. Mirrors the provider
    layer's ``resolve_key`` discipline; missing token is a configuration
    error with an actionable hint, not a network failure."""

    section = config.get("google") or {}
    env_name = section.get("access_token_env") or DEFAULT_TOKEN_ENV
    value = os.environ.get(env_name) or SECRETS.get(env_name)
    if not value:
        value = SECRETS.get("google.access_token")
    if not value:
        raise ProviderConfigError(
            f"[google] no OAuth access token: set access_token_env={env_name!r} and export "
            f"{env_name}=<token> (a token with the drive.file scope; tokens are never logged)"
        )
    SECRETS.register(value)
    return value


@dataclass(frozen=True)
class GSlidesUpload:
    """Result of a successful upload: the Drive file id, its name and the
    link to the converted Google Slides presentation."""

    file_id: str
    name: str
    link: str


def _multipart_metadata(section: dict, deck: Path, name: str | None) -> dict:
    metadata = {"name": name or deck.stem, "mimeType": SLIDES_MIME}
    folder_id = section.get("folder_id")
    if folder_id:
        metadata["parents"] = [folder_id]
    return metadata


def slides_link(file_id: str, web_view_link: str | None) -> str:
    """webViewLink when Drive reports it, else the canonical link."""

    if web_view_link:
        return web_view_link
    return f"https://docs.google.com/presentation/d/{file_id}/edit"


def export_gslides(
    deck: Path,
    config: dict,
    *,
    transport: Transport | None = None,
    timeout_s: float | None = None,
    name: str | None = None,
) -> GSlidesUpload:
    """Upload ``deck`` (PPTX) to Google Drive with conversion to Google
    Slides; returns the file id and the presentation link.

    Resumable protocol, two calls: (1) POST the JSON metadata to receive
    the session URI (``Location`` header), (2) PUT the deck bytes there.
    Conversion is requested by the metadata mimeType; PPTX bytes travel
    as-is in the second call.
    """
    if not deck.is_file():
        raise ProviderError(f"gdrive: cannot read {deck.name}: not a file")
    send = transport or urllib_transport
    section = config.get("google") or {}
    timeout = float(timeout_s if timeout_s is not None else section.get("timeout_s", DEFAULT_TIMEOUT_S))
    token = resolve_google_token(config)
    metadata = _multipart_metadata(section, deck, name)
    deck_bytes = deck.read_bytes()

    session_request = Request(
        method="POST",
        url=f"{DRIVE_FILES_URL}?uploadType=resumable&fields={DRIVE_UPLOAD_FIELDS}",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=UTF-8",
            "X-Upload-Content-Type": SLIDES_MIME,
            "X-Upload-Content-Length": str(len(deck_bytes)),
        },
        body=json.dumps(metadata).encode("utf-8"),
        timeout_s=timeout,
    )
    session_response = _send(send, session_request)
    session_uri = session_response.headers.get("location")
    if not session_uri:
        raise ProviderError(f"{PROVIDER}: no upload session URI in the response (HTTP {session_response.status})")

    upload_request = Request(
        method="PUT",
        url=session_uri,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": PPTX_MIME,
        },
        body=deck_bytes,
        timeout_s=timeout,
    )
    upload_response = _send(send, upload_request)
    try:
        data = json.loads(upload_response.body)
    except json.JSONDecodeError:
        raise ProviderError(f"{PROVIDER}: upload response is not valid JSON") from None
    if not isinstance(data, dict) or not isinstance(data.get("id"), str) or not data["id"]:
        raise ProviderError(f"{PROVIDER}: upload response has no file id")
    return GSlidesUpload(
        file_id=data["id"],
        name=data.get("name") or metadata["name"],
        link=slides_link(data["id"], data.get("webViewLink")),
    )
