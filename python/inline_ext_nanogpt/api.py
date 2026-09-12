"""HTTP client for the NanoGPT API (https://nano-gpt.com).

Same key resolution as the ComfyUI-NanoGPT node, so one key serves both apps:
  1. NANOGPT_API_KEY environment variable
  2. ~/.config/nano-gpt/api_key            (chmod 600, recommended)
  3. <extension data dir>/api_key.txt

All modalities go through here: text, image, video, audio.
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

API_BASE = os.environ.get("NANOGPT_API_BASE", "https://nano-gpt.com/api").rstrip("/")

#: Set at register() time by the registrar (the extension's private scratch dir).
DATA_DIR: pathlib.Path | None = None


class NanoGPTError(RuntimeError):
    """A user-readable API error."""


# --------------------------------------------------------------------- api key
def api_key() -> str:
    key = os.environ.get("NANOGPT_API_KEY", "").strip()
    if key:
        return key
    candidates = [pathlib.Path.home() / ".config" / "nano-gpt" / "api_key"]
    if DATA_DIR is not None:
        candidates.append(pathlib.Path(DATA_DIR) / "api_key.txt")
    for path in candidates:
        try:
            if path.is_file():
                key = path.read_text().strip()
                if key:
                    return key
        except OSError:
            continue
    raise NanoGPTError(
        "No NanoGPT API key found. Set NANOGPT_API_KEY or write the key to "
        f"{pathlib.Path.home() / '.config' / 'nano-gpt' / 'api_key'} (chmod 600)."
    )


def has_api_key() -> bool:
    try:
        api_key()
        return True
    except NanoGPTError:
        return False


def save_api_key(key: str) -> pathlib.Path | None:
    key = key.strip()
    if not key or DATA_DIR is None:
        raise NanoGPTError("Empty key or no extension data dir.")
    path = pathlib.Path(DATA_DIR) / "api_key.txt"
    path.write_text(key, encoding="utf-8")
    os.chmod(path, 0o600)
    return path


# ------------------------------------------------------------------ transport
def _request(
    path: str,
    payload: Any = None,
    method: str = "POST",
    timeout: int = 300,
    raw: bool = False,
    attempts: int = 3,
) -> Any:
    url = path if path.startswith("http") else API_BASE + path
    headers = {"Authorization": f"Bearer {api_key()}", "User-Agent": "inline-nanogpt"}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"

    last: Exception | None = None
    for attempt in range(attempts):
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
                return body if raw else (json.loads(body) if body else {})
        except urllib.error.HTTPError as exc:
            body = exc.read()
            detail = body.decode("utf-8", "replace")[:500]
            # 4xx (except 429) is definitive - no point retrying.
            if 400 <= exc.code < 500 and exc.code != 429:
                raise NanoGPTError(f"HTTP {exc.code} on {path}: {detail}") from None
            last = NanoGPTError(f"HTTP {exc.code} on {path}: {detail}")
        except Exception as exc:  # timeout, DNS, reset...
            last = NanoGPTError(f"{type(exc).__name__} on {path}: {exc}")
        if attempt < attempts - 1:
            time.sleep(1.5 * (attempt + 1))
    raise NanoGPTError(str(last))


def post_json(path: str, payload: Any, timeout: int = 300) -> Any:
    return _request(path, payload, "POST", timeout)


def post_raw(path: str, payload: Any, timeout: int = 600) -> bytes:
    return _request(path, payload, "POST", timeout, raw=True)


def get_json(path: str, timeout: int = 60) -> Any:
    return _request(path, None, "GET", timeout)


def download(url: str, timeout: int = 600) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "inline-nanogpt"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def b64_to_bytes(value: str) -> bytes:
    if value.startswith("data:"):
        value = value.split(",", 1)[1]
    return base64.b64decode(value)


def bytes_to_data_url(data: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


# ---------------------------------------------------------------- async jobs (video)
def submit_job(payload: dict[str, Any]) -> str:
    """Submit a video job; returns the runId (HTTP 202 expected)."""
    try:
        data = post_json("/generate-video", payload, timeout=180)
    except NanoGPTError as exc:
        if "HTTP 202" in str(exc):  # 202 = accepted; urllib classes it as an error
            raise NanoGPTError("Job submission returned 202 without a usable body.") from None
        raise
    run_id = (data or {}).get("runId") or (data or {}).get("id") or (data or {}).get("requestId")
    if not run_id:
        raise NanoGPTError(f"No runId in the response: {json.dumps(data)[:300]}")
    return str(run_id)


def poll_job(
    run_id: str,
    cancelled: Callable[[], bool] | None = None,
    timeout_s: int = 1800,
    interval_s: float = 5.0,
) -> dict[str, Any]:
    """Poll the unified status endpoint until COMPLETED/FAILED. Returns the final payload."""
    start = time.time()
    while True:
        if cancelled is not None and cancelled():
            raise KeyboardInterrupt("cancelled")
        payload = get_json(f"/video/status?requestId={urllib.parse.quote(run_id)}", timeout=60)
        data = payload.get("data") if isinstance(payload, dict) and isinstance(payload.get("data"), dict) else payload
        status = str((data or {}).get("status", "")).upper()
        if status == "COMPLETED":
            return data or {}
        if status in ("FAILED", "ERROR", "CANCELLED"):
            raise NanoGPTError(f"Job {run_id} {status}: {str(data)[:300]}")
        if time.time() - start > timeout_s:
            raise NanoGPTError(f"Job {run_id} timed out after {timeout_s}s (status {status or 'UNKNOWN'}).")
        time.sleep(interval_s)


def job_assets(final: dict[str, Any]) -> list[tuple[str, str]]:
    """(url, extension) pairs out of a completed job payload, covering the known response shapes."""
    urls: list[tuple[str, str]] = []
    for key in ("output", "assets", "results", "videos", "images", "files", "artifacts"):
        value = final.get(key)
        if isinstance(value, str) and value.startswith("http"):
            urls.append((value, ""))
        elif isinstance(value, dict):
            url = value.get("url") or value.get("video_url") or value.get("image_url")
            if isinstance(url, str) and url.startswith("http"):
                urls.append((url, str(value.get("format") or value.get("type") or "")))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.startswith("http"):
                    urls.append((item, ""))
                elif isinstance(item, dict):
                    url = item.get("url") or item.get("video_url") or item.get("image_url")
                    if isinstance(url, str) and url.startswith("http"):
                        urls.append((url, str(item.get("format") or item.get("type") or "")))
    return urls


def balance() -> str:
    data = post_json("/check-balance", {}, timeout=30)
    return str((data or {}).get("usd_balance", "?"))


# ---------------------------------------------------------------- model catalogs
CATALOGS = {
    "text": "/v1/models",
    "image": "/v1/images/models",
    "video": "/v1/video-models",
    "audio": "/v1/audio-models",
}

#: Offline fallback: one reliable id per modality (all validated against the live API).
FALLBACK: dict[str, list[str]] = {
    "text": ["openai/gpt-oss-120b", "openai/gpt-5.6-sol", "deepseek-chat",
             "anthropic/claude-sonnet-5", "google/gemini-3.1-pro-preview"],
    "image": ["krea-2/turbo", "flux-2-dev", "gpt-image-2"],
    "video": ["lightricks-ltx-2-fast", "minimax-hailuo-02", "grok-imagine-video"],
    "audio": ["Kokoro-82m", "tts-1", "gpt-4o-mini-tts"],
}

CACHE_TTL = 12 * 3600


def _catalog_file() -> pathlib.Path | None:
    if DATA_DIR is None:
        return None
    return pathlib.Path(DATA_DIR) / "model_cache.json"


def _read_disk_cache() -> dict[str, Any]:
    import json as _json

    file = _catalog_file()
    try:
        return _json.loads(file.read_text()) if file else {}
    except (OSError, ValueError):
        return {}


def _write_disk_cache(cache: dict[str, Any]) -> None:
    import json as _json

    file = _catalog_file()
    if file is None:
        return
    try:
        file.write_text(_json.dumps(cache))
    except OSError:
        pass


def _fetch_catalog(kind: str) -> list[dict[str, Any]]:
    """One modal catalog. Short timeout, single attempt: this runs at server start."""
    path = CATALOGS[kind]
    try:
        data = _request(f"{path}?detailed=true" if kind == "text" else path,
                        None, "GET", timeout=10, attempts=1)
    except NanoGPTError:
        if kind == "text":  # the detailed flag is not always accepted
            data = _request(path, None, "GET", timeout=10, attempts=1)
        else:
            raise
    if isinstance(data, dict):
        data = data.get("data") or []
    return [m for m in data if isinstance(m, dict) and m.get("id")]


def load_catalog(kind: str, *, refresh: bool = False) -> list[dict[str, Any]]:
    """A modal catalog, disk-cached for CACHE_TTL. Never raises; degrades to []."""
    disk = _read_disk_cache()
    catalogs: dict[str, Any] = dict(disk.get("catalogs") or {})
    fetched_at = float(disk.get("fetched_at") or 0)
    fresh = (time.time() - fetched_at) < CACHE_TTL
    if kind in catalogs and fresh and not refresh:
        return catalogs[kind]
    if (kind not in catalogs or not fresh) and has_api_key():
        try:
            catalogs[kind] = _fetch_catalog(kind)
            _write_disk_cache({"fetched_at": time.time(), "catalogs": catalogs})
            return catalogs[kind]
        except Exception:
            pass  # keep whatever we have
    return catalogs.get(kind) or []


def model_ids(kind: str, categories: list[str] | None = None) -> list[str]:
    """Sorted unique model ids for a modal catalog, falling back offline.

    ``categories`` filters the audio catalog by its category field (e.g. TTS only),
    mirroring the ComfyUI node. All of FALLBACK[kind] is always kept available."""
    entries = load_catalog(kind)
    if categories:
        wanted = {c.lower() for c in categories}
        entries = [e for e in entries if str(e.get("category", "")).lower() in wanted]
    ids = [str(e["id"]) for e in entries]
    for fallback in FALLBACK.get(kind, []):
        if fallback not in ids:
            ids.append(fallback)
    return sorted(dict.fromkeys(ids))
