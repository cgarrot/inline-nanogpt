"""The four NanoGPT nodes: image, text, video, audio.

The shape to copy from the reference extension: one ``@inline_node``-decorated
``NodeRunner`` per node, a no-arg constructor (the registrar instantiates ``cls()``),
media saved through ``ctx.takes`` or a manual ``Take`` for downloaded files.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
from typing import Any
from uuid import uuid4

from inline_core.errors import CancelledError, InlineCoreError
from inline_core.extensions.api import inline_node
from inline_core.graph.descriptor import Option, ParamField, Port, Widget
from inline_core.graph.runners import NodeResult, NodeRunner
from inline_core.graph.schema import PortKind
from inline_core.media import MediaKind
from inline_core.takes import AssetRef, Take

from . import api


def _options(*values: str) -> tuple[Option, ...]:
    return tuple(Option(v, v) for v in values)


#: Which catalog feeds each node's model dropdown.
NODE_CATALOGS: dict[type, str] = {}  # filled by apply_catalogs() at register time


def _model_id(params: dict[str, Any], fallback: str) -> str:
    """custom_model wins over the dropdown pick (an id the list may not carry)."""
    return str(params.get("custom_model", "")).strip() or str(params.get("model", "")).strip() or fallback


def apply_catalogs() -> None:
    """Rebuild each node's model param as a SELECT over the FULL NanoGPT catalog.

    Called from register(): the decorator attached a TEXT param as a placeholder; here it
    is swapped for a dropdown listing every model of that modality (disk-cached, with an
    offline fallback), plus a custom_model override. A server restart refreshes the list.
    """
    from dataclasses import replace as _replace

    from inline_core.extensions.api import DESCRIPTOR_ATTR, descriptor_of

    for cls, kind in NODE_CATALOGS.items():
        descriptor = descriptor_of(cls)
        if descriptor is None:
            continue
        ids = api.model_ids(kind)
        params = []
        for field in descriptor.params:
            if field.key == "model":
                default = field.default if field.default in ids else (ids[0] if ids else field.default)
                params.append(_replace(field, widget=Widget.SELECT, default=default,
                                       options=_options(*ids), on_face=True))
                params.append(ParamField("custom_model", "Custom model id", Widget.TEXT, "",
                                         advanced=True))
            else:
                params.append(field)
        setattr(cls, DESCRIPTOR_ATTR, _replace(descriptor, params=tuple(params)))


RESOLUTIONS = (
    "auto", "512x512", "768x768", "1024x1024", "1024x768", "768x1024",
    "1152x896", "896x1152", "1216x832", "832x1216", "1344x768", "768x1344",
    "1536x1024", "1024x1536", "1792x1024", "1024x1792", "2048x2048",
)
ASPECT_RATIOS = ("auto", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9", "9:21")
VIDEO_ASPECT_RATIOS = ("auto", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16", "9:21")
QUALITIES = ("auto", "low", "medium", "high")
VIDEO_RESOLUTIONS = ("auto", "480p", "512p", "580p", "720p", "768p", "1080p", "2k", "4k")
DURATIONS = ("auto", "4", "5", "6", "8", "10", "12", "15", "20", "30", "60")
VIDEO_MODES = ("auto", "reference-to-video", "video-edit", "video-extend")
AUDIO_FORMATS = ("mp3", "wav", "opus", "aac", "flac", "pcm")


# ------------------------------------------------------------------ helpers
def _first(values: list[Any] | None) -> Any:
    return values[0] if values else None


def _first_str(values: list[Any] | None) -> str:
    value = _first(values)
    return str(value).strip() if value is not None else ""


def _params(cls: type, node: Any) -> dict[str, Any]:
    return {**getattr(cls, "__inline_descriptor__").defaults(), **node.params}


def _merge_extra(payload: dict[str, Any], extra_json: str) -> None:
    if not (extra_json or "").strip():
        return
    try:
        extra = json.loads(extra_json)
    except ValueError as exc:
        raise api.NanoGPTError(f"Invalid extra_json: {exc}") from None
    if isinstance(extra, dict):
        payload.update(extra)


#: Caps for inline reference images. The nano-gpt.com edge rejects request bodies over ~4.5 MB
#: (HTTP 413 FUNCTION_PAYLOAD_TOO_LARGE), so a wired reference is downscaled and recompressed
#: as JPEG before being embedded - a 1536px q88 JPEG lands around 300-600 KB.
MAX_SIDE = int(os.environ.get("NANOGPT_IMAGE_MAX_SIDE", "1536"))
MAX_BYTES = int(os.environ.get("NANOGPT_IMAGE_MAX_BYTES", str(3 * 1024 * 1024)))


def _jpeg_under_budget(image: Any) -> bytes:
    """Encode as JPEG within MAX_BYTES, halving the side until it fits."""
    from PIL import Image  # noqa: F401 - clarity

    side = MAX_SIDE
    for _ in range(5):
        img = image.copy()
        img.thumbnail((side, side), Image.LANCZOS)
        for quality in (88, 75, 60):
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality, optimize=True)
            data = buffer.getvalue()
            if len(data) <= MAX_BYTES:
                return data
        side = max(320, side // 2)
    raise api.NanoGPTError(
        f"Reference image still {len(data) // 1024} KB after downsampling; refusing to build a "
        f">{MAX_BYTES // 1024} KB request (nano-gpt.com answers HTTP 413)."
    )


def _image_to_data_url(ref: Any) -> str:
    """A wired image (AssetRef or upstream Take) as a size-capped JPEG data URL."""
    from PIL import Image

    path = None
    if isinstance(ref, AssetRef) and ref.ref == "path" and ref.path:
        path = ref.path
    elif isinstance(ref, Take) and ref.uri:
        path = ref.uri.removeprefix("file://")
    if not path:
        raise api.NanoGPTError("The image input is not a readable file.")
    image = Image.open(path).convert("RGB")
    return api.bytes_to_data_url(_jpeg_under_budget(image), "image/jpeg")


def _file_take(ctx: Any, node: Any, kind: MediaKind, data: bytes, ext: str, params: dict[str, Any]) -> Take:
    """Persist downloaded bytes as an immutable take (mirrors FileTakeStore's hashing)."""
    import pathlib
    import tempfile

    base = pathlib.Path(api.DATA_DIR) if api.DATA_DIR else pathlib.Path(tempfile.gettempdir())
    folder = base / "downloads"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"nanogpt_{uuid4().hex[:12]}{ext}"
    path.write_bytes(data)
    return Take(
        id=path.stem,
        run_id=ctx.run_id,
        node_id=node.id,
        kind=kind,
        uri=str(path),
        hash=f"sha256-{hashlib.sha256(data).hexdigest()}",
        params=dict(params),
        created_at=int(time.time() * 1000),
    )


def _safe_params(params: dict[str, Any]) -> dict[str, Any]:
    """JSON-serializable params only: a Take's params end up in recipes and event payloads."""
    return {k: v for k, v in params.items() if isinstance(v, (str, int, float, bool))}


def _audio_key(model_id: str) -> str:
    """'generate_audio' ou 'generateAudio', selon le schéma détaillé du modèle."""
    try:
        for entry in api.load_catalog("video"):
            if str(entry.get("id")) == model_id:
                schema = ((entry.get("supported_parameters") or {}).get("parameters") or {})
                if "generate_audio" in schema:
                    return "generate_audio"
                if "generateAudio" in schema:
                    return "generateAudio"
                break
    except Exception:  # noqa: BLE001 - catalog best-effort
        pass
    return "generateAudio"


# ------------------------------------------------------------------ image
@inline_node(
    type="nanogpt/image",
    title="NanoGPT Image",
    category="Image",
    icon="wand",
    output_kind=MediaKind.IMAGE,
    inputs=(
        Port("prompt", "Prompt", PortKind.TEXT, required=True),
        # IMAGE_LIST: several references may be wired (fusion pipelines) — a plain IMAGE port
        # keeps only the last wire at run time (graph_build), which starves multi-reference models.
        Port("image", "Reference image(s)", PortKind.IMAGE_LIST, required=False),
    ),
    outputs=(Port("image", "Image", PortKind.IMAGE),),
    params=(
        ParamField("model", "Model id", Widget.TEXT, "krea-2/turbo", on_face=True),
        ParamField("resolution", "Resolution", Widget.SELECT, "auto", options=_options(*RESOLUTIONS)),
        ParamField("aspect_ratio", "Aspect ratio", Widget.SELECT, "auto", options=_options(*ASPECT_RATIOS)),
        ParamField("quality", "Quality", Widget.SELECT, "auto", options=_options(*QUALITIES)),
        ParamField("n", "Images", Widget.NUMBER, 1, min=1, max=10, step=1),
        ParamField("seed", "Seed", Widget.SEED, -1),
        ParamField("extra_json", "Extra JSON", Widget.TEXTAREA, "", advanced=True),
    ),
)
class NanoGPTImageNode(NodeRunner):
    """Text-to-image or image-to-image via any of NanoGPT's image models."""

    produces_takes = True

    def run(self, node: Any, inputs: dict[str, list[Any]], ctx: Any) -> NodeResult:
        prompt = _first_str(inputs.get("prompt"))
        if not prompt:
            raise InlineCoreError("NanoGPT Image needs a prompt.")
        params = _params(NanoGPTImageNode, node)
        payload: dict[str, Any] = {
            "model": _model_id(params, "krea-2/turbo"),
            "prompt": prompt,
            "n": max(1, int(params["n"])),
        }
        for key in ("resolution", "aspect_ratio", "quality"):
            if params.get(key) and params[key] != "auto":
                payload[key] = params[key]
        seed = int(params.get("seed", -1))
        if seed >= 0:
            payload["seed"] = seed
        image_refs = list(inputs.get("image") or [])
        if image_refs:
            # ALL wired references ride along — a fusion node may take four study sheets.
            # Two wire formats, because models differ: `image` (OpenAI gpt-image style —
            # flare/edit counts references there and 400s without them) and
            # `input_references` (the shape other models accept). Unknown keys are ignored.
            urls = [_image_to_data_url(ref) for ref in image_refs[:8]]
            payload["image"] = urls
            payload["input_references"] = [
                {"type": "image_url", "image_url": {"url": url}} for url in urls
            ]
        _merge_extra(payload, str(params.get("extra_json", "")))

        data = api.post_json("/v1/images", payload, timeout=900)
        items = (data or {}).get("data") or []
        if not items:
            raise api.NanoGPTError(f"Response without an image: {str(data)[:300]}")

        from PIL import Image

        takes: list[Take] = []
        safe = _safe_params(params)
        for index, item in enumerate(items):
            raw = _item_bytes(item)
            if raw is None:
                continue
            takes.append(ctx.takes.save(ctx.run_id, node.id, Image.open(io.BytesIO(raw)), safe))
        if not takes:
            raise api.NanoGPTError(f"No usable image in the response: {str(items[0])[:200]}")
        return NodeResult(outputs={"image": takes[0]}, takes=takes)


def _item_bytes(item: Any) -> bytes | None:
    if isinstance(item, dict):
        if item.get("b64_json"):
            return api.b64_to_bytes(item["b64_json"])
        url = item.get("url") or item.get("image_url") or item.get("image")
        if isinstance(url, dict):
            url = url.get("url")
        if isinstance(url, str) and url.startswith("http"):
            return api.download(url)
    elif isinstance(item, str):
        return api.download(item) if item.startswith("http") else api.b64_to_bytes(item)
    return None


# ------------------------------------------------------------------ text
@inline_node(
    type="nanogpt/text",
    title="NanoGPT Text",
    category="Generate",
    icon="type",
    inputs=(
        Port("prompt", "Prompt", PortKind.TEXT, required=True),
        Port("image", "Image (vision)", PortKind.IMAGE, required=False),
    ),
    outputs=(Port("text", "Text", PortKind.TEXT),),
    params=(
        ParamField("model", "Model id", Widget.TEXT, "openai/gpt-oss-120b", on_face=True),
        ParamField("system", "System prompt", Widget.TEXTAREA, ""),
        ParamField("temperature", "Temperature", Widget.NUMBER, 1.0, min=0.0, max=2.0, step=0.05),
        ParamField("max_tokens", "Max tokens", Widget.NUMBER, 2048, min=1, max=200000, step=1),
        ParamField("extra_json", "Extra JSON", Widget.TEXTAREA, "", advanced=True),
    ),
)
class NanoGPTTextNode(NodeRunner):
    """Chat completion via NanoGPT's OpenAI-compatible endpoint (vision included)."""

    produces_takes = False

    def run(self, node: Any, inputs: dict[str, list[Any]], ctx: Any) -> NodeResult:
        prompt = _first_str(inputs.get("prompt"))
        if not prompt:
            raise InlineCoreError("NanoGPT Text needs a prompt.")
        params = _params(NanoGPTTextNode, node)
        messages: list[dict[str, Any]] = []
        system = str(params.get("system", "")).strip()
        if system:
            messages.append({"role": "system", "content": system})
        image_ref = _first(inputs.get("image"))
        if image_ref is not None:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(image_ref)}},
                ],
            })
        else:
            messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": _model_id(params, "openai/gpt-oss-120b"),
            "messages": messages,
            "temperature": float(params["temperature"]),
            "max_tokens": int(params["max_tokens"]),
        }
        _merge_extra(payload, str(params.get("extra_json", "")))

        data = api.post_json("/v1/chat/completions", payload, timeout=900)
        choices = (data or {}).get("choices") or []
        if not choices:
            raise api.NanoGPTError(f"Response without a choice: {str(data)[:300]}")
        text = choices[0].get("message", {}).get("content", "")
        if isinstance(text, list):
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
        return NodeResult(outputs={"text": text or ""})


# ------------------------------------------------------------------ video
@inline_node(
    type="nanogpt/video",
    title="NanoGPT Video",
    category="Video",
    icon="film",
    output_kind=MediaKind.VIDEO,
    inputs=(
        Port("prompt", "Prompt", PortKind.TEXT, required=True),
        Port("image", "Start image", PortKind.IMAGE, required=False),
        Port("last_image", "End image (last frame)", PortKind.IMAGE, required=False),
    ),
    outputs=(Port("video", "Video", PortKind.VIDEO),),
    params=(
        ParamField("model", "Model id", Widget.TEXT, "lightricks-ltx-2-fast", on_face=True),
        ParamField("resolution", "Resolution", Widget.SELECT, "auto", options=_options(*VIDEO_RESOLUTIONS)),
        ParamField("aspect_ratio", "Aspect ratio", Widget.SELECT, "auto", options=_options(*VIDEO_ASPECT_RATIOS)),
        ParamField("duration", "Duration (s)", Widget.SELECT, "auto", options=_options(*DURATIONS)),
        ParamField("negative_prompt", "Negative prompt", Widget.TEXTAREA, ""),
        ParamField("seed", "Seed", Widget.SEED, -1),
        # Communs aux catalogues : generateAudio (23 modeles), enable_prompt_expansion (19),
        # enable_web_search (13). Envoyes seulement actives ; le reste via Extra JSON.
        ParamField("generate_audio", "Audio soundtrack (⚠ coût — OFF si sound design en post)", Widget.BOOLEAN, False),
        ParamField("prompt_expansion", "Prompt expansion", Widget.BOOLEAN, False),
        # Parametres specifiques (spec /generate-video) : envoyes seulement s'ils sont remplis.
        ParamField("mode", "Mode", Widget.SELECT, "auto", options=_options(*VIDEO_MODES), advanced=True),
        ParamField("voice", "Voice", Widget.TEXT, "", advanced=True),
        ParamField("reference_images", "Reference image URLs", Widget.TEXTAREA, "", advanced=True),
        ParamField("reference_videos", "Reference video URLs", Widget.TEXTAREA, "", advanced=True),
        ParamField("reference_audios", "Reference audio URLs", Widget.TEXTAREA, "", advanced=True),
        ParamField("lora_url_1", "LoRA 1 URL", Widget.TEXT, "", advanced=True),
        ParamField("lora_scale_1", "LoRA 1 strength", Widget.NUMBER, 1.0, min=0.0, max=2.0, step=0.1, advanced=True),
        ParamField("lora_url_2", "LoRA 2 URL", Widget.TEXT, "", advanced=True),
        ParamField("lora_scale_2", "LoRA 2 strength", Widget.NUMBER, 1.0, min=0.0, max=2.0, step=0.1, advanced=True),
        ParamField("lora_url_3", "LoRA 3 URL", Widget.TEXT, "", advanced=True),
        ParamField("lora_scale_3", "LoRA 3 strength", Widget.NUMBER, 1.0, min=0.0, max=2.0, step=0.1, advanced=True),
        ParamField("show_explicit", "Allow explicit content", Widget.BOOLEAN, False, advanced=True),
        ParamField("web_search", "Web search", Widget.BOOLEAN, False, advanced=True),
        ParamField("extra_json", "Model-specific params (JSON)", Widget.TEXTAREA, "", advanced=True),
    ),
)
class NanoGPTVideoNode(NodeRunner):
    """Text-to-video or image-to-video (LTX, MiniMax, Grok...). Async: submits, polls, downloads."""

    produces_takes = True

    def run(self, node: Any, inputs: dict[str, list[Any]], ctx: Any) -> NodeResult:
        prompt = _first_str(inputs.get("prompt"))
        image_ref = _first(inputs.get("image"))
        if not prompt and image_ref is None:
            raise InlineCoreError("NanoGPT Video needs a prompt or a start image.")
        params = _params(NanoGPTVideoNode, node)
        payload: dict[str, Any] = {
            "model": _model_id(params, "lightricks-ltx-2-fast"),
            "prompt": prompt,
        }
        if params.get("resolution") and params["resolution"] != "auto":
            payload["resolution"] = params["resolution"]
        if params.get("aspect_ratio") and params["aspect_ratio"] != "auto":
            payload["aspect_ratio"] = params["aspect_ratio"]
        if params.get("duration") and params["duration"] != "auto":
            payload["duration"] = str(params["duration"])
        negative = str(params.get("negative_prompt", "")).strip()
        if negative:
            payload["negative_prompt"] = negative
        seed = int(params.get("seed", -1))
        if seed >= 0:
            payload["seed"] = seed
        if params.get("generate_audio"):
            # Le nom du param depend du modele (generateAudio vs generate_audio) : le schéma
            # détaillé du catalogue tranche, sinon orthographe majoritaire.
            payload[_audio_key(str(payload["model"]))] = True
        if params.get("prompt_expansion"):
            payload["enable_prompt_expansion"] = True
        if params.get("web_search"):
            payload["enable_web_search"] = True
        if params.get("mode") and params["mode"] != "auto":
            payload["mode"] = params["mode"]
        voice = str(params.get("voice", "")).strip()
        if voice:
            payload["voice"] = voice
        for key in ("reference_images", "reference_videos", "reference_audios"):
            value = str(params.get(key, "")).strip()
            if value:
                payload[key] = value
        for index in (1, 2, 3):
            url = str(params.get(f"lora_url_{index}", "")).strip()
            if url:
                payload[f"lora_url_{index}"] = url
                payload[f"lora_scale_{index}"] = float(params.get(f"lora_scale_{index}", 1.0))
        if params.get("show_explicit"):
            payload["showExplicitContent"] = True
        if image_ref is not None:
            payload["imageDataUrl"] = _image_to_data_url(image_ref)
        last_ref = _first(inputs.get("last_image"))
        if last_ref is not None:
            payload["last_image"] = _image_to_data_url(last_ref)
        _merge_extra(payload, str(params.get("extra_json", "")))

        run_id = api.submit_job(payload)
        try:
            final = api.poll_job(run_id, cancelled=lambda: ctx.cancel.cancelled)
        except KeyboardInterrupt as exc:
            raise CancelledError("Run cancelled.") from exc
        assets = api.job_assets(final)
        if not assets:
            raise api.NanoGPTError(f"Completed job without a file: {str(final)[:300]}")
        url, fmt = assets[0]
        raw = api.download(url, timeout=1200)
        ext = "." + (fmt.strip(". ") or url.rsplit(".", 1)[-1].split("?")[0] or "mp4").lower()
        if ext not in (".mp4", ".webm", ".mov", ".gif"):
            ext = ".mp4"
        take = _file_take(ctx, node, MediaKind.VIDEO, raw, ext, _safe_params(params))
        return NodeResult(outputs={"video": take}, takes=[take])


# ------------------------------------------------------------------ audio
@inline_node(
    type="nanogpt/audio",
    title="NanoGPT Speech",
    category="Audio",
    icon="audio",
    output_kind=MediaKind.AUDIO,
    inputs=(Port("text", "Text", PortKind.TEXT, required=True),),
    outputs=(Port("audio", "Audio", PortKind.AUDIO),),
    params=(
        ParamField("model", "Model id", Widget.TEXT, "Kokoro-82m", on_face=True),
        ParamField("voice", "Voice", Widget.TEXT, "alloy"),
        ParamField("speed", "Speed", Widget.NUMBER, 1.0, min=0.25, max=4.0, step=0.05),
        ParamField("instructions", "Style instructions", Widget.TEXTAREA, ""),
        ParamField("response_format", "Format", Widget.SELECT, "mp3", options=_options(*AUDIO_FORMATS)),
        ParamField("extra_json", "Extra JSON", Widget.TEXTAREA, "", advanced=True),
    ),
)
class NanoGPTAudioNode(NodeRunner):
    """Text-to-speech via /v1/audio/speech (OpenAI, Kokoro, ElevenLabs, music models...)."""

    produces_takes = True

    def run(self, node: Any, inputs: dict[str, list[Any]], ctx: Any) -> NodeResult:
        text = _first_str(inputs.get("text"))
        if not text:
            raise InlineCoreError("NanoGPT Speech needs text.")
        params = _params(NanoGPTAudioNode, node)
        fmt = str(params.get("response_format", "mp3"))
        payload: dict[str, Any] = {
            "model": _model_id(params, "Kokoro-82m"),
            "input": text,
            "voice": str(params.get("voice", "alloy")),
            "response_format": fmt,
            "speed": float(params.get("speed", 1.0)),
        }
        instructions = str(params.get("instructions", "")).strip()
        if instructions:
            payload["instructions"] = instructions
        _merge_extra(payload, str(params.get("extra_json", "")))

        raw = api.post_raw("/v1/audio/speech", payload, timeout=900)
        if not raw:
            raise api.NanoGPTError("Empty response from /v1/audio/speech.")
        take = _file_take(ctx, node, MediaKind.AUDIO, raw, f".{fmt}", _safe_params(params))
        return NodeResult(outputs={"audio": take}, takes=[take])


# Which catalog feeds each node's model dropdown (module level: classes are defined above).
NODE_CATALOGS[NanoGPTImageNode] = "image"
NODE_CATALOGS[NanoGPTTextNode] = "text"
NODE_CATALOGS[NanoGPTVideoNode] = "video"
NODE_CATALOGS[NanoGPTAudioNode] = "audio"
