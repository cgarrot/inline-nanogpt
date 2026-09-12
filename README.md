# inline-nanogpt

OpenChar Studio / Inline Studio extension: **NanoGPT** hosted models
(https://nano-gpt.com) as canvas nodes.

Nodes:

| Node | What it does | Endpoint |
| ---- | ------------ | -------- |
| `nanogpt/image` | text-to-image / image-to-image, ~230 models | `POST /v1/images` |
| `nanogpt/text` | chat completion (vision capable) | `POST /v1/chat/completions` |
| `nanogpt/video` | text/image-to-video, async job | `POST /generate-video` + `/video/status` |
| `nanogpt/audio` | text-to-speech / music | `POST /v1/audio/speech` |

## API key

Same resolution chain as the ComfyUI-NanoGPT node, so one key serves both apps:

1. `NANOGPT_API_KEY` environment variable
2. `~/.config/nano-gpt/api_key` (chmod 600 — recommended)
3. `<extension data dir>/api_key.txt`

## Install

From the app: Extensions -> install from source `file:///home/cgarrot/inline-nanogpt`
(or push this repo to GitHub and use the https URL). Accept the `network-egress`
consent — the extension contacts `nano-gpt.com`, which is the point.

## Model ids

Model ids come from a dropdown over the FULL catalog (disk-cached 12h; restart refreshes). Custom ids go in **Custom model id**. (e.g. `krea-2/turbo`, `google/gemini-3-flash`,
`lightricks-ltx-2-fast`, `Kokoro-82m`). Full catalogs:
`https://nano-gpt.com/api/v1/images/models`, `/v1/models`, `/v1/video-models`,
`/v1/audio-models`. Anything model-specific goes in **Extra JSON** and is merged
into the request.

MIT licensed.
