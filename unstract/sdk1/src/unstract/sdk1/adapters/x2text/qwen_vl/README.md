# Qwen 2.5 VL X2Text Adapter

Layout-preserving document text extraction powered by **Qwen 2.5 VL** (or any
OpenAI-compatible vision-language model) served via
[vLLM](https://docs.vllm.ai/), [Ollama](https://ollama.com/),
[TGI](https://huggingface.co/docs/text-generation-inference/) or any endpoint
that exposes the OpenAI Chat Completions API.

## Configuration

| Field           | Required | Default                            | Description                                                |
|-----------------|----------|------------------------------------|------------------------------------------------------------|
| `adapter_name`  | Yes      | `qwen-vl-1`                       | Unique name for this adapter instance.                     |
| `base_url`      | Yes      | `http://localhost:8000/v1`         | URL of the OpenAI-compatible API endpoint.                 |
| `api_key`       | No       | `not-needed`                       | API key (most self-hosted servers don't require one).      |
| `model_name`    | No       | `Qwen/Qwen2.5-VL-72B-Instruct`   | Model identifier as registered on the serving endpoint.    |
| `custom_prompt` | No       | (empty)                            | Additional instructions appended to the system prompt.     |

## Environment Variables

| Variable         | Description                                      |
|------------------|--------------------------------------------------|
| `QWEN_BASE_URL`  | Fallback for `base_url` when the field is empty.  |
| `QWEN_API_KEY`   | Fallback for `api_key` when the field is empty.   |

## Supported Input Formats

PDF, PNG, JPEG, GIF, WebP.

> **Note:** PDFs are converted to per-page images before being sent to the
> model, since most OpenAI-compatible vision endpoints do not accept raw PDF
> bytes. Image inputs are sent directly as base64.

