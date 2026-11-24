# Native Text Extractor Adapters: Gemini 2.0 Flash & Qwen 2.5 VL

This document tracks the technical design, implementation details, configuration reference, and operational guide for two native multimodal Text Extractor (`x2text`) adapters in Unstract:
1. **Gemini 2.0 Flash Extractor** (`gemini_flash`)
2. **Qwen 2.5 VL Extractor** (`qwen_vl`)

Both adapters implement Unstract's `X2TextAdapter` base contract and are auto-discovered by `X2TextRegistry`.

---

## 1. Overview & Architecture

Both adapters live in `unstract/sdk1/src/unstract/sdk1/adapters/x2text/` and follow the standard adapter layout:

```
unstract/sdk1/src/unstract/sdk1/adapters/x2text/
├── gemini_flash/
│   ├── pyproject.toml
│   ├── README.md
│   ├── src/
│   │   ├── __init__.py               # Registry metadata (is_active=True)
│   │   ├── constants.py              # Config keys, defaults, MIME map, system prompt
│   │   ├── gemini_flash.py           # GeminiFlashAdapter implementation
│   │   └── static/
│   │       └── json_schema.json      # UI form schema
│   └── tests/
│       ├── __init__.py
│       └── test_gemini_flash.py      # Unit test suite (35 tests)
│
└── qwen_vl/
    ├── pyproject.toml
    ├── README.md
    ├── src/
    │   ├── __init__.py               # Registry metadata (is_active=True)
    │   ├── constants.py              # Config keys, defaults, MIME map, system prompt
    │   ├── qwen_vl.py                # QwenVLAdapter implementation
    │   └── static/
    │       └── json_schema.json      # UI form schema
    └── tests/
        ├── __init__.py
        └── test_qwen_vl.py           # Unit test suite (44 tests)
```

### Integration Points
- **Auto-Discovery**: `X2TextRegistry` scans each subdirectory for `src/__init__.py` and inspects `metadata["is_active"]`. Both register automatically without requiring hardcoded factory entries.
- **Base Class**: Inherit from `X2TextAdapter(Adapter, ABC)`.
- **Return Type**: `TextExtractionResult(extracted_text, extraction_metadata, page_count)`.
- **Metadata Sidecar**: Saves extraction metadata and token usage to `{extract_dir}/metadata/{stem}.json`.
- **UI Form Rendering**: Driven by `static/json_schema.json` under **Settings → Text Extractor → New Text Extractor**.

---

## 2. Gemini 2.0 Flash Extractor (`gemini_flash`)

### Capabilities
- **Engine**: Google's `gemini-2.0-flash` multimodal model via the official `google-genai` SDK.
- **Supported Formats**: PDF, PNG, JPEG, GIF, WebP. Sent directly as inline multimodal parts (`genai_types.Part.from_bytes`).
- **Temperature**: `0.0`.
- **Max Output Tokens**: `8192`.

### Configuration Reference

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `adapter_name` | string | Yes | `gemini-flash-1` | Unique instance label |
| `api_key` | string (password) | No | `""` | Google AI Studio API key (falls back to `GEMINI_API_KEY` env var) |
| `model_name` | string | No | `gemini-2.0-flash` | Gemini model name (e.g. `gemini-2.0-flash`, `gemini-1.5-flash`) |
| `custom_prompt` | string (textarea) | No | `""` | Additional user instructions appended to system prompt |

### 503 Overload Mitigation & Retry Capping
Google AI Studio's free tier for `gemini-2.0-flash` frequently experiences capacity spikes resulting in `503 UNAVAILABLE: "This model is currently experiencing high demand"`.

By default, the `google-genai` client retries up to 5 times with exponential backoff up to 60 seconds each, causing a 5-minute indefinite UI hang.

**Applied Fix**:
The adapter enforces explicit HTTP retry and timeout options:
```python
http_options = genai_types.HttpOptions(
    timeout=90_000,
    retry_options=genai_types.HttpRetryOptions(
        attempts=3,
        initial_delay=1.0,
        max_delay=10.0,
    ),
)
client = genai.Client(api_key=api_key, http_options=http_options)
```
- Total retry duration is capped to ~15–20 seconds instead of 5 minutes.
- When 503 occurs, the exception message directly instructs the user to switch to `gemini-1.5-flash` or use a paid tier API key.
- `test_connection()` uses a 15-second timeout and 2 attempts to prevent UI stalls during profile verification.

---

## 3. Qwen 2.5 VL Extractor (`qwen_vl`)

### Capabilities
- **Engine**: Qwen 2.5 VL (or any OpenAI-compatible vision-language model) served on self-hosted infrastructure via vLLM, Ollama, TGI, SGLang, or LiteLLM proxy.
- **Client**: Standard `openai` SDK pointing to custom `base_url`.
- **PDF Ingestion Pipeline**: Because OpenAI-compatible vision endpoints do not accept raw PDF byte streams, the adapter automatically converts PDF pages to high-resolution PNG images in-memory using `pdfplumber` + `Pillow`:
  ```python
  page_image = page.to_image(resolution=200)
  pil_img = page_image.original.convert("RGB")
  ```
  Rendered images are sent as base64-encoded `image_url` data parts (`data:image/png;base64,...`).
- **Image Ingestion**: Single images (PNG, JPEG, GIF, WebP) are sent directly as base64 `image_url` parts.
- **Temperature**: `0.0`.
- **Max Output Tokens**: `8192`.

### Configuration Reference

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `adapter_name` | string | Yes | `qwen-vl-1` | Unique instance label |
| `base_url` | string (URI) | Yes | `http://localhost:8000/v1` | Self-hosted OpenAI-compatible API endpoint |
| `api_key` | string (password) | No | `not-needed` | API key (optional for self-hosted instances) |
| `model_name` | string | No | `Qwen/Qwen2.5-VL-72B-Instruct` | Model identifier on the serving endpoint |
| `custom_prompt` | string (textarea) | No | `""` | Additional user instructions appended to system prompt |

### Environment Variable Fallbacks
- `QWEN_BASE_URL`: Overrides empty `base_url`.
- `QWEN_API_KEY`: Overrides empty `api_key`.

---

## 4. Layout-Preserving Extraction Prompt

Both extractors use a strict system prompt to ensure tabular structures and page boundaries are preserved without conversational filler:

```
You are a precise, layout-preserving OCR and document parsing engine.
Your sole task is to faithfully transcribe the content of the provided document exactly as it appears, with these strict rules:

1. STRUCTURE
   - Preserve every table as a Markdown table (use | column | delimiters |).
   - Keep columns and rows in the original order; do not merge cells.
   - Preserve form key-value pairs as "Key: Value" on individual lines.
   - Maintain the original reading order (left-to-right, top-to-bottom).

2. PAGE BOUNDARIES
   - Insert the exact string "<<< PAGE_{n} >>>" (where {n} is the 1-based page number) on its own line at the START of each page's content — before any text from that page.
   - Do NOT omit the delimiter for single-page documents.

3. FIDELITY
   - Do not paraphrase, summarise, or add commentary.
   - Preserve all numbers, dates, currency values, and special characters exactly.
   - If a region is illegible, write "[ILLEGIBLE]" in its place.
   - Preserve blank lines where they visually separate sections.

4. FORBIDDEN
   - Do NOT include any preamble, greeting, explanation, or conclusion.
   - Do NOT wrap the output in code fences or Markdown headings.
   - Do NOT translate or correct spelling.

Output only the transcribed document text, nothing else.
```

### Page Splitting Logic
The output is parsed using regex `<<<\s*PAGE_(\d+)\s*>>>` to populate per-page dictionaries (`[{"page_number": n, "text": "..."}]`) and compute `page_count`. If no delimiters are returned, the full response is treated as page 1.

---

## 5. Dependencies & Workspace Lockfiles

The following direct dependencies support these adapters:
- `google-genai>=1.0.0`
- `openai>=1.0.0`
- `pdfplumber>=0.11.2`

Declared across the relevant workspace projects:
1. `unstract/sdk1/pyproject.toml`
2. `backend/pyproject.toml`
3. `workers/pyproject.toml`

All lockfiles are kept in sync via `uv lock`:
- `unstract/sdk1/uv.lock`
- `backend/uv.lock`
- `workers/uv.lock`
- `platform-service/uv.lock`
- Root `uv.lock`

Verify lockfile integrity at any time with:
```bash
uv lock --check && \
(cd unstract/sdk1 && uv lock --check) && \
(cd backend && uv lock --check) && \
(cd workers && uv lock --check) && \
(cd platform-service && uv lock --check)
```

---

## 6. Testing & Verification

Unit test suites run without network access or live API credentials:

```bash
# Run Gemini Flash tests (35 tests)
cd unstract/sdk1 && uv run python -m unittest discover -s src/unstract/sdk1/adapters/x2text/gemini_flash/tests

# Run Qwen VL tests (44 tests)
cd unstract/sdk1 && uv run python -m unittest discover -s src/unstract/sdk1/adapters/x2text/qwen_vl/tests
```

### Verification in Registry
Verify discovery of all text extraction adapters:
```bash
cd unstract/sdk1 && uv run python -c "
import sys; from types import ModuleType; sys.modules.setdefault('magic', ModuleType('magic'))
from unstract.sdk1.adapters.x2text.register import X2TextRegistry
adapters = {}
X2TextRegistry.register_adapters(adapters)
for aid, info in adapters.items():
    print(f'  - {info[\"metadata\"][\"adapter\"].get_name()} ({aid})')
"
```
Output:
```
  - Unstructured IO Enterprise (unstructuredenterprise|eb1b6c58-221f-4db0-a4a5-e5f9cdca44e1)
  - Gemini 2.0 Flash Extractor (geminiflash|6f1a2b3c-4d5e-6f7a-8b9c-0d1e2f3a4b5c)
  - LLMWhisperer V2 (llmwhisperer|a5e6b8af-3e1f-4a80-b006-d017e8e67f93)
  - Qwen 2.5 VL Extractor (qwenvl|a1b2c3d4-e5f6-7a8b-9c0d-e1f2a3b4c5d6)
  - No Op X2Text (noOpX2text|mp66d1op-7100-d340-9101-846fc7115676)
  - Unstructured IO Community (unstructuredcommunity|eeed506f-1875-457f-9101-846fc7115676)
  - LlamaParse (llamaparse|78860239-b3cc-4cc5-b3de-f84315f75d14)
```

