class QwenVLConfig:
    """Dictionary keys used to configure the Qwen VL adapter."""

    BASE_URL = "base_url"
    API_KEY = "api_key"
    MODEL_NAME = "model_name"
    CUSTOM_PROMPT = "custom_prompt"
    MAX_TOKENS = "max_tokens"


class QwenVLDefaults:
    """Default values for the Qwen VL adapter."""

    MODEL_NAME = "Qwen/Qwen2.5-VL-72B-Instruct"
    BASE_URL = "http://localhost:8000/v1"
    API_KEY = "not-needed"
    TEMPERATURE = 0.0
    # 4096 tokens per page fits comfortably within an 8192 context window
    # alongside ~1200-1800 vision prompt tokens.
    MAX_TOKENS = 4096
    # Maximum dimension (width or height) for image inputs to avoid token explosion.
    MAX_IMAGE_DIM = 1536
    # PDF rasterization resolution in DPI (150 DPI is crisp for OCR while compact).
    PDF_RENDER_DPI = 150
    # Page delimiter inserted by the extraction prompt and parsed to split pages.
    PAGE_DELIMITER_TEMPLATE = "<<< PAGE_{n} >>>"
    PAGE_DELIMITER_PREFIX = "<<< PAGE_"
    PAGE_DELIMITER_SUFFIX = " >>>"


class QwenVLMimeTypes:
    """MIME types supported by the adapter."""

    PDF = "application/pdf"
    PNG = "image/png"
    JPEG = "image/jpeg"
    JPG = "image/jpeg"  # alias
    GIF = "image/gif"
    WEBP = "image/webp"

    # Map lowercase file extensions → MIME type
    EXTENSION_MAP: dict[str, str] = {
        ".pdf": PDF,
        ".png": PNG,
        ".jpeg": JPEG,
        ".jpg": JPEG,
        ".gif": GIF,
        ".webp": WEBP,
    }


# System prompt sent to the model.  Custom instructions from the adapter config
# are appended after the final sentence so users can narrow or extend the
# behaviour.
EXTRACTION_SYSTEM_PROMPT = """\
You are a precise, layout-preserving OCR and document parsing engine.
Your sole task is to faithfully transcribe the content of the provided document \
exactly as it appears, with these strict rules:

1. STRUCTURE
   - Preserve every table as a Markdown table (use | column | delimiters |).
   - Keep columns and rows in the original order; do not merge cells.
   - Preserve form key-value pairs as "Key: Value" on individual lines.
   - Maintain the original reading order (left-to-right, top-to-bottom).

2. PAGE BOUNDARIES
   - Insert the exact string "<<< PAGE_{n} >>>" (where {n} is the 1-based page \
number) on its own line at the START of each page's content — before any text \
from that page.
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

Output only the transcribed document text, nothing else.\
"""

