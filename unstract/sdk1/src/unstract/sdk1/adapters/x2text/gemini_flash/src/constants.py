class GeminiFlashConfig:
    """Dictionary keys used to configure the Gemini Flash adapter."""

    API_KEY = "api_key"
    MODEL_NAME = "model_name"
    CUSTOM_PROMPT = "custom_prompt"


class GeminiFlashDefaults:
    """Default values for the Gemini Flash adapter."""

    MODEL_NAME = "gemini-2.0-flash"
    TEMPERATURE = 0.0
    MAX_OUTPUT_TOKENS = 8192
    # Page delimiter inserted by the extraction prompt and parsed to split pages.
    PAGE_DELIMITER_TEMPLATE = "<<< PAGE_{n} >>>"
    PAGE_DELIMITER_PREFIX = "<<< PAGE_"
    PAGE_DELIMITER_SUFFIX = " >>>"


class GeminiFlashMimeTypes:
    """MIME types supported by the Gemini inline data API."""

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


# System prompt sent to Gemini.  Custom instructions from the adapter config are
# appended after the final sentence so users can narrow or extend the behaviour.
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

