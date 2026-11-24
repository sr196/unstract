"""Qwen 2.5 VL X2Text adapter.

Sends document pages as base64-encoded images to an OpenAI-compatible
vision-language model endpoint (vLLM, Ollama, TGI, etc.) and uses a
layout-preserving extraction prompt to produce structured text output that maps
cleanly to Unstract's ``TextExtractionResult`` contract.

PDFs are converted to per-page images internally because most OpenAI-compatible
vision endpoints do not accept raw PDF bytes.  Image inputs (PNG, JPEG, etc.)
are sent directly.

Supported input formats: PDF, PNG, JPEG, GIF, WebP.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any

from unstract.sdk1.adapters.exceptions import AdapterError
from unstract.sdk1.adapters.x2text.dto import TextExtractionResult
from unstract.sdk1.adapters.x2text.qwen_vl.src.constants import (
    EXTRACTION_SYSTEM_PROMPT,
    QwenVLConfig,
    QwenVLDefaults,
    QwenVLMimeTypes,
)
from unstract.sdk1.adapters.x2text.x2text_adapter import X2TextAdapter
from unstract.sdk1.file_storage import FileStorage, FileStorageProvider

logger = logging.getLogger(__name__)


class QwenVLAdapter(X2TextAdapter):
    """Text extractor adapter backed by Qwen 2.5 VL via an OpenAI-compatible API.

    Connects to a self-hosted vision-language model exposed through an
    OpenAI-compatible chat completions endpoint and returns a
    ``TextExtractionResult`` that is structurally identical to what the
    LLMWhisperer and Gemini Flash adapters produce, so callers require no
    special-casing.
    """

    def __init__(self, settings: dict[str, Any]) -> None:
        """Initialise the Qwen VL text extraction adapter.

        Args:
            settings: Adapter configuration dict.  Recognised keys are
                defined in ``QwenVLConfig``.  ``base_url`` and ``api_key``
                fall back to ``QWEN_BASE_URL`` / ``QWEN_API_KEY`` environment
                variables when absent or empty.
        """
        super().__init__("QwenVL")
        self.config = settings

    SCHEMA_PATH = f"{os.path.dirname(__file__)}/static/json_schema.json"

    # ------------------------------------------------------------------
    # Static adapter metadata (called by the registry)
    # ------------------------------------------------------------------

    @staticmethod
    def get_id() -> str:
        return "qwenvl|a1b2c3d4-e5f6-7a8b-9c0d-e1f2a3b4c5d6"

    @staticmethod
    def get_name() -> str:
        return "Qwen 2.5 VL Extractor"

    @staticmethod
    def get_description() -> str:
        return (
            "Layout-preserving document extraction powered by Qwen 2.5 VL "
            "via an OpenAI-compatible API endpoint"
        )

    @staticmethod
    def get_icon() -> str:
        return "/icons/adapter-icons/QwenVL.png"

    @staticmethod
    def get_doc_url() -> str:
        return "https://docs.unstract.com/unstract/unstract_platform/adapters/text_extractors/qwen_vl_text_extractor/"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_base_url(self) -> str:
        """Return the effective API base URL, raising AdapterError if absent."""
        url = self.config.get(QwenVLConfig.BASE_URL) or os.getenv("QWEN_BASE_URL")
        if not url:
            raise AdapterError(
                "No OpenAI-compatible API URL configured.  Set 'base_url' in the "
                "adapter profile or export the QWEN_BASE_URL environment variable."
            )
        return url.rstrip("/")

    def _resolve_api_key(self) -> str:
        """Return the effective API key, defaulting to 'not-needed'."""
        key = self.config.get(QwenVLConfig.API_KEY) or os.getenv("QWEN_API_KEY")
        if not key:
            return QwenVLDefaults.API_KEY
        return key

    def _resolve_model_name(self) -> str:
        return (
            self.config.get(QwenVLConfig.MODEL_NAME)
            or QwenVLDefaults.MODEL_NAME
        )

    def _resolve_max_tokens(self) -> int:
        val = self.config.get(QwenVLConfig.MAX_TOKENS)
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                pass
        return QwenVLDefaults.MAX_TOKENS

    def _build_system_prompt(self) -> str:
        """Combine the base extraction prompt with any user-supplied additions."""
        base = EXTRACTION_SYSTEM_PROMPT
        custom = (self.config.get(QwenVLConfig.CUSTOM_PROMPT) or "").strip()
        if custom:
            base = f"{base}\n\nADDITIONAL INSTRUCTIONS FROM USER:\n{custom}"
        return base

    @staticmethod
    def _detect_mime_type(file_path: str, fs: FileStorage) -> str:
        """Determine the MIME type for *file_path*.

        Resolution order:
        1. File extension (fast, works for most inputs).
        2. ``fs.mime_type()`` (content sniffing via python-magic or fsspec).
        3. Hard fallback to ``application/pdf``.
        """
        ext = Path(file_path).suffix.lower()
        mime = QwenVLMimeTypes.EXTENSION_MAP.get(ext)
        if mime:
            return mime
        try:
            mime = fs.mime_type(path=file_path)
            if mime in QwenVLMimeTypes.EXTENSION_MAP.values():
                return mime
        except Exception:
            pass
        logger.warning(
            "QwenVL: could not detect MIME type for '%s', falling back to PDF",
            file_path,
        )
        return QwenVLMimeTypes.PDF

    @staticmethod
    def _split_pages(raw_text: str) -> list[dict[str, Any]]:
        """Split the model output on page-delimiter markers into per-page dicts.

        Each dict has keys ``page_number`` (int, 1-based) and ``text`` (str).
        When no delimiters are found the entire text is treated as page 1.
        Delimiters are matched case-insensitively so minor model formatting
        variations are tolerated.

        Returns:
            List of ``{"page_number": n, "text": "..."}`` dicts.
        """
        # Match: <<< PAGE_N >>> (N = one or more digits, optional whitespace)
        pattern = re.compile(
            r"<<<\s*PAGE_(\d+)\s*>>>", re.IGNORECASE
        )
        parts = pattern.split(raw_text)
        # ``parts`` alternates: [pre-first-delim, page_num_str, page_text, ...]
        pages: list[dict[str, Any]] = []

        if len(parts) == 1:
            # No delimiter found — treat everything as page 1.
            text = parts[0].strip()
            if text:
                pages.append({"page_number": 1, "text": text})
            return pages

        # Discard anything before the first delimiter (usually empty / BOM).
        i = 1  # start at the first captured page number
        while i < len(parts) - 1:
            page_num_str = parts[i]
            page_text = parts[i + 1].strip()
            try:
                page_number = int(page_num_str)
            except ValueError:
                page_number = len(pages) + 1
            pages.append({"page_number": page_number, "text": page_text})
            i += 2

        # Sort ascending in case the model emitted pages out of order.
        pages.sort(key=lambda p: p["page_number"])
        return pages

    @staticmethod
    def _pdf_to_base64_images(file_bytes: bytes) -> list[tuple[str, str]]:
        """Convert PDF bytes to a list of ``(base64_data, mime_type)`` tuples.

        Each tuple represents one page rendered as a PNG image. Uses
        ``pdfplumber`` for page geometry and ``Pillow`` for rendering.
        Pages are rendered at 150 DPI and capped at MAX_IMAGE_DIM to
        prevent exceeding LLM context windows.

        Returns:
            List of ``(base64_encoded_png, "image/png")`` for each page.

        Raises:
            AdapterError: When ``pdfplumber`` or ``Pillow`` is not available,
                or the PDF cannot be parsed.
        """
        try:
            import pdfplumber  # type: ignore[import]
            from PIL import Image  # type: ignore[import]
        except ImportError as exc:
            raise AdapterError(
                "PDF processing requires 'pdfplumber' and 'Pillow'. "
                "Install them with: pip install pdfplumber Pillow"
            ) from exc

        images: list[tuple[str, str]] = []
        try:
            with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    # Render page to a PIL Image at 150 DPI.
                    page_image = page.to_image(resolution=QwenVLDefaults.PDF_RENDER_DPI)
                    pil_img: Image.Image = page_image.original

                    # Convert to RGB if needed (e.g. RGBA, P mode).
                    if pil_img.mode not in ("RGB", "L"):
                        pil_img = pil_img.convert("RGB")

                    # Cap maximum dimension to avoid excessive vision tokens.
                    max_dim = QwenVLDefaults.MAX_IMAGE_DIM
                    w, h = pil_img.size
                    if max(w, h) > max_dim:
                        if w > h:
                            new_w = max_dim
                            new_h = max(1, int(h * (max_dim / w)))
                        else:
                            new_h = max_dim
                            new_w = max(1, int(w * (max_dim / h)))
                        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

                    buf = BytesIO()
                    pil_img.save(buf, format="PNG")
                    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                    images.append((b64, "image/png"))
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError(
                f"Failed to convert PDF to images: {exc}"
            ) from exc

        if not images:
            raise AdapterError("PDF appears to have zero pages.")
        return images

    @staticmethod
    def _image_to_base64(file_bytes: bytes, mime_type: str) -> str:
        """Encode raw image bytes as a base64 string, downscaling if oversized."""
        try:
            from PIL import Image  # type: ignore[import]

            img = Image.open(BytesIO(file_bytes))
            max_dim = QwenVLDefaults.MAX_IMAGE_DIM
            w, h = img.size
            if max(w, h) > max_dim:
                if w > h:
                    new_w = max_dim
                    new_h = max(1, int(h * (max_dim / w)))
                else:
                    new_h = max_dim
                    new_w = max(1, int(w * (max_dim / h)))
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                buf = BytesIO()
                save_fmt = "JPEG" if mime_type in (QwenVLMimeTypes.JPEG, "image/jpeg") else "PNG"
                img.save(buf, format=save_fmt)
                return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            pass
        return base64.b64encode(file_bytes).decode("ascii")

    @staticmethod
    def _write_metadata(
        output_file_path: str,
        metadata: dict[str, Any],
        fs: FileStorage,
    ) -> None:
        """Write a JSON sidecar metadata file alongside *output_file_path*.

        Follows the same layout as LLMWhisperer: ``{extract_dir}/metadata/
        {stem}.json``.  Errors are only logged — a metadata write failure
        must never abort the extraction.
        """
        try:
            out_path = Path(output_file_path)
            metadata_dir = out_path.parent / "metadata"
            metadata_path = metadata_dir / out_path.with_suffix(".json").name
            fs.mkdir(create_parents=True, path=str(metadata_dir))
            fs.write(
                path=str(metadata_path),
                mode="w",
                encoding="utf-8",
                data=json.dumps(metadata, ensure_ascii=False, indent=4),
            )
        except Exception as exc:
            logger.warning("QwenVL: could not write metadata sidecar: %s", exc)

    def _call_model_for_page(
        self,
        client: Any,
        b64_data: str,
        img_mime: str,
        page_num: int,
        total_pages: int,
        model_name: str,
        system_prompt: str,
        max_tokens: int,
        base_url: str,
    ) -> tuple[str, dict[str, Any]]:
        """Submit a single page image to the model and return its extracted text."""
        content_parts = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{img_mime};base64,{b64_data}",
                },
            },
            {
                "type": "text",
                "text": (
                    f"Extract all text from page {page_num} of this document, inserting "
                    f"<<< PAGE_{page_num} >>> delimiters as instructed."
                ),
            },
        ]

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content_parts},
                ],
                temperature=QwenVLDefaults.TEMPERATURE,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            exc_name = type(exc).__name__
            msg = str(exc)

            if isinstance(exc, AdapterError):
                raise

            try:
                import openai as _openai

                if isinstance(exc, _openai.RateLimitError):
                    raise AdapterError(
                        f"API rate limit exceeded on page {page_num}. Retry after a short wait. Detail: {msg}"
                    ) from exc
                if isinstance(exc, _openai.AuthenticationError):
                    raise AdapterError(
                        f"API authentication error. Check your API key. Detail: {msg}"
                    ) from exc
                if isinstance(exc, _openai.APITimeoutError):
                    raise AdapterError(
                        f"API request timed out on page {page_num}. Detail: {msg}"
                    ) from exc
                if isinstance(exc, _openai.APIConnectionError):
                    raise AdapterError(
                        f"Cannot connect to the API endpoint at '{base_url}'. Check that the server is running. Detail: {msg}"
                    ) from exc
            except AdapterError:
                raise
            except Exception:
                pass

            if "context" in msg.lower() and (
                "exceed" in msg.lower() or "too long" in msg.lower() or "maximum context" in msg.lower()
            ):
                raise AdapterError(
                    f"Page {page_num} exceeded model context window ({msg}). "
                    f"Try increasing the context window on your serving endpoint (e.g. vLLM '--max-model-len 32768' or Ollama 'num_ctx 32768'), "
                    f"or lower 'max_tokens' in profile settings."
                ) from exc
            if "429" in msg or "rate" in msg.lower():
                raise AdapterError(f"API rate limit exceeded on page {page_num}. Detail: {msg}") from exc
            if "401" in msg or "403" in msg or "auth" in msg.lower():
                raise AdapterError(f"API authentication error. Detail: {msg}") from exc
            if "timeout" in msg.lower() or "deadline" in msg.lower():
                raise AdapterError(f"API request timed out on page {page_num}. Detail: {msg}") from exc
            if "connection" in msg.lower() or "refused" in msg.lower():
                raise AdapterError(f"Cannot connect to API endpoint at '{base_url}'. Detail: {msg}") from exc
            raise AdapterError(f"API error on page {page_num} ({exc_name}): {msg}") from exc

        try:
            page_text: str = response.choices[0].message.content or ""
        except (IndexError, AttributeError):
            page_text = ""

        if not page_text:
            raise AdapterError(
                f"The model returned an empty response for page {page_num}. "
                "The page may be blank, corrupted, or unsupported."
            )

        # Ensure page delimiter is present in the output.
        if f"PAGE_{page_num}" not in page_text.upper():
            page_text = f"<<< PAGE_{page_num} >>>\n{page_text}"

        usage: dict[str, Any] = {}
        try:
            if response.usage is not None:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens or 0,
                    "completion_tokens": response.usage.completion_tokens or 0,
                    "total_tokens": response.usage.total_tokens or 0,
                }
        except Exception:
            pass

        return page_text, usage

    def _call_model(
        self,
        file_bytes: bytes,
        mime_type: str,
        model_name: str,
        system_prompt: str,
    ) -> tuple[str, dict[str, Any]]:
        """Submit document pages individually to the OpenAI-compatible API.

        Processes multi-page documents page-by-page so that each page image
        comfortably fits inside the model's available context window (e.g. 8192).

        Args:
            file_bytes: Raw document/image bytes.
            mime_type: MIME type string for the input.
            model_name: Model identifier on the serving endpoint.
            system_prompt: Full system/instruction prompt text.

        Returns:
            A tuple of ``(extracted_text, usage_metadata_dict)``.

        Raises:
            AdapterError: On any API, network, or response-parsing failure.
        """
        # Import lazily so the SDK does not hard-depend on openai unless
        # this adapter is actually used.
        try:
            import openai  # type: ignore[import]
        except ImportError as exc:
            raise AdapterError(
                "The 'openai' package is required for the Qwen VL adapter. "
                "Install it with: pip install openai>=1.0.0"
            ) from exc

        base_url = self._resolve_base_url()
        api_key = self._resolve_api_key()
        max_tokens = self._resolve_max_tokens()

        client = openai.OpenAI(base_url=base_url, api_key=api_key)

        if mime_type == QwenVLMimeTypes.PDF:
            page_images = self._pdf_to_base64_images(file_bytes)
        else:
            b64_data = self._image_to_base64(file_bytes, mime_type)
            page_images = [(b64_data, mime_type)]

        page_texts: list[str] = []
        total_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        for idx, (b64_data, img_mime) in enumerate(page_images, 1):
            page_text, usage = self._call_model_for_page(
                client=client,
                b64_data=b64_data,
                img_mime=img_mime,
                page_num=idx,
                total_pages=len(page_images),
                model_name=model_name,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                base_url=base_url,
            )
            page_texts.append(page_text)
            for k in total_usage:
                total_usage[k] += usage.get(k, 0)

        raw_text = "\n\n".join(page_texts)
        return raw_text, total_usage

    # ------------------------------------------------------------------
    # Public interface (X2TextAdapter contract)
    # ------------------------------------------------------------------

    def test_connection(self) -> bool:
        """Verify that the configured endpoint is reachable.

        Sends a minimal text-only chat completion to confirm the endpoint and
        model are operational, and returns ``True`` on success or raises
        ``AdapterError``.
        """
        try:
            import openai  # type: ignore[import]
        except ImportError as exc:
            raise AdapterError(
                "The 'openai' package is not installed.  "
                "Run: pip install openai>=1.0.0"
            ) from exc

        base_url = self._resolve_base_url()
        api_key = self._resolve_api_key()
        model_name = self._resolve_model_name()

        try:
            client = openai.OpenAI(base_url=base_url, api_key=api_key)

            # Minimal chat completion as a connectivity/auth check.
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
            )
            # Verify we got a valid response.
            if not response.choices:
                raise AdapterError(
                    "Endpoint returned an empty response.  "
                    "Check that the model is loaded and serving."
                )
            logger.info(
                "QwenVL: test_connection succeeded for model '%s' at '%s'.",
                model_name,
                base_url,
            )
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError(
                f"QwenVL test_connection failed: {exc}"
            ) from exc
        return True

    def process(
        self,
        input_file_path: str,
        output_file_path: str | None = None,
        fs: FileStorage | None = None,
        **kwargs: dict[Any, Any],
    ) -> TextExtractionResult:
        """Extract text from *input_file_path* using Qwen 2.5 VL.

        Args:
            input_file_path: Path (local or FileStorage-relative) of the
                document to extract.
            output_file_path: Optional path where the extracted text should be
                written.  When provided a JSON sidecar with token usage is also
                written beside it (same directory, ``metadata/`` subfolder).
            fs: FileStorage instance.  Defaults to local disk.
            **kwargs: Unused; accepted for interface compatibility.

        Returns:
            ``TextExtractionResult`` with ``extracted_text`` and ``page_count``.

        Raises:
            AdapterError: On any configuration, API, or I/O error.
        """
        if fs is None:
            fs = FileStorage(provider=FileStorageProvider.LOCAL)

        model_name = self._resolve_model_name()
        system_prompt = self._build_system_prompt()

        # 1. Read the input file.
        try:
            file_bytes: bytes = bytes(fs.read(path=input_file_path, mode="rb"))
        except OSError as exc:
            logger.error("QwenVL: failed to read '%s': %s", input_file_path, exc)
            raise AdapterError(str(exc)) from exc

        mime_type = self._detect_mime_type(input_file_path, fs)
        logger.info(
            "QwenVL: extracting '%s' (mime=%s, model=%s, %d bytes)",
            input_file_path,
            mime_type,
            model_name,
            len(file_bytes),
        )

        # 2. Call the model via OpenAI-compatible API.
        raw_text, usage_metadata = self._call_model(
            file_bytes=file_bytes,
            mime_type=mime_type,
            model_name=model_name,
            system_prompt=system_prompt,
        )

        # 3. Split into pages.
        pages = self._split_pages(raw_text)
        page_count = len(pages) if pages else None

        logger.info(
            "QwenVL: extraction complete — %d page(s), %d chars (model=%s)",
            page_count or 0,
            len(raw_text),
            model_name,
        )

        # 4. Optionally write extracted text to file and emit metadata sidecar.
        if output_file_path:
            try:
                fs.write(
                    path=output_file_path,
                    mode="w",
                    encoding="utf-8",
                    data=raw_text,
                )
            except Exception as exc:
                logger.error(
                    "QwenVL: failed to write output to '%s': %s",
                    output_file_path,
                    exc,
                )
                raise AdapterError(str(exc)) from exc

            metadata: dict[str, Any] = {
                "engine": "qwen_vl",
                "model": model_name,
                "page_count": page_count,
                "pages": pages,
                **usage_metadata,
            }
            self._write_metadata(output_file_path, metadata, fs)

        return TextExtractionResult(
            extracted_text=raw_text,
            extraction_metadata=None,
            page_count=page_count,
        )

