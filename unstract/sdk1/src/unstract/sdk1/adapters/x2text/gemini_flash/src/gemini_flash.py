"""Gemini 2.0 Flash X2Text adapter.

Sends document bytes (PDF or images) to Google's Gemini multimodal API as an
inline data part and uses a layout-preserving extraction prompt to produce
structured text output that maps cleanly to Unstract's ``TextExtractionResult``
contract.

Supported input formats: PDF, PNG, JPEG, GIF, WebP.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from unstract.sdk1.adapters.exceptions import AdapterError
from unstract.sdk1.adapters.x2text.dto import TextExtractionResult
from unstract.sdk1.adapters.x2text.gemini_flash.src.constants import (
    EXTRACTION_SYSTEM_PROMPT,
    GeminiFlashConfig,
    GeminiFlashDefaults,
    GeminiFlashMimeTypes,
)
from unstract.sdk1.adapters.x2text.x2text_adapter import X2TextAdapter
from unstract.sdk1.file_storage import FileStorage, FileStorageProvider

logger = logging.getLogger(__name__)


class GeminiFlashAdapter(X2TextAdapter):
    """Text extractor adapter backed by Google Gemini 2.0 Flash.

    Sends the document as an inline multimodal part to the Gemini API and
    returns a ``TextExtractionResult`` that is structurally identical to what
    the LLMWhisperer and LlamaParse adapters produce, so callers require no
    special-casing.
    """

    def __init__(self, settings: dict[str, Any]) -> None:
        """Initialise the Gemini Flash text extraction adapter.

        Args:
            settings: Adapter configuration dict.  Recognised keys are
                defined in ``GeminiFlashConfig``.  ``api_key`` falls back to
                the ``GEMINI_API_KEY`` environment variable when absent or
                empty so deployments can inject credentials without storing
                them in the adapter profile.
        """
        super().__init__("GeminiFlash")
        self.config = settings

    SCHEMA_PATH = f"{os.path.dirname(__file__)}/static/json_schema.json"

    # ------------------------------------------------------------------
    # Static adapter metadata (called by the registry)
    # ------------------------------------------------------------------

    @staticmethod
    def get_id() -> str:
        return "geminiflash|6f1a2b3c-4d5e-6f7a-8b9c-0d1e2f3a4b5c"

    @staticmethod
    def get_name() -> str:
        return "Gemini 2.0 Flash Extractor"

    @staticmethod
    def get_description() -> str:
        return "Layout-preserving document extraction powered by Google Gemini 2.0 Flash"

    @staticmethod
    def get_icon() -> str:
        return "/icons/adapter-icons/GeminiFlash.png"

    @staticmethod
    def get_doc_url() -> str:
        return "https://docs.unstract.com/unstract/unstract_platform/adapters/text_extractors/gemini_flash_text_extractor/"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_api_key(self) -> str:
        """Return the effective API key, raising AdapterError if absent."""
        key = self.config.get(GeminiFlashConfig.API_KEY) or os.getenv("GEMINI_API_KEY")
        if not key:
            raise AdapterError(
                "No Google AI Studio API key configured.  Set 'api_key' in the "
                "adapter profile or export the GEMINI_API_KEY environment variable."
            )
        return key

    def _resolve_model_name(self) -> str:
        return (
            self.config.get(GeminiFlashConfig.MODEL_NAME)
            or GeminiFlashDefaults.MODEL_NAME
        )

    def _build_system_prompt(self) -> str:
        """Combine the base extraction prompt with any user-supplied additions."""
        base = EXTRACTION_SYSTEM_PROMPT
        custom = (self.config.get(GeminiFlashConfig.CUSTOM_PROMPT) or "").strip()
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
        mime = GeminiFlashMimeTypes.EXTENSION_MAP.get(ext)
        if mime:
            return mime
        try:
            mime = fs.mime_type(path=file_path)
            if mime in GeminiFlashMimeTypes.EXTENSION_MAP.values():
                return mime
        except Exception:
            pass
        logger.warning(
            "GeminiFlash: could not detect MIME type for '%s', falling back to PDF",
            file_path,
        )
        return GeminiFlashMimeTypes.PDF

    @staticmethod
    def _split_pages(raw_text: str) -> list[dict[str, Any]]:
        """Split Gemini's output on page-delimiter markers into per-page dicts.

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
        # e.g. ["", "1", "text of p1", "2", "text of p2", ...]
        pages: list[dict[str, Any]] = []

        if len(parts) == 1:
            # No delimiter found — treat everything as page 1.
            text = parts[0].strip()
            if text:
                pages.append({"page_number": 1, "text": text})
            return pages

        # Discard anything before the first delimiter (usually empty / BOM).
        # parts[0] is text before first PAGE_, which we drop.
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
            logger.warning("GeminiFlash: could not write metadata sidecar: %s", exc)

    def _call_gemini(
        self,
        file_bytes: bytes,
        mime_type: str,
        model_name: str,
        system_prompt: str,
    ) -> tuple[str, dict[str, Any]]:
        """Submit *file_bytes* to the Gemini API and return ``(text, usage)``.

        Args:
            file_bytes: Raw document bytes.
            mime_type: MIME type string for the inline data part.
            model_name: Gemini model identifier.
            system_prompt: Full system/instruction prompt text.

        Returns:
            A tuple of ``(extracted_text, usage_metadata_dict)``.

        Raises:
            AdapterError: On any API, network, or response-parsing failure.
        """
        # Import lazily so the SDK does not hard-depend on google-genai unless
        # this adapter is actually used.
        try:
            import google.genai as genai  # type: ignore[import]
            from google.genai import types as genai_types  # type: ignore[import]
        except ImportError as exc:
            raise AdapterError(
                "The 'google-genai' package is required for the Gemini Flash adapter. "
                "Install it with: pip install google-genai>=1.0.0"
            ) from exc

        api_key = self._resolve_api_key()

        try:
            client = genai.Client(api_key=api_key)

            # Build the generation config.
            generation_config = genai_types.GenerateContentConfig(
                temperature=GeminiFlashDefaults.TEMPERATURE,
                max_output_tokens=GeminiFlashDefaults.MAX_OUTPUT_TOKENS,
                system_instruction=system_prompt,
            )

            # Construct the user message: inline document bytes + text request.
            inline_data_part = genai_types.Part.from_bytes(
                data=file_bytes,
                mime_type=mime_type,
            )
            text_part = genai_types.Part.from_text(
                text=(
                    "Extract all text from this document, inserting "
                    "<<< PAGE_{n} >>> delimiters as instructed."
                )
            )
            contents = [genai_types.Content(role="user", parts=[inline_data_part, text_part])]

            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=generation_config,
            )
        except Exception as exc:
            # Map Google API exceptions → AdapterError with helpful messages.
            exc_name = type(exc).__name__
            msg = str(exc)

            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "rate" in msg.lower():
                raise AdapterError(
                    f"Gemini API rate limit exceeded. Retry after a short wait. "
                    f"Detail: {msg}"
                ) from exc
            if "401" in msg or "403" in msg or "API_KEY" in msg or "permission" in msg.lower():
                raise AdapterError(
                    f"Gemini API authentication error. Check your API key. Detail: {msg}"
                ) from exc
            if "timeout" in msg.lower() or "deadline" in msg.lower():
                raise AdapterError(
                    f"Gemini API request timed out. Detail: {msg}"
                ) from exc
            raise AdapterError(
                f"Gemini API error ({exc_name}): {msg}"
            ) from exc

        # Extract text from the response.
        try:
            extracted_text: str = response.text or ""
        except Exception:
            # ``response.text`` raises if the response has no usable candidate.
            extracted_text = ""

        if not extracted_text:
            raise AdapterError(
                "Gemini returned an empty response.  The document may be blank, "
                "corrupted, or not supported in this output configuration."
            )

        # Collect token usage metadata for the sidecar.
        usage: dict[str, Any] = {}
        try:
            um = response.usage_metadata
            if um is not None:
                usage = {
                    "prompt_token_count": getattr(um, "prompt_token_count", None),
                    "candidates_token_count": getattr(um, "candidates_token_count", None),
                    "total_token_count": getattr(um, "total_token_count", None),
                }
        except Exception:
            pass

        return extracted_text, usage

    # ------------------------------------------------------------------
    # Public interface (X2TextAdapter contract)
    # ------------------------------------------------------------------

    def test_connection(self) -> bool:
        """Verify that the configured API key is valid.

        Sends a minimal text-only ping (no document) to the models.list
        endpoint or a lightweight generate call, depending on what the SDK
        exposes, and returns ``True`` on success or raises ``AdapterError``.
        """
        try:
            import google.genai as genai  # type: ignore[import]
        except ImportError as exc:
            raise AdapterError(
                "The 'google-genai' package is not installed.  "
                "Run: pip install google-genai>=1.0.0"
            ) from exc

        api_key = self._resolve_api_key()
        model_name = self._resolve_model_name()
        try:
            client = genai.Client(api_key=api_key)
            # List models as a lightweight credential check.
            # If this fails with an auth error it surfaces before any real work.
            models = list(client.models.list())
            if not models:
                raise AdapterError("Gemini API returned an empty model list; check permissions.")
            # Verify that the configured model is reachable.
            model_ids = [getattr(m, "name", "") for m in models]
            # model names come back as "models/gemini-2.0-flash" etc.
            short_id = model_name.split("/")[-1]
            if not any(short_id in mid for mid in model_ids):
                logger.warning(
                    "GeminiFlash: model '%s' not found in account model list; "
                    "proceeding anyway (may be allowlisted separately).",
                    model_name,
                )
            logger.info("GeminiFlash: test_connection succeeded for model '%s'.", model_name)
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError(
                f"Gemini test_connection failed: {exc}"
            ) from exc
        return True

    def process(
        self,
        input_file_path: str,
        output_file_path: str | None = None,
        fs: FileStorage | None = None,
        **kwargs: dict[Any, Any],
    ) -> TextExtractionResult:
        """Extract text from *input_file_path* using Gemini 2.0 Flash.

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
            logger.error("GeminiFlash: failed to read '%s': %s", input_file_path, exc)
            raise AdapterError(str(exc)) from exc

        mime_type = self._detect_mime_type(input_file_path, fs)
        logger.info(
            "GeminiFlash: extracting '%s' (mime=%s, model=%s, %d bytes)",
            input_file_path,
            mime_type,
            model_name,
            len(file_bytes),
        )

        # 2. Call the Gemini API.
        raw_text, usage_metadata = self._call_gemini(
            file_bytes=file_bytes,
            mime_type=mime_type,
            model_name=model_name,
            system_prompt=system_prompt,
        )

        # 3. Split into pages.
        pages = self._split_pages(raw_text)
        page_count = len(pages) if pages else None

        logger.info(
            "GeminiFlash: extraction complete — %d page(s), %d chars (model=%s)",
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
                    "GeminiFlash: failed to write output to '%s': %s",
                    output_file_path,
                    exc,
                )
                raise AdapterError(str(exc)) from exc

            metadata: dict[str, Any] = {
                "engine": "gemini_flash",
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

