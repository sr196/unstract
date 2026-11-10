"""Unit tests for the Gemini Flash X2Text adapter.

Mocks all google-genai calls so no real API credentials or network
connectivity are required.  Tests verify:

- Correct output of ``_split_pages`` for delimiter-delimited and
  delimiter-free text.
- ``process()`` returns a valid ``TextExtractionResult`` with the right
  page count, text, and optional file writes.
- ``test_connection()`` propagates API errors as ``AdapterError``.
- Error cases: missing API key, rate limit, empty response, unsupported
  MIME type fallback.
"""

from __future__ import annotations

import json
import os
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from unstract.sdk1.adapters.exceptions import AdapterError
from unstract.sdk1.adapters.x2text.gemini_flash.src.gemini_flash import (
    GeminiFlashAdapter,
)
from unstract.sdk1.adapters.x2text.gemini_flash.src.constants import (
    GeminiFlashDefaults,
    GeminiFlashMimeTypes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PDF_MAGIC = b"%PDF-1.4\n"


def _make_adapter(extra_config: dict | None = None) -> GeminiFlashAdapter:
    config: dict = {"api_key": "test-api-key-abc123"}
    if extra_config:
        config.update(extra_config)
    return GeminiFlashAdapter(settings=config)


def _fake_genai_response(text: str, prompt_tokens: int = 100, out_tokens: int = 200):
    """Build a fake google.genai response namespace."""
    usage = SimpleNamespace(
        prompt_token_count=prompt_tokens,
        candidates_token_count=out_tokens,
        total_token_count=prompt_tokens + out_tokens,
    )
    return SimpleNamespace(text=text, usage_metadata=usage)


# ---------------------------------------------------------------------------
# _split_pages
# ---------------------------------------------------------------------------

class TestSplitPages(unittest.TestCase):
    """Unit tests for the static ``_split_pages`` helper."""

    def test_single_page_no_delimiter(self):
        raw = "Hello world"
        pages = GeminiFlashAdapter._split_pages(raw)
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0]["page_number"], 1)
        self.assertEqual(pages[0]["text"], "Hello world")

    def test_blank_text_returns_empty(self):
        pages = GeminiFlashAdapter._split_pages("   ")
        self.assertEqual(pages, [])

    def test_single_delimiter(self):
        raw = "<<< PAGE_1 >>>\nContent of page one."
        pages = GeminiFlashAdapter._split_pages(raw)
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0]["page_number"], 1)
        self.assertIn("Content of page one", pages[0]["text"])

    def test_multi_page(self):
        raw = (
            "<<< PAGE_1 >>>\nFirst page text.\n"
            "<<< PAGE_2 >>>\nSecond page text.\n"
            "<<< PAGE_3 >>>\nThird page text."
        )
        pages = GeminiFlashAdapter._split_pages(raw)
        self.assertEqual(len(pages), 3)
        self.assertEqual(pages[0]["page_number"], 1)
        self.assertEqual(pages[2]["page_number"], 3)
        self.assertIn("Third page", pages[2]["text"])

    def test_delimiter_case_insensitive(self):
        raw = "<<< page_1 >>>\nLower case delimiter."
        pages = GeminiFlashAdapter._split_pages(raw)
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0]["page_number"], 1)

    def test_pages_sorted_by_number(self):
        # Simulate a model that emitted page 2 before page 1 (unlikely but covered).
        raw = "<<< PAGE_2 >>>\nSecond.\n<<< PAGE_1 >>>\nFirst."
        pages = GeminiFlashAdapter._split_pages(raw)
        self.assertEqual(pages[0]["page_number"], 1)
        self.assertEqual(pages[1]["page_number"], 2)

    def test_whitespace_in_delimiter(self):
        raw = "<<<  PAGE_4  >>>\nContent."
        pages = GeminiFlashAdapter._split_pages(raw)
        self.assertEqual(pages[0]["page_number"], 4)


# ---------------------------------------------------------------------------
# MIME type detection
# ---------------------------------------------------------------------------

class TestDetectMimeType(unittest.TestCase):

    def _make_fs(self, mime_return: str = "application/pdf") -> MagicMock:
        fs = MagicMock()
        fs.mime_type.return_value = mime_return
        return fs

    def test_pdf_extension(self):
        fs = self._make_fs()
        mime = GeminiFlashAdapter._detect_mime_type("invoice.pdf", fs)
        self.assertEqual(mime, GeminiFlashMimeTypes.PDF)
        fs.mime_type.assert_not_called()  # extension match — no sniffing needed

    def test_png_extension(self):
        fs = self._make_fs()
        mime = GeminiFlashAdapter._detect_mime_type("scan.PNG", fs)
        self.assertEqual(mime, GeminiFlashMimeTypes.PNG)

    def test_jpeg_extension(self):
        fs = self._make_fs()
        mime = GeminiFlashAdapter._detect_mime_type("photo.jpeg", fs)
        self.assertEqual(mime, GeminiFlashMimeTypes.JPEG)

    def test_jpg_extension(self):
        fs = self._make_fs()
        mime = GeminiFlashAdapter._detect_mime_type("photo.jpg", fs)
        self.assertEqual(mime, GeminiFlashMimeTypes.JPEG)

    def test_unknown_extension_falls_back_to_sniff(self):
        fs = self._make_fs(mime_return="image/png")
        mime = GeminiFlashAdapter._detect_mime_type("SOURCE", fs)
        self.assertEqual(mime, "image/png")

    def test_unknown_extension_sniff_fails_returns_pdf(self):
        fs = self._make_fs()
        fs.mime_type.side_effect = Exception("not available")
        mime = GeminiFlashAdapter._detect_mime_type("SOURCE", fs)
        self.assertEqual(mime, GeminiFlashMimeTypes.PDF)


# ---------------------------------------------------------------------------
# API key resolution
# ---------------------------------------------------------------------------

class TestResolveApiKey(unittest.TestCase):

    def test_key_from_config(self):
        adapter = _make_adapter({"api_key": "my-key"})
        self.assertEqual(adapter._resolve_api_key(), "my-key")

    def test_key_from_env(self):
        adapter = GeminiFlashAdapter(settings={})
        with patch.dict(os.environ, {"GEMINI_API_KEY": "env-key"}):
            self.assertEqual(adapter._resolve_api_key(), "env-key")

    def test_missing_key_raises(self):
        adapter = GeminiFlashAdapter(settings={})
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GEMINI_API_KEY", None)
            with self.assertRaises(AdapterError):
                adapter._resolve_api_key()

    def test_empty_config_key_falls_back_to_env(self):
        adapter = GeminiFlashAdapter(settings={"api_key": ""})
        with patch.dict(os.environ, {"GEMINI_API_KEY": "fallback-key"}):
            self.assertEqual(adapter._resolve_api_key(), "fallback-key")


# ---------------------------------------------------------------------------
# process() — happy path with mocked Gemini
# ---------------------------------------------------------------------------

FAKE_RESPONSE_TEXT = (
    "<<< PAGE_1 >>>\nThis is page one content.\n"
    "<<< PAGE_2 >>>\nThis is page two content."
)


def _patch_genai(response_text: str = FAKE_RESPONSE_TEXT):
    """Return a context-manager that patches google.genai inside the adapter module."""
    fake_response = _fake_genai_response(response_text)

    mock_genai = MagicMock()
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_client.models.generate_content.return_value = fake_response

    mock_types = MagicMock()
    mock_types.GenerateContentConfig = MagicMock(return_value=MagicMock())
    mock_types.Part.from_bytes = MagicMock(return_value=MagicMock())
    mock_types.Part.from_text = MagicMock(return_value=MagicMock())
    mock_types.Content = MagicMock(return_value=MagicMock())

    return patch.dict(
        "sys.modules",
        {
            "google": MagicMock(),
            "google.genai": mock_genai,
            "google.genai.types": mock_types,
        },
    )


class TestProcess(unittest.TestCase):

    def _make_fs(self, file_bytes: bytes = PDF_MAGIC) -> MagicMock:
        fs = MagicMock()
        fs.read.return_value = file_bytes
        fs.mime_type.return_value = GeminiFlashMimeTypes.PDF
        return fs

    def test_basic_extraction_returns_result(self):
        adapter = _make_adapter()
        fs = self._make_fs()

        with _patch_genai():
            result = adapter.process(
                input_file_path="invoice.pdf",
                output_file_path=None,
                fs=fs,
            )

        self.assertIsNotNone(result)
        self.assertEqual(result.extracted_text, FAKE_RESPONSE_TEXT)
        self.assertEqual(result.page_count, 2)

    def test_writes_output_file_when_path_given(self):
        adapter = _make_adapter()
        fs = self._make_fs()

        with _patch_genai():
            adapter.process(
                input_file_path="invoice.pdf",
                output_file_path="/tmp/invoice.txt",
                fs=fs,
            )

        fs.write.assert_called()
        # First write call must be the extracted text, not the metadata sidecar.
        first_call = fs.write.call_args_list[0]
        self.assertIn("invoice.txt", str(first_call))

    def test_single_page_no_delimiter(self):
        single_page_text = "No page markers here, just raw text."
        adapter = _make_adapter()
        fs = self._make_fs()

        with _patch_genai(response_text=single_page_text):
            result = adapter.process(input_file_path="doc.pdf", fs=fs)

        self.assertEqual(result.page_count, 1)
        self.assertEqual(result.extracted_text, single_page_text)

    def test_model_name_configurable(self):
        """Confirm that a custom model_name reaches the Gemini client."""
        adapter = _make_adapter({"model_name": "gemini-1.5-pro"})
        fs = self._make_fs()

        with _patch_genai() as patches:
            import sys
            mock_genai = sys.modules["google.genai"]
            mock_client = mock_genai.Client.return_value
            adapter.process(input_file_path="file.pdf", fs=fs)
            call_args = mock_client.models.generate_content.call_args
            self.assertIn("gemini-1.5-pro", str(call_args))

    def test_read_error_raises_adapter_error(self):
        adapter = _make_adapter()
        fs = MagicMock()
        fs.read.side_effect = OSError("Permission denied")
        fs.mime_type.return_value = GeminiFlashMimeTypes.PDF

        with _patch_genai():
            with self.assertRaises(AdapterError) as ctx:
                adapter.process(input_file_path="invoice.pdf", fs=fs)
        self.assertIn("Permission denied", str(ctx.exception))


# ---------------------------------------------------------------------------
# process() — error cases
# ---------------------------------------------------------------------------

class TestProcessErrors(unittest.TestCase):

    def _make_fs(self, file_bytes: bytes = PDF_MAGIC) -> MagicMock:
        fs = MagicMock()
        fs.read.return_value = file_bytes
        fs.mime_type.return_value = GeminiFlashMimeTypes.PDF
        return fs

    def _make_failing_genai(self, exc: Exception):
        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.generate_content.side_effect = exc

        mock_types = MagicMock()
        mock_types.GenerateContentConfig = MagicMock(return_value=MagicMock())
        mock_types.Part.from_bytes = MagicMock(return_value=MagicMock())
        mock_types.Part.from_text = MagicMock(return_value=MagicMock())
        mock_types.Content = MagicMock(return_value=MagicMock())

        return patch.dict(
            "sys.modules",
            {"google": MagicMock(), "google.genai": mock_genai, "google.genai.types": mock_types},
        )

    def test_rate_limit_raises_adapter_error(self):
        adapter = _make_adapter()
        fs = self._make_fs()

        with self._make_failing_genai(Exception("429 RESOURCE_EXHAUSTED quota exceeded")):
            with self.assertRaises(AdapterError) as ctx:
                adapter.process(input_file_path="file.pdf", fs=fs)
        self.assertIn("rate limit", str(ctx.exception).lower())

    def test_auth_error_raises_adapter_error(self):
        adapter = _make_adapter()
        fs = self._make_fs()

        with self._make_failing_genai(Exception("401 API_KEY_INVALID")):
            with self.assertRaises(AdapterError) as ctx:
                adapter.process(input_file_path="file.pdf", fs=fs)
        self.assertIn("authentication", str(ctx.exception).lower())

    def test_timeout_raises_adapter_error(self):
        adapter = _make_adapter()
        fs = self._make_fs()

        with self._make_failing_genai(Exception("deadline exceeded timeout")):
            with self.assertRaises(AdapterError) as ctx:
                adapter.process(input_file_path="file.pdf", fs=fs)
        self.assertIn("timed out", str(ctx.exception).lower())

    def test_empty_response_raises_adapter_error(self):
        adapter = _make_adapter()
        fs = self._make_fs()

        fake_resp = SimpleNamespace(text="", usage_metadata=None)
        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.generate_content.return_value = fake_resp
        mock_types = MagicMock()
        mock_types.GenerateContentConfig = MagicMock(return_value=MagicMock())
        mock_types.Part.from_bytes = MagicMock(return_value=MagicMock())
        mock_types.Part.from_text = MagicMock(return_value=MagicMock())
        mock_types.Content = MagicMock(return_value=MagicMock())

        with patch.dict(
            "sys.modules",
            {"google": MagicMock(), "google.genai": mock_genai, "google.genai.types": mock_types},
        ):
            with self.assertRaises(AdapterError) as ctx:
                adapter.process(input_file_path="file.pdf", fs=fs)
        self.assertIn("empty", str(ctx.exception).lower())

    def test_missing_google_genai_package_raises(self):
        adapter = _make_adapter()
        fs = self._make_fs()

        # Remove google.genai from sys.modules to simulate ImportError.
        import sys
        saved = {k: sys.modules.pop(k) for k in list(sys.modules) if "google.genai" in k}
        with patch("builtins.__import__", side_effect=ImportError("No module named 'google.genai'")):
            with self.assertRaises((AdapterError, ImportError)):
                adapter.process(input_file_path="file.pdf", fs=fs)
        sys.modules.update(saved)


# ---------------------------------------------------------------------------
# test_connection()
# ---------------------------------------------------------------------------

class TestTestConnection(unittest.TestCase):

    def _make_genai_with_models(self, model_names: list[str] | None = None):
        if model_names is None:
            model_names = ["models/gemini-2.0-flash", "models/gemini-1.5-pro"]
        models = [SimpleNamespace(name=n) for n in model_names]

        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = iter(models)
        return mock_genai

    def test_success_returns_true(self):
        adapter = _make_adapter()
        mock_genai = self._make_genai_with_models()

        with patch.dict("sys.modules", {"google": MagicMock(), "google.genai": mock_genai}):
            self.assertTrue(adapter.test_connection())

    def test_auth_failure_raises_adapter_error(self):
        adapter = _make_adapter()
        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.side_effect = Exception("403 permission denied")

        with patch.dict("sys.modules", {"google": MagicMock(), "google.genai": mock_genai}):
            with self.assertRaises(AdapterError):
                adapter.test_connection()

    def test_empty_model_list_raises(self):
        adapter = _make_adapter()
        mock_genai = self._make_genai_with_models([])

        with patch.dict("sys.modules", {"google": MagicMock(), "google.genai": mock_genai}):
            with self.assertRaises(AdapterError):
                adapter.test_connection()


# ---------------------------------------------------------------------------
# Registry metadata check
# ---------------------------------------------------------------------------

class TestRegistryMetadata(unittest.TestCase):

    def test_metadata_has_required_keys(self):
        from unstract.sdk1.adapters.x2text.gemini_flash.src import metadata

        for key in ("name", "version", "adapter", "description", "is_active"):
            self.assertIn(key, metadata, f"Missing metadata key: {key}")

    def test_is_active(self):
        from unstract.sdk1.adapters.x2text.gemini_flash.src import metadata

        self.assertTrue(metadata["is_active"])

    def test_adapter_id_unique_format(self):
        adapter_id = GeminiFlashAdapter.get_id()
        self.assertIn("|", adapter_id, "ID should follow '{name}|{uuid}' format")
        self.assertTrue(adapter_id.startswith("geminiflash|"))

    def test_adapter_type_is_x2text(self):
        from unstract.sdk1.adapters.enums import AdapterTypes

        self.assertEqual(GeminiFlashAdapter.get_adapter_type(), AdapterTypes.X2TEXT)


if __name__ == "__main__":
    unittest.main()

