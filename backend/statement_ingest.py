from __future__ import annotations

import io
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import pdfplumber
except Exception:  # pragma: no cover
    pdfplumber = None

try:
    from rapidocr_onnxruntime import RapidOCR
except Exception:  # pragma: no cover
    RapidOCR = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


DATE_PATTERN = re.compile(
    r"(\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}|\d{2}\s+[A-Za-z]{3,9}\s+\d{4}|[A-Za-z]{3,9}\s+\d{1,2},\s*\d{4})"
)
AMOUNT_PATTERN = re.compile(r"[-+]?\$?\d[\d,]*\.?\d{0,2}")


@dataclass
class IngestMeta:
    source_type: str
    filename: str
    extracted_rows: int
    warnings: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "source_type": self.source_type,
            "filename": self.filename,
            "extracted_rows": self.extracted_rows,
            "warnings": self.warnings,
        }


class StatementIngestor:
    def __init__(self) -> None:
        self.ocr_engine = None
        if RapidOCR is not None:
            try:
                self.ocr_engine = RapidOCR()
            except Exception:
                self.ocr_engine = None

    def ingest(self, filename: str, raw_bytes: bytes) -> tuple[list[dict[str, object]], dict[str, object]]:
        lowered = filename.lower()
        warnings: list[str] = []
        if lowered.endswith(".csv"):
            rows = self._parse_csv(raw_bytes)
            meta = IngestMeta("csv", filename, len(rows), warnings)
            return rows, meta.to_dict()
        if lowered.endswith(".pdf"):
            rows = self._parse_pdf(raw_bytes, warnings)
            meta = IngestMeta("pdf", filename, len(rows), warnings)
            return rows, meta.to_dict()
        if lowered.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
            rows = self._parse_image(raw_bytes, warnings)
            meta = IngestMeta("image", filename, len(rows), warnings)
            return rows, meta.to_dict()
        raise ValueError("Unsupported file type. Upload CSV, PDF, PNG, JPG, JPEG, WEBP, or BMP.")

    def _parse_csv(self, raw_bytes: bytes) -> list[dict[str, object]]:
        text = self._decode_text(raw_bytes)
        frame = pd.read_csv(io.StringIO(text))
        return self._normalize_csv_frame(frame)

    def _parse_pdf(self, raw_bytes: bytes, warnings: list[str]) -> list[dict[str, object]]:
        if pdfplumber is None:
            raise RuntimeError("pdfplumber is not installed. Install backend requirements first.")

        candidate_rows: list[dict[str, object]] = []
        text_lines: list[str] = []
        with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
            for page in pdf.pages[:30]:
                page_tables = page.extract_tables() or []
                for table in page_tables:
                    for row in table:
                        parsed = self._parse_row_values([cell for cell in (row or []) if cell is not None])
                        if parsed is not None:
                            candidate_rows.append(parsed)
                page_text = page.extract_text() or ""
                text_lines.extend(line.strip() for line in page_text.splitlines() if line.strip())

        line_rows = self._parse_lines(text_lines)
        combined = candidate_rows + line_rows
        normalized = self._normalize_transactions(combined)
        if not normalized:
            warnings.append(
                "No structured transactions found in PDF. Try a clearer statement PDF or upload CSV."
            )
        return normalized

    def _parse_image(self, raw_bytes: bytes, warnings: list[str]) -> list[dict[str, object]]:
        if self.ocr_engine is None:
            raise RuntimeError(
                "OCR model unavailable. Install backend requirements and restart to enable image ingestion."
            )
        if Image is None:
            raise RuntimeError("Pillow is required for image ingestion.")

        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        image_array = np.array(image)
        ocr_result, _ = self.ocr_engine(image_array)
        lines: list[str] = []
        for item in ocr_result or []:
            if not item or len(item) < 2:
                continue
            text = str(item[1]).strip()
            if text:
                lines.append(text)

        rows = self._normalize_transactions(self._parse_lines(lines))
        if not rows:
            warnings.append(
                "Could not confidently parse transactions from image OCR. Try higher resolution or PDF."
            )
        return rows

    def _normalize_csv_frame(self, frame: pd.DataFrame) -> list[dict[str, object]]:
        columns = {str(column).strip().lower(): column for column in frame.columns}
        required = {"date", "description", "amount"}
        if not required.issubset(columns):
            raise ValueError("CSV must include at least date, description, amount columns.")

        merchant_column = columns.get("merchant", columns["description"])
        type_column = columns.get("type")
        normalized: list[dict[str, object]] = []
        for row in frame.to_dict(orient="records"):
            date_value = row.get(columns["date"])
            description = str(row.get(columns["description"], "")).strip()
            if not description:
                continue
            amount = self._to_float(row.get(columns["amount"]))
            if amount is None:
                continue
            try:
                normalized_date = self._normalize_date(date_value)
            except ValueError:
                continue
            tx_type = str(row.get(type_column, "debit")).strip().lower() if type_column else "debit"
            normalized.append(
                {
                    "date": normalized_date,
                    "description": description,
                    "merchant": str(row.get(merchant_column, description)).strip() or description,
                    "amount": round(abs(amount), 2),
                    "type": "credit" if tx_type == "credit" else "debit",
                }
            )
        if not normalized:
            raise ValueError("No valid rows found in CSV.")
        return normalized

    def _parse_row_values(self, raw_values: list[object]) -> dict[str, object] | None:
        values = [str(value).strip() for value in raw_values if str(value).strip()]
        if len(values) < 2:
            return None
        joined = " | ".join(values)
        date_match = DATE_PATTERN.search(joined)
        amounts = AMOUNT_PATTERN.findall(joined)
        if not date_match or not amounts:
            return None

        date_value = date_match.group(1)
        amount = self._to_float(amounts[-1])
        if amount is None:
            return None

        description = re.sub(DATE_PATTERN, " ", joined)
        description = re.sub(AMOUNT_PATTERN, " ", description)
        description = re.sub(r"\s+", " ", description).strip(" |-")
        description = description or "Statement transaction"
        tx_type = self._infer_type(description, amount)
        try:
            normalized_date = self._normalize_date(date_value)
        except ValueError:
            return None

        return {
            "date": normalized_date,
            "description": description,
            "merchant": self._guess_merchant(description),
            "amount": round(abs(amount), 2),
            "type": tx_type,
        }

    def _parse_lines(self, lines: list[str]) -> list[dict[str, object]]:
        parsed: list[dict[str, object]] = []
        for line in lines:
            if len(line) < 12:
                continue
            date_match = DATE_PATTERN.search(line)
            amounts = AMOUNT_PATTERN.findall(line)
            if not date_match or not amounts:
                continue

            amount = self._to_float(amounts[-1])
            if amount is None:
                continue

            date_value = date_match.group(1)
            try:
                normalized_date = self._normalize_date(date_value)
            except ValueError:
                continue
            description = re.sub(DATE_PATTERN, " ", line)
            description = re.sub(AMOUNT_PATTERN, " ", description)
            description = re.sub(r"\s+", " ", description).strip(" |-")
            if len(description) < 2:
                description = "Statement transaction"

            parsed.append(
                {
                    "date": normalized_date,
                    "description": description,
                    "merchant": self._guess_merchant(description),
                    "amount": round(abs(amount), 2),
                    "type": self._infer_type(description, amount),
                }
            )
        return parsed

    def _normalize_transactions(self, rows: list[dict[str, object]]) -> list[dict[str, object]]:
        deduped: list[dict[str, object]] = []
        seen: set[tuple[str, str, float]] = set()
        for row in rows:
            date_value = row.get("date")
            description = str(row.get("description", "")).strip()
            amount_raw = row.get("amount")
            if not date_value or not description:
                continue
            amount = self._to_float(amount_raw)
            if amount is None:
                continue
            try:
                normalized_date = self._normalize_date(date_value)
            except ValueError:
                continue

            normalized = {
                "date": normalized_date,
                "description": description,
                "merchant": str(row.get("merchant", self._guess_merchant(description))).strip() or description,
                "amount": round(abs(amount), 2),
                "type": "credit" if str(row.get("type", "debit")).lower() == "credit" else "debit",
            }
            key = (normalized["date"], normalized["description"].lower(), normalized["amount"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(normalized)

        deduped.sort(key=lambda item: item["date"])
        return deduped

    def _decode_text(self, raw_bytes: bytes) -> str:
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                return raw_bytes.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw_bytes.decode("utf-8", errors="ignore")

    def _to_float(self, value: object) -> float | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        text = text.replace("$", "").replace(",", "")
        text = text.replace("CR", "").replace("DR", "").strip()
        try:
            return float(text)
        except ValueError:
            return None

    def _normalize_date(self, value: object) -> str:
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            raise ValueError(f"Unable to parse date: {value}")
        return parsed.date().isoformat()

    def _guess_merchant(self, description: str) -> str:
        compact = re.sub(r"\s+", " ", description).strip()
        if not compact:
            return "Unknown merchant"
        tokens = compact.split(" ")
        return " ".join(tokens[:3]).strip()

    def _infer_type(self, description: str, amount: float) -> str:
        lowered = description.lower()
        credit_markers = [
            "salary",
            "payroll",
            "refund",
            "reversal",
            "interest",
            "cashback",
            "deposit",
            "credit",
            "income",
        ]
        if any(marker in lowered for marker in credit_markers):
            return "credit"
        if amount < 0:
            return "credit"
        return "debit"
