"""OCR a single image-only PDF page using pypdfium2 for rendering and pytesseract for recognition."""

import pypdfium2 as pdfium
import pytesseract
from PIL import Image

_OCR_DPI = 300
_PDF_BASE_DPI = 72


def ocr_page(pdf_path: str, page_number: int) -> str:
    """Render the given 1-based PDF page and return its OCR'd English text as UTF-8 str.

    Raises:
        TypeError: if page_number is not an int.
        ValueError: if page_number is outside the document's page range.
    """
    if isinstance(page_number, bool) or not isinstance(page_number, int):
        raise TypeError("page_number must be an int")

    pdf = pdfium.PdfDocument(pdf_path)
    try:
        page_count = len(pdf)
        if not 1 <= page_number <= page_count:
            raise ValueError(
                f"page_number {page_number} out of range (document has {page_count} pages)"
            )

        page = pdf[page_number - 1]
        try:
            scale = _OCR_DPI / _PDF_BASE_DPI
            bitmap = page.render(scale=scale)
            try:
                image: Image.Image = bitmap.to_pil()
                if image.mode not in ("L", "RGB"):
                    image = image.convert("RGB")
                text = pytesseract.image_to_string(image, lang="eng")
            finally:
                bitmap.close()
        finally:
            page.close()
    finally:
        pdf.close()

    return text.encode("utf-8", errors="replace").decode("utf-8")
