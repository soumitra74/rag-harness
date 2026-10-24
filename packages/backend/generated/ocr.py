import pypdfium2 as pdfium
import pytesseract
from PIL import Image


def ocr_page(pdf_path: str, page_number: int) -> str:
    """Render one (1-based) PDF page with pypdfium2 and OCR it with Tesseract (English).

    Returns the recognized plain text as a UTF-8 string.
    """
    if page_number < 1:
        raise ValueError(f"page_number must be 1-based and >= 1, got {page_number}")

    pdf = pdfium.PdfDocument(pdf_path)
    page = None
    try:
        page_count = len(pdf)
        if page_number > page_count:
            raise ValueError(
                f"page_number {page_number} is out of range (document has {page_count} pages)"
            )

        page = pdf[page_number - 1]

        # Render at 300 DPI for reasonable OCR accuracy (PDF default is 72 DPI).
        scale = 300.0 / 72.0
        bitmap = page.render(scale=scale)
        image = bitmap.to_pil()

        # Normalize to RGB so Tesseract receives a consistent image mode.
        if image.mode != "RGB":
            image = image.convert("RGB")

        text = pytesseract.image_to_string(image, lang="eng")

        # Ensure the result is clean UTF-8 text.
        return text.encode("utf-8", errors="replace").decode("utf-8")
    finally:
        if page is not None:
            page.close()
        pdf.close()

