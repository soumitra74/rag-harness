import pypdfium2 as pdfium
import pytesseract


def ocr_page(pdf_path: str, page_number: int) -> str:
    """OCR a single 1-based page of a PDF that has no text layer.

    The page is rendered with pypdfium2 at roughly 300 DPI and passed to
    Tesseract (English). Returns the recognized text as a UTF-8 string.
    """
    if not isinstance(page_number, int) or isinstance(page_number, bool):
        raise TypeError("page_number must be an integer")

    pdf = pdfium.PdfDocument(pdf_path)
    try:
        page_count = len(pdf)
        if page_number < 1 or page_number > page_count:
            raise ValueError(
                f"page_number {page_number} out of range (document has {page_count} pages)"
            )

        page = pdf[page_number - 1]
        try:
            # 300 DPI: PDF user space is 72 units per inch.
            bitmap = page.render(scale=300 / 72)
            try:
                image = bitmap.to_pil().convert("L")
            finally:
                bitmap.close()

            text = pytesseract.image_to_string(image, lang="eng")
        finally:
            page.close()
    finally:
        pdf.close()

    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    return text.strip()
