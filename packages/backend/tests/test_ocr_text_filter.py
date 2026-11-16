from app.services.ingest import usable_ocr_text


def test_readable_text_is_kept():
    text = "REPORT ON\nTREND AND PROGRESS OF\nBANKING IN INDIA\n\n2024-25 the banks lent more money"

    assert usable_ocr_text(text) == text


def test_garbled_text_is_dropped():
    text = "2 bi RE hf eo PSR RRR FRPP PRL PCR\noi ttt ee ati ee NST ere Lee reer.\ni | tins i mz EC ie"

    assert usable_ocr_text(text) == ""


def test_empty_text_is_dropped():
    assert usable_ocr_text("") == ""
    assert usable_ocr_text("   \n  ") == ""
