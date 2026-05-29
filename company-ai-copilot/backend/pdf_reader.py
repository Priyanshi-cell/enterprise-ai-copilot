import logging
from io import BytesIO

from pypdf import PdfReader

logger = logging.getLogger(__name__)


def extract_pdf_text(pdf_bytes: bytes) -> dict:
    """
    Extract all text from a PDF file given as raw bytes.

    Args:
        pdf_bytes: The raw bytes of the uploaded PDF file.

    Returns:
        {
            "text": str,   # all extracted text joined together
            "pages": int,  # total number of pages in the PDF
        }
    """
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        total_pages = len(reader.pages)
        logger.info("Reading PDF | pages=%d", total_pages)

        page_texts = []

        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    page_texts.append(page_text.strip())
            except Exception as e:
                # One bad page should not stop the whole extraction
                logger.warning("Could not extract text from page %d: %s", i + 1, str(e))
                continue

        combined_text = "\n\n".join(page_texts)

        logger.info(
            "PDF extraction done | pages=%d | characters=%d",
            total_pages, len(combined_text),
        )

        return {
            "text": combined_text,
            "pages": total_pages,
        }

    except Exception as e:
        logger.error("PDF extraction failed: %s", str(e))
        return {
            "text": "",
            "pages": 0,
        }