import io
import re
from typing import Optional

from PIL import Image, ImageFilter, ImageOps
import pytesseract

# If tesseract is not in PATH on Windows, uncomment and fix the path:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def clean_extracted_text(text: str) -> str:
    """
    Clean OCR output:
    - Remove non-printable junk characters
    - Normalize whitespace
    """
    if not text:
        return ""

    # Remove common Tesseract form feed char
    text = text.replace("\x0c", " ")

    # Remove non-printable / weird unicode (basic Latin kept)
    text = re.sub(r"[^\x09\x0a\x0d\x20-\x7E]", " ", text)

    # Collapse multiple spaces/newlines into single space
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Basic preprocessing to help Tesseract:
    - Convert to grayscale
    - Increase size (upscale)
    - Apply slight sharpening / contrast
    - Binarize
    """
    # Convert to grayscale
    image = image.convert("L")

    # Upscale image to help Tesseract see strokes better
    w, h = image.size
    scale_factor = 2
    image = image.resize((w * scale_factor, h * scale_factor), Image.LANCZOS)

    # Increase contrast a bit
    image = ImageOps.autocontrast(image)

    # Optional slight blur and sharpen
    # image = image.filter(ImageFilter.MedianFilter(size=3))

    # Simple binarization (threshold)
    # You can tweak 150 → 120/130/160 based on your images
    image = image.point(lambda x: 0 if x < 150 else 255, "1")

    return image


def extract_text_from_image(image_bytes: bytes, lang: str = "eng") -> Optional[str]:
    """
    Takes raw image bytes and returns cleaned extracted text using Tesseract.
    """
    try:
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes))

        # Preprocess
        image = preprocess_image(image)

        # Tesseract config:
        # --oem 3 : default LSTM engine
        # --psm 6 : assume a block of text (good general setting)
        config = "--oem 3 --psm 6"

        text = pytesseract.image_to_string(image, lang=lang, config=config)

        text = clean_extracted_text(text)

        return text if text else None
    except Exception as e:
        print(f"OCR error: {e}")
        return None
