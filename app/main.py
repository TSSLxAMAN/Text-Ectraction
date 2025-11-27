from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from .ocr import extract_text_from_image  # 👈 new import
from .ocr import extract_text_from_image
from .schema import (
    OCRResult,
    PDFOCRResult,
    OCRPageResult,
    ErrorResponse,
)
from fastapi.staticfiles import StaticFiles

import fitz 

app = FastAPI(
    title="Handwriting OCR API",
    description="API to extract text from handwritten images",
    version="0.1.0"
)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return {"message": "Handwriting OCR API is alive"}


@app.post("/ocr/upload")
async def upload_handwritten_image(file: UploadFile = File(...)):
    """
    Upload a handwritten image file and extract text using Tesseract OCR.
    """
    if not file.content_type.startswith("image/"):
        return JSONResponse(
            status_code=400,
            content={"detail": "Invalid file type. Please upload an image."}
        )

    # Read the file bytes
    image_bytes = await file.read()

    # Call OCR function
    extracted_text = extract_text_from_image(image_bytes)

    if extracted_text is None or extracted_text == "":
        return JSONResponse(
            status_code=500,
            content={"detail": "Failed to extract text from the image."}
        )

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "extracted_text": extracted_text
    }

@app.post(
    "/ocr/pdf",
    response_model=PDFOCRResult,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def upload_pdf_and_extract_text(file: UploadFile = File(...)):
    """
    Upload a PDF file, process each page as an image, and extract text page-wise.
    """
    if file.content_type not in ("application/pdf",) and not file.filename.lower().endswith(".pdf"):
        return JSONResponse(
            status_code=400,
            content={"detail": "Invalid file type. Please upload a PDF file."}
        )

    pdf_bytes = await file.read()

    try:
        # Open PDF from bytes
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        print(f"PDF open error: {e}")
        return JSONResponse(
            status_code=400,
            content={"detail": "Unable to read the PDF file."}
        )

    pages_results: list[OCRPageResult] = []

    if doc.page_count == 0:
        return JSONResponse(
            status_code=400,
            content={"detail": "PDF has no pages."}
        )

    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)

        # Render page to image (increase dpi if needed)
        pix = page.get_pixmap(dpi=150)

        # Convert pixmap to PNG bytes
        image_bytes = pix.tobytes("png")

        extracted_text = extract_text_from_image(image_bytes)

        if extracted_text is None:
            extracted_text = ""

        pages_results.append(
            OCRPageResult(
                page=page_index + 1,  # 1-based indexing for humans
                extracted_text=extracted_text,
            )
        )

    if not any(p.extracted_text for p in pages_results):
        return JSONResponse(
            status_code=500,
            content={"detail": "Failed to extract text from all pages."}
        )

    return PDFOCRResult(
        filename=file.filename,
        num_pages=doc.page_count,
        pages=pages_results,
    )
