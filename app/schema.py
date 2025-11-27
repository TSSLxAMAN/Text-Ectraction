from typing import Optional, List
from pydantic import BaseModel


class OCRResult(BaseModel):
    filename: str
    content_type: str
    extracted_text: str


class OCRPageResult(BaseModel):
    page: int
    extracted_text: str


class PDFOCRResult(BaseModel):
    filename: str
    num_pages: int
    pages: List[OCRPageResult]


class ErrorResponse(BaseModel):
    detail: str
