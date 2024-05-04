from pydantic import BaseModel
from typing import Dict

class OCRBaseRequest(BaseModel):
    pass

class OCRBoundingRect(BaseModel):
    x: int
    y: int
    width: int
    height: int

class OCRJSONResponse(BaseModel):
    bounding_rect: OCRBoundingRect
    font: str
    confidence_threshold: float

class OCRBaseResponse(BaseModel):
    detected_fonts: OCRJSONResponse[]