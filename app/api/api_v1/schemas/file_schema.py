from fastapi import UploadFile, File
from pydantic import BaseModel

class Image(BaseModel):
    file: UploadFile = File(...)