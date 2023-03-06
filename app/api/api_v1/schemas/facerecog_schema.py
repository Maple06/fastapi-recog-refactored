from fastapi import UploadFile, File, Form
from pydantic import BaseModel

class ImageAbsen(BaseModel):
    file: UploadFile = File(...)
    user_id: str = Form(...)