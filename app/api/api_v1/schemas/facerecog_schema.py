from fastapi import UploadFile, File, Form
from pydantic import BaseModel

class Image(BaseModel):
    file: UploadFile = File(...)

class UserID(BaseModel):
    user_id: str = Form(...)