from fastapi import Form
from pydantic import BaseModel

class UserID(BaseModel):
    user_id: str = Form(...)