# API core module for all endpoints
from fastapi import APIRouter
from fastapi import UploadFile, File, Form
from .endpoints.facerecog_endpoint import Recog

router = APIRouter(
    prefix='/api/v1',
    responses = {
        404: {'description': 'Not Found'}
    }
)

@router.post('/')
async def german_ner(user_id : str = Form(...), file: UploadFile = File(...)):
    recog = Recog()
    result = recog.get_prediction(file, user_id)

    return result