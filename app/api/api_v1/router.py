# API core module for all endpoints
from fastapi import APIRouter
from .endpoints.facerecog_endpoint import Recog
from .schemas.file_schema import Image
from .schemas.user_id_schema import UserID

router = APIRouter(
    prefix='/api/v1',
    responses = {
        404: {'description': 'Not Found'}
    }
)

@router.post('/')
async def faceRecog(user_id: UserID, file: Image):
    recog = Recog()
    result = recog.get_prediction(file, user_id)

    return result