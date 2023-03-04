from ....core.logging import logger
from ..services.facerecog_service import recogService

# Module of an endpoint
class Recog:
    def __init__(self):
        pass

    def get_prediction(self, image, user_id):
        try:
            result = recogService.process(image, user_id)
            return result

        except Exception as e:
            logger.error('Error analysing an image :', e, "User ID input:", user_id)
            return {"faceDetected": None, "confidence": None, "match-status": False, "error-status": 1, "error-message": f"Error analysing an image: {e}"}