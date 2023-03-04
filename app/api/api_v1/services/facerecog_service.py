from ....core.logging import logger
from ..load_models import math, time, cv2, face_recognition, numpy as np, dlib, requests, os, shutil

CWD = os.getcwd()

# Module specific business logic (will be use for endpoints)
class RecogService:
    def __init__(self):
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.known_face_encodings = []
        self.known_face_names = []

    def process(self, image, user_id):
        logger.info(f"API Request Received. User ID: {user_id}")
        # If user id not provided return error
        if user_id == "":
            logger.warning("No user id provided")
            return {"faceDetected": None, "confidence": None, "match-status": False, "error-status": 1, "error-message": "No user id provided"}

        # Get time now for filename
        timeNow = self.getTimeNow()
        filename = f'{CWD}/data/images/api-{timeNow}.png'

        # In case any file currently processed have the same
        # filename, to anticipate error rename the file
        filenamesInImages = os.listdir(f'{CWD}/data/images')
        count = 0
        while filename.split("/")[-1] in filenamesInImages:
            filename = f'{CWD}/data/images/api-{user_id}-{timeNow}-{count}.png'
            count += 1

        # Save the image that is sent from the request and reject if filename is not valid
        with open(filename, "wb") as f:
            if image.filename.split(".")[-1].lower() not in ["jpg", "png", "jpeg", "heif"]:
                logger.warning(f"Filename not supported. User ID: {user_id}")
                return {"faceDetected": None, "confidence": None, "match-status": False, "error-status": 1, "error-message": "Filename not supported"}
            shutil.copyfileobj(image.file, f)

        # Read image as cv2
        frame = cv2.imread(filename)

        try:
            if filename == None:
                logger.warning(f"Not a valid filename. User ID: {user_id}")
                return {"faceDetected": None, "confidence": None, "match-status": False, "error-status": 1, "error-message": "Not a valid filename"}
        except ValueError:
            pass

        frame = self.resize(filename, 480)
        frame = self.convertBGRtoRGB(frame)

        faceNames = list(self.getFaceNames(frame))

        tmpFaceNames = []
        for i in faceNames:
            IDdetected = i.split("-")[0]
            if IDdetected == "Unknown (0%)":
                IDdetected = "Unknown"
            confidence = i.split("(")[1].split(")")[0]
            if float(confidence.split("%")[0]) > 85:
                tmpFaceNames.append([IDdetected, confidence])
        faceNames = tmpFaceNames

        if len(faceNames) == 0:
            logger.info(f"API returned error: No face detected. User ID: {user_id}")
            return {"faceDetected": None, "confidence": None, "match-status": False, "error-status": 1, "error-message": "No face detected"}
        if len(faceNames) > 1:
            user_ids_detected = (i[0] for i in faceNames)
            logger.info(f"user_ids_detected: {user_ids_detected}")
            if user_id in user_ids_detected:
                logger.info(f"API returned success with exception: Found more than 1 face, but one face matched. User ID: {user_id}")
                return {"faceDetected": (i[0] for i in faceNames), "confidence": (i[1] for i in faceNames), "match-status": True, "error-status": 0, "error-message": "Found more than 1 face, but one face matched"}
            else:
                logger.info(f"API returned error: Found more than 1 face, and none of the faces matched. User ID: {user_id}")
                return {"faceDetected": None, "confidence": None, "match-status": False, "error-status": 1, "error-message": "Found more than 1 face, and none of the faces matched"}
        if faceNames[0][0] == "Unknown":
            logger.info(f"API returned error: Face detected but not in dataset. User ID: {user_id}")
            return {"faceDetected": None, "confidence": None, "match-status": False, "error-status": 1, "error-message": "Face detected but not in dataset"}

        # Delete image after process to save storage
        os.remove(filename)

        if faceNames[0][0] == user_id:
            matchStatus = True
        else:
            matchStatus = False

        # Return all result if the process succeed
        logger.info(f"API return success. User ID: {user_id}")
        return {"faceDetected": faceNames[0][0], "confidence": faceNames[0][1], "match-status": matchStatus, "error-status": 0}

    def getTimeNow(self):
        return time.strftime("%d-%b-%y.%H-%M-%S", time.gmtime())
    
    def faceConfidence(self, face_distance, face_match_threshold=0.6):
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)

        if face_distance > face_match_threshold:
            return str(round(linear_val * 100, 2)) + '%'
        else:
            value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
            return str(round(value, 2)) + '%'

    def resize(self, filename: str, resolution: int):
        frame = cv2.imread(filename)
        if frame.shape[0] >= resolution or frame.shape[1] >= resolution:
            return cv2.resize(frame, (0, 0), fx=1-(frame.shape[1]-resolution)/frame.shape[1], fy=1-(frame.shape[1]-resolution)/frame.shape[1])
        else:
            return frame

    def convertBGRtoRGB(self, frame):
        return frame[:, :, ::-1]

    def getFaceNames(self, frame):
        # Find all the faces and face encodings in the image
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            confidence = '0%'

            # Calculate the shortest distance to face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                confidence = self.faceConfidence(face_distances[best_match_index])
            
            face_names.append(f'{name} ({confidence})')

        return face_names

    def encodeFaces(self):
        # Update the dataset before encoding
        self.updateDataset()
        
        # Encoding faces (Re-training for face detection algorithm)
        logger.info("Encoding Faces... (This may take a while)")
        for image in os.listdir(f'{CWD}/data/faces'):
            face_image = face_recognition.load_image_file(f'{CWD}/data/faces/{image}')
            try:
                face_encoding = face_recognition.face_encodings(face_image)[0]
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(image)
            except IndexError:
                pass
        
        logger.info("Encoding Done!")

    def updateDataset(self):
        logger.info("Updating datasets... (This may took a while)")
        # Grab images from web waktoo open API
        r = requests.get('https://web.waktoo.com/open-api/get-selfie?token=05e41dfb64d82ff61f50ec6691ab87fb', headers={'Accept': 'application/json'})

        # Get user IDs and all images from request
        response = r.json()
        idPerusahaan = 1 # PT Kazee Digital Indonesia
        response = response["data"][idPerusahaan-1]["user"]

        # Loop through users
        for i in response:
            try:
                count = 0
                # Loop through images per user
                for j in i["foto"]:
                    url = j["foto_absen"]

                    r = requests.get(url)

                    filename = f'{CWD}/data/faces/{i["user_id"]}-pic{count}.png'

                    # Save grabbed image to {CWD}/data/faces/
                    with open(filename, 'wb') as f:
                        f.write(r.content)         
                    try :
                        # Read recently grabbed image to cv2
                        img = cv2.imread(filename)

                        # Convert into grayscale
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        
                        # Load the cascade
                        face_cascade = cv2.CascadeClassifier(f'{CWD}/ml-models/haarcascade_frontalface_default.xml')
                        
                        # Detect faces
                        faces = face_cascade.detectMultiScale(gray, 1.4, 7)
                        
                        # Crop the faces
                        for (x, y, w, h) in faces:
                            faces = img[y:y + h, x:x + w]

                        # Upscale the image so face detection is more accurate
                        sr = cv2.dnn_superres.DnnSuperResImpl_create()
                        path = f'{CWD}/ml-models/FSRCNN_x4.pb'
                        sr.readModel(path)
                        sr.setModel("fsrcnn", 4)
                        upscaled = sr.upsample(faces)

                        # Save cropped image only if any face is detected 
                        # and the amount of images saved is less than 3
                        if count <= 3 :
                            cv2.imwrite(filename, upscaled)
                            count += 1
                        else :
                            try:
                                # If images saved is already at 3, remove the file even if there is a face
                                os.remove(filename)
                                break
                            except:
                                break
                    except :
                        # Remove any images that doesn't contain a face
                        os.remove(filename)
                        break
            # Pass if user does not have any image in waktoo API
            except IndexError:
                pass

        logger.info("Datasets updated!")

recogService = RecogService()
recogService.encodeFaces()