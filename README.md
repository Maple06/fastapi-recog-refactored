# Face Recognition Website using FastAPI, OpenCV, and face-recognition library.

### Usage
- Using docker <br>
`docker compose up`
    - When container is running successfully, it will take several minutes until localhost is available and usable. Just wait until FastAPI shows "Application startup complete" in the logs.

- Native <br>
`uvicorn main:app --host 0.0.0.0 --port 3345`
    - This runs the app on localhost port 3345

Send a post request to the main directory "/" (localhost:3345) that includes 2 body requests, "file" which is an image upload/image binary string and "user_id" which is the user ID to match (string).

This API updates then re-train datasets on 01:00 a.m. local time or when a new user is detected 

### This is a semi-ready for deployment module by interns at PT Kazee Digital Indonesia for private company usage, Waktoo Product.
