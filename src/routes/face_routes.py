from fastapi import APIRouter, Form, File, UploadFile, Depends
from typing import List
from src.controllers import face_controller
from src.middleware.middleware import authorize

router = APIRouter(
    prefix="/api",
    tags=["Face Recognition API"]
)

@router.post("/recognize")
async def recognize_face(image_base64: str = Form(...), user_payload: dict = Depends(authorize) ):

    return await face_controller.verify_logic(image_base64, user_payload)

@router.post("/register")
async def register_face(images: List[UploadFile] = File(...), user_payload: dict = Depends(authorize)):

    return await face_controller.register_logic(images, user_payload)

@router.post("/train")
async def trigger_training(user_payload: dict = Depends(authorize)):

    return await face_controller.train_logic(user_payload)

@router.post("/verify-blink")
async def verify_with_blink(video: UploadFile = File(...), user_payload: dict = Depends(authorize)):
    return await face_controller.verify_logic_with_blink(video, user_payload)
