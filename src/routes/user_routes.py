from fastapi import APIRouter
from src.controllers import user_controller

router = APIRouter(
    prefix="/api",
    tags=["User Route API"]
)


@router.delete("/users/{username}")
async def delete_user(username: str):
    return await user_controller.delete_user_logic(username)