from fastapi.middleware.cors import CORSMiddleware
from src.config.env import FRONTEND_URL, BACKEND_URL

def cors(app):

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[FRONTEND_URL, BACKEND_URL],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
