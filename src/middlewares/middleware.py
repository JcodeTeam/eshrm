from fastapi.middleware.cors import CORSMiddleware

def cors(app):
    origins = [
        "https://eshrm.jcode.my.id", 
        "https://eshrm-backend.jcode.my.id", 
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
