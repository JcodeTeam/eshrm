from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routes import face_routes


app = FastAPI(
    title="API Absensi Wajah",
    description="API untuk registrasi, training, dan pengenalan wajah menggunakan LBPH.",
    version="1.0.0"
)

origins = [
    "http://localhost:5173", 
    "http://localhost:5000", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(face_routes.router)

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Selamat datang di API Absensi Wajah. Kunjungi /docs untuk dokumentasi."}