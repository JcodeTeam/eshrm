from fastapi import FastAPI
from src.middlewares.middleware import cors
from src.routes import face_routes
from src.routes import user_routes


app = FastAPI(
    title="API Absensi Wajah",
    description="API untuk registrasi, training, dan pengenalan wajah.",
    version="1.0.0"
)

cors(app)

app.include_router(face_routes.router)
app.include_router(user_routes.router)

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Selamat datang di API Absensi Wajah. Kunjungi /docs untuk dokumentasi."}