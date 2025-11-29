from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routes import face_routes

# Inisialisasi Aplikasi
app = FastAPI(title="ESHRM Face Engine")

# --- BAGIAN INI MENGGANTIKAN IMPORT YANG ERROR ---
# Konfigurasi CORS agar Frontend (Vite) bisa akses
origins = [
    "http://localhost:5173",    # Frontend Default
    "http://127.0.0.1:5173",
    "http://localhost:5000",    # Backend Express
    "*"                         # Izinkan semua (untuk development)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -------------------------------------------------

# Masukkan Router Wajah
app.include_router(face_routes.router)

# Endpoint Test Sederhana
@app.get("/")
def read_root():
    return {"status": "Engine is running!", "docs": "/docs"}

# Opsional: Biar bisa dijalankan dengan "python eshrm.py"
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("eshrm:app", host="0.0.0.0", port=8000, reload=True)