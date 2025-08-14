import asyncio
import os
import pickle
from PIL import Image
import numpy as np
import face_recognition
from fastapi import HTTPException

REGISTERED_FACES_DIR = "data/faces"
ENCODINGS_FILE = "trainer/face_encodings.pkl"

def get_face_encoding_from_image(image_pil: Image.Image):
    try:
        image_np = np.array(image_pil)
        face_locations = face_recognition.face_locations(image_np)

        if len(face_locations) == 0:
            print("⚠ Tidak ada wajah terdeteksi.")
            return None
        if len(face_locations) > 1:
            print("⚠ Lebih dari satu wajah terdeteksi, dilewati.")
            return None

        face_encodings = face_recognition.face_encodings(image_np, known_face_locations=face_locations)
        return face_encodings[0] if face_encodings else None
    except Exception as e:
        print(f"❌ Error saat memproses gambar: {e}")
        return None


async def train_logic():
    encodings = []
    names = []

    if not os.path.exists(REGISTERED_FACES_DIR):
        raise HTTPException(status_code=404, detail=f"Direktori data wajah '{REGISTERED_FACES_DIR}' tidak ditemukan.")

    for person_name in os.listdir(REGISTERED_FACES_DIR):
        person_folder = os.path.join(REGISTERED_FACES_DIR, person_name)
        if not os.path.isdir(person_folder):
            continue

        print(f"📂 Memproses user: {person_name}")
        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            try:
                img_pil = Image.open(img_path).convert("RGB")
                encoding = get_face_encoding_from_image(img_pil)
                if encoding is not None:
                    encodings.append(encoding)
                    names.append(person_name)
                else:
                    print(f"   ⏭ Skip: {img_name}")
            except Exception as e:
                print(f"   ❌ Gagal memproses {img_name}: {e}")

    if not encodings:
        raise HTTPException(status_code=500, detail="Training gagal. Tidak ada wajah valid.")

    os.makedirs(os.path.dirname(ENCODINGS_FILE), exist_ok=True)

    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": encodings, "names": names}, f)

    print(f"✅ Training selesai. {len(encodings)} wajah dari {len(set(names))} orang berhasil disimpan.")
    return {
        "status": "success",
        "message": f"Training selesai. {len(encodings)} wajah dari {len(set(names))} orang berhasil diproses."
    }

if __name__ == "__main__":
    print("🚀 Memulai proses training wajah...")
    result = asyncio.run(train_logic())
    print(result)