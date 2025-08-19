import os
import io
import pickle
import base64
import numpy as np
from PIL import Image
from fastapi import UploadFile, HTTPException
from typing import List
import face_recognition

REGISTERED_FACES_DIR = "data/faces"
ENCODINGS_FILE = "trainer/face_encodings.pkl" 

def get_face_encoding_from_image(image_pil: Image.Image):

    try:
        image_np = np.array(image_pil)
        face_locations = face_recognition.face_locations(image_np)

        if len(face_locations) == 0:
            print("Peringatan: Tidak ada wajah yang terdeteksi.")
            return None
        
        if len(face_locations) > 1:
            print("Peringatan: Terdeteksi lebih dari satu wajah. Gambar ini akan dilewati untuk menjaga kualitas data.")
            return None

        face_encodings = face_recognition.face_encodings(image_np, known_face_locations=face_locations)
        return face_encodings[0]

    except Exception as e:
        print(f"Error saat encoding wajah: {e}")
        return None
    

async def register_logic(images: List[UploadFile], user_payload: dict):
    username = user_payload.get("name")
    if not username:
        raise HTTPException(status_code=400, detail="Nama user tidak ditemukan di token.")

    user_dir = os.path.join(REGISTERED_FACES_DIR, username)
    os.makedirs(user_dir, exist_ok=True)

    valid_count = 0

    for image_file in images:
        contents = await image_file.read()

        try:
            img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
            encoding = get_face_encoding_from_image(img_pil)

            if encoding is None:
                print(f" -> {image_file.filename} dilewati (tidak ada wajah / lebih dari 1 wajah).")
                continue
        except Exception as e:
            print(f" -> {image_file.filename} gagal diproses: {e}")
            continue

        file_path = os.path.join(user_dir, image_file.filename)
        with open(file_path, "wb") as f:
            f.write(contents)
        valid_count += 1

    if valid_count == 0:
        raise HTTPException(status_code=400, detail="Tidak ada foto valid dengan wajah tunggal.")

    train_result = await train_logic(user_payload)

    return {
        "status": "success",
        "message": f"{valid_count} foto wajah untuk '{username}' berhasil disimpan.",
        "train_result": train_result
    }


async def train_logic(user_payload: dict):
    username = user_payload.get("name")
    if not username:
        raise HTTPException(status_code=400, detail="Nama user tidak ditemukan di token.")
    encodings = []
    names = []

    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
            encodings = data["encodings"]
            names = data["names"]

    person_folder = os.path.join(REGISTERED_FACES_DIR, username)
    if not os.path.isdir(person_folder):
        raise HTTPException(status_code=404, detail=f"Tidak ada folder untuk user {username}")

    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        try:
            img_pil = Image.open(img_path).convert("RGB")
            encoding = get_face_encoding_from_image(img_pil)

            if encoding is not None:
                encodings.append(encoding)
                names.append(username)
            else:
                print(f"  -> Melewati gambar {img_name} (tidak ada wajah / >1 wajah).")
        except Exception as e:
            print(f"  -> Gagal membuka atau memproses {img_name}: {e}")

    if not encodings:
        raise HTTPException(status_code=500, detail="Tidak ada wajah valid untuk ditambahkan.")

    os.makedirs(os.path.dirname(ENCODINGS_FILE), exist_ok=True)
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": encodings, "names": names}, f)

    return {"status": "success", "message": "Training selesai"}


async def verify_logic(image_base64: str, user_payload: dict):
    if not os.path.exists(ENCODINGS_FILE):
        raise HTTPException(status_code=400, detail="Model belum dilatih. Hapus file .pkl lama dan jalankan /train.")

    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
    
    all_encodings = np.array(data["encodings"])
    all_names = np.array(data["names"])

    user_name = user_payload.get("name")
    if not user_name:
        raise HTTPException(status_code=400, detail="Nama user tidak ditemukan di token.")
    
    user_mask = (all_names == user_name)
    if not np.any(user_mask):
        raise HTTPException(status_code=404, detail=f"Tidak ada data wajah terdaftar untuk user {user_name}")
    
    user_encodings = all_encodings[user_mask]

    try:
        if "," in image_base64:
            image_base64 = image_base64.split(',')[1]
        img_data = base64.b64decode(image_base64)
        img_pil = Image.open(io.BytesIO(img_data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal memproses gambar base64: {e}")

    input_encoding = get_face_encoding_from_image(img_pil)
    if input_encoding is None:
        raise HTTPException(status_code=400, detail="Wajah tidak dapat dideteksi atau terdeteksi lebih dari satu wajah pada gambar input.")

    distances = face_recognition.face_distance(user_encodings, input_encoding)
    min_distance = np.min(distances)

    threshold = 0.5
    if min_distance < threshold:
        return {
            "verified": True,
            "message": f"Verifikasi berhasil untuk {user_name}!",
            "distance": float(min_distance)
        }
    else:
        return {
            "verified": False,
            "message": "Wajah tidak cocok.",
            "distance": float(min_distance)
        }
    
