import os
import io
import pickle
import base64
import traceback
import numpy as np
from PIL import Image
from fastapi import UploadFile, HTTPException
from typing import List
import face_recognition
import requests

# Optional Cloudinary support for storing face images and encodings (recommended on Render)
import cloudinary
from cloudinary import uploader, api, utils

# Read cloudinary config from environment
USE_CLOUDINARY = os.getenv("USE_CLOUDINARY", "false").lower() == "true"
if USE_CLOUDINARY:
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
    api_key = os.getenv("CLOUDINARY_API_KEY")
    api_secret = os.getenv("CLOUDINARY_API_SECRET")
    cloudinary.config(
        cloud_name=cloud_name,
        api_key=api_key,
        api_secret=api_secret,
        secure=True,
    )

REGISTERED_FACES_DIR = "data/faces"
ENCODINGS_FILE = "trainer/face_encodings.pkl" 


def _cloudinary_upload_image(bytes_data: bytes, public_id: str, folder: str):
    """Upload image bytes to Cloudinary under given folder with public_id.
    Returns the uploaded resource dict.
    """
    res = uploader.upload(
        io.BytesIO(bytes_data),
        public_id=public_id,
        folder=folder,
        resource_type="image",
        overwrite=True,
    )
    return res


def _cloudinary_list_images_for_user(username: str):
    prefix = f"faces/{username}"
    # Cloudinary paginates; fetch first page (usually fine for small datasets)
    try:
        res = api.resources(type="upload", resource_type="image", prefix=prefix, max_results=500)
        return res.get("resources", [])
    except Exception as e:
        print(f"Cloudinary list error: {e}")
        return []


def _cloudinary_download_url(url: str) -> bytes:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.content


def _cloudinary_upload_encodings(pickle_bytes: bytes, public_id: str = "trainer/face_encodings"):
    # upload as raw so we can download later
    res = uploader.upload(
        io.BytesIO(pickle_bytes),
        public_id=public_id,
        resource_type="raw",
        overwrite=True,
    )
    return res


def _cloudinary_get_encodings_url(public_id: str = "trainer/face_encodings") -> str | None:
    try:
        url, _ = utils.cloudinary_url(public_id, resource_type="raw", format="pkl")
        return url
    except Exception as e:
        print(f"Cloudinary get encodings url error: {e}")
        return None

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

        if USE_CLOUDINARY:
            # upload to Cloudinary under faces/<username>
            pub_id = os.path.splitext(image_file.filename)[0]
            try:
                _cloudinary_upload_image(contents, public_id=pub_id, folder=f"faces/{username}")
            except Exception as e:
                print(f" -> Gagal upload ke Cloudinary: {e}")
                continue
        else:
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

    # Load existing encodings (from local or Cloudinary)
    if USE_CLOUDINARY:
        # try to download encodings raw file from Cloudinary
        enc_url = _cloudinary_get_encodings_url()
        if enc_url:
            try:
                enc_bytes = _cloudinary_download_url(enc_url)
                data = pickle.loads(enc_bytes)
                encodings = data.get("encodings", [])
                names = data.get("names", [])
            except Exception as e:
                print(f"Gagal memuat encodings dari Cloudinary: {e}")
    else:
        if os.path.exists(ENCODINGS_FILE):
            with open(ENCODINGS_FILE, "rb") as f:
                data = pickle.load(f)
                encodings = data["encodings"]
                names = data["names"]

    # Gather images either from local folder or Cloudinary
    if USE_CLOUDINARY:
        resources = _cloudinary_list_images_for_user(username)
        if not resources:
            raise HTTPException(status_code=404, detail=f"Tidak ada folder untuk user {username} di Cloudinary")
        for res in resources:
            url = res.get("secure_url")
            try:
                img_bytes = _cloudinary_download_url(url)
                img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                encoding = get_face_encoding_from_image(img_pil)
                if encoding is not None:
                    encodings.append(encoding)
                    names.append(username)
                else:
                    print(f"  -> Melewati gambar {res.get('public_id')} (tidak ada wajah / >1 wajah).")
            except Exception as e:
                print(f"  -> Gagal memproses {res.get('public_id')}: {e}")
    else:
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

    # Save encodings locally or upload to Cloudinary as raw file
    pickled = pickle.dumps({"encodings": encodings, "names": names})
    if USE_CLOUDINARY:
        try:
            _cloudinary_upload_encodings(pickled, public_id="trainer/face_encodings")
        except Exception as e:
            print(f"Gagal mengupload encodings ke Cloudinary: {e}")
            # fallback: save locally
            os.makedirs(os.path.dirname(ENCODINGS_FILE), exist_ok=True)
            with open(ENCODINGS_FILE, "wb") as f:
                f.write(pickled)
    else:
        os.makedirs(os.path.dirname(ENCODINGS_FILE), exist_ok=True)
        with open(ENCODINGS_FILE, "wb") as f:
            f.write(pickled)

    return {"status": "success", "message": "Training selesai"}


async def verify_logic(image_base64: str, user_payload: dict):
    # Load encodings either from Cloudinary (raw) or local file
    data = None
    if USE_CLOUDINARY:
        enc_url = _cloudinary_get_encodings_url()
        if not enc_url:
            raise HTTPException(status_code=400, detail="Model belum dilatih di Cloudinary. Jalankan /train.")
        try:
            enc_bytes = _cloudinary_download_url(enc_url)
            data = pickle.loads(enc_bytes)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gagal memuat model dari Cloudinary: {e}")
    else:
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

    # Wrap the rest in try/except to capture unexpected errors and print a traceback
    # This will help debugging why we previously got 500 without details.
    try:
        if "," in image_base64:
            image_base64 = image_base64.split(',')[1]
        img_data = base64.b64decode(image_base64)
        img_pil = Image.open(io.BytesIO(img_data)).convert("RGB")

        input_encoding = get_face_encoding_from_image(img_pil)
        if input_encoding is None:
            raise HTTPException(status_code=400, detail="Wajah tidak dapat dideteksi atau terdeteksi lebih dari satu wajah pada gambar input.")

        distances = face_recognition.face_distance(user_encodings, input_encoding)

        # distances may be empty or malformed; guard against NumPy errors
        if distances is None or len(distances) == 0:
            raise HTTPException(status_code=500, detail="Tidak ada encoding wajah yang dapat dibandingkan untuk user ini.")

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
    except HTTPException:
        # Reraise HTTPExceptions (client errors / expected validation)
        raise
    except Exception as e:
        # Print traceback to server logs for debugging, then return a 500 with a helpful message
        print("--- Exception in verify_logic ---")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error during face verify: {e}")
    
