import os
import shutil
import pickle
from fastapi import HTTPException

REGISTERED_FACES_DIR = "data/faces"
ENCODINGS_FILE = "trainer/face_encodings.pkl" 

async def delete_user_logic(username: str):
    user_dir = os.path.join(REGISTERED_FACES_DIR, username)
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)
    else:
        raise HTTPException(status_code=404, detail=f"Folder wajah untuk {username} tidak ditemukan")

    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)

        encodings = data["encodings"]
        names = data["names"]

        new_encodings = []
        new_names = []
        for enc, name in zip(encodings, names):
            if name != username:
                new_encodings.append(enc)
                new_names.append(name)

        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump({"encodings": new_encodings, "names": new_names}, f)

    return {"status": "success", "message": f"User {username} berhasil dihapus dari sistem"}


