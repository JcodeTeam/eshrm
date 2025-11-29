"""Simple script to upload existing local data/faces/<username> images to Cloudinary.

Run with CLOUDINARY_* env vars set and USE_CLOUDINARY=true.
"""
import os
import sys
import cloudinary
from cloudinary import uploader

USE_CLOUDINARY = os.getenv("USE_CLOUDINARY", "false").lower() == "true"
if not USE_CLOUDINARY:
    print("Set USE_CLOUDINARY=true to use this script")
    sys.exit(1)

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True,
)

BASE = os.path.join(os.path.dirname(__file__), "..", "data", "faces")
BASE = os.path.abspath(BASE)

if not os.path.isdir(BASE):
    print("No local faces folder found at:", BASE)
    sys.exit(1)

for username in os.listdir(BASE):
    user_dir = os.path.join(BASE, username)
    if not os.path.isdir(user_dir):
        continue
    for fname in os.listdir(user_dir):
        fpath = os.path.join(user_dir, fname)
        if not os.path.isfile(fpath):
            continue
        pub = os.path.splitext(fname)[0]
        folder = f"faces/{username}"
        try:
            with open(fpath, "rb") as f:
                uploader.upload(f, public_id=pub, folder=folder, resource_type="image", overwrite=True)
            print(f"Uploaded {fpath} -> {folder}/{pub}")
        except Exception as e:
            print(f"Failed {fpath}: {e}")

print("Done")
