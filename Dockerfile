FROM continuumio/miniconda3:latest

# Install dlib from conda-forge (prebuilt, no compile) + pip packages
RUN conda install -y -c conda-forge python=3.10 dlib=19.24 && \
    conda clean -afy

WORKDIR /app
COPY . /app

# Install remaining dependencies via pip
RUN pip install --no-cache-dir \
    fastapi==0.116.1 \
    uvicorn==0.18.3 \
    numpy==1.24.4 \
    Pillow==11.3.0 \
    face-recognition==1.3.0 \
    face_recognition_models==0.3.0 \
    cloudinary==1.29.0 \
    requests==2.32.4 \
    python-dotenv==1.1.1 \
    python-jose==3.5.0 \
    passlib==1.7.4 \
    python-multipart==0.0.20

EXPOSE 8000
CMD ["uvicorn", "eshrm:app", "--host", "0.0.0.0", "--port", "8000"]
