## Requirements

Python 3.10.11

# Create Virtual Env

python3.10 -m venv venv

venv/Scripts/activate (Windows)
source venv/bin/activate (Linux)

pip -r install requirements.txt

cp .env.example .env.development.local

## Run This Project

uvicorn eshrm:app --reload