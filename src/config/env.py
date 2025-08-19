from dotenv import load_dotenv
import os

node_env = os.getenv("ENV", "development")
env_file = f".env.{node_env}.local"
load_dotenv(env_file)




JWT_SECRET = os.getenv("JWT_SECRET")
ALGORITHM = os.getenv("ALGORITHM") 

FRONTEND_URL = os.getenv("FRONTEND_URL")
BACKEND_URL = os.getenv("BACKEND_URL")