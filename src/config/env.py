from dotenv import load_dotenv
import os

# Load optional env file for the current node environment (e.g. .env.development.local)
node_env = os.getenv("ENV", "development")
env_file = f".env.{node_env}.local"
load_dotenv(env_file)
# also attempt to load a generic .env fallback (harmless if missing)
load_dotenv()

# Required secret for signing/verifying JWTs. Fail fast with a clear message when missing
JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    # Raise on import so the app fails to start with a clear error instead of confusing jwt internals
    raise RuntimeError(
        "Missing required environment variable JWT_SECRET.\n"
        "Set JWT_SECRET in your environment (e.g. Render dashboard or .env file)."
    )

# Use a safe default for ALGORITHM if the environment variable is not set
ALGORITHM = os.getenv("ALGORITHM", "HS256")

FRONTEND_URL = os.getenv("FRONTEND_URL")
BACKEND_URL = os.getenv("BACKEND_URL")
from dotenv import load_dotenv
import os

node_env = os.getenv("ENV", "development")
env_file = f".env.{node_env}.local"
load_dotenv(env_file)




JWT_SECRET = os.getenv("JWT_SECRET")
ALGORITHM = os.getenv("ALGORITHM") 

FRONTEND_URL = os.getenv("FRONTEND_URL")
BACKEND_URL = os.getenv("BACKEND_URL")