from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from src.config.env import JWT_SECRET, ALGORITHM


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

async def authorize(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token tidak valid atau kedaluwarsa",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        
        username: str = payload.get("name") 
        if username is None:
            raise credentials_exception
        
        return payload
    except JWTError as e:
        print(f"====== JWT DECODE ERROR ======\n{e}\n==============================") 
        raise credentials_exception