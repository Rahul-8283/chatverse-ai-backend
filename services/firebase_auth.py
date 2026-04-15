"""
Initializes and configures the Firebase Admin SDK.
Provides a function to verify user authentication tokens.
"""
import firebase_admin
from firebase_admin import credentials, auth
from fastapi import HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer

from .config import FIREBASE_CREDENTIALS

# --- Firebase Admin SDK Initialization ---
try:
    cred = credentials.Certificate(FIREBASE_CREDENTIALS)
    firebase_admin.initialize_app(cred)
    print("Firebase Admin SDK initialized successfully.")
except Exception as e:
    print(f"Error initializing Firebase Admin SDK: {e}")
    # In a real app, you might want to handle this more gracefully
    # For now, we'll let it raise an exception if credentials are bad
    raise

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_firebase_token(token: str = Depends(oauth2_scheme)):
    """
    Dependency function to verify Firebase ID token from the Authorization header.
    
    Usage:
    @app.post("/secure-endpoint", dependencies=[Depends(verify_firebase_token)])
    async def secure_endpoint(request: Request):
        user_id = request.state.user_id
        ...
    """
    try:
        # The token comes in as "Bearer <token>", so we split it.
        id_token = token.split(" ")[1] if " " in token else token
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token['uid']
    except auth.ExpiredIdTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired. Please refresh.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except auth.InvalidIdTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token. Please re-authenticate.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during token verification: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )
