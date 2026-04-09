from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from sqlalchemy.orm import Session
from core.database import get_db
from models.user import User

security = HTTPBearer()

KEYCLOAK_URL = "http://localhost:8080/realms/lyrical-realm"
JWKS_URL = f"{KEYCLOAK_URL}/protocol/openid-connect/certs"

jwks_client = jwt.PyJWKClient(JWKS_URL)

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    token = credentials.credentials
    try:
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            options={"verify_aud": False}
        )
        user_id: str = payload.get("sub")
        email: str = payload.get("email")
        name: str = payload.get("name") or payload.get("preferred_username")
        
        if user_id is None:
            raise ValueError("No Subject in Token")
            
    except Exception as e:
        print(f"Token validation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Could not validate credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        user = User(id=user_id, email=email, name=name, password_hash="")
        db.add(user)
        db.commit()
        
    return user
