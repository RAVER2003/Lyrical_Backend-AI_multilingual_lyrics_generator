import os
from pydantic import BaseModel

class Settings(BaseModel):
    PROJECT_NAME: str = "Lyrical Backend"
    SQLALCHEMY_DATABASE_URI: str = "sqlite:///./lyrical.db"
    JWT_SECRET: str = os.getenv("JWT_SECRET", "supersecretjwtkeythatyoushouldchange")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7 # 7 days
    HF_API_KEY: str = "gR8XuBNXCFz3WbJh0PmRxpuMeNmEAjqaVtNVjkteGM4nWicztmw6ctBzA6NiCMXHfLqQzBh4j7z7aNme1a2jQw6F1wZ5k4VyXzHG"
    HF_TRANSLATE_URL: str = "https://kpatil092-indictrans2-fastapi.hf.space/translate"
    
    KEYCLOAK_ADMIN_USER: str = "admin"
    KEYCLOAK_ADMIN_PASS: str = "admin"
    KEYCLOAK_CLIENT_ID: str = "lyrical-frontend"
    KEYCLOAK_REALM: str = "lyrical-realm"
    KEYCLOAK_BASE_URL: str = "http://localhost:8080"

settings = Settings()
