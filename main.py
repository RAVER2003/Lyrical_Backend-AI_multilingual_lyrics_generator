from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.database import Base, engine
from api.routes import workspaces, versions, services

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Lyrical Backend")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Expand to precise frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(workspaces.router, prefix="/api/workspaces", tags=["workspaces"])
app.include_router(versions.router, prefix="/api/workspaces", tags=["versions"])
app.include_router(services.router, prefix="/api/services", tags=["services"])

@app.get("/")
def root():
    return {"message": "Lyrical API is running"}

if __name__ == "__main__":
    import uvicorn
    import subprocess
    import time
    
    print("[SYSTEM] Auto-starting Keycloak Docker Container...")
    try:
        # Check if docker is in path or try absolute path for Windows
        docker_cmd = "docker"
        try:
            subprocess.run([docker_cmd, "--version"], check=True, capture_output=True)
        except Exception:
            docker_cmd = r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"
            
        subprocess.run([docker_cmd, "start", "keycloak-lyrical"], check=False)
        print("[SYSTEM] Waiting for Keycloak to initialize...")
        
        from core.keycloak_provisioner import KeycloakProvisioner
        KeycloakProvisioner.run()
        
    except Exception as e:
        print(f"[WARNING] Keycloak startup/provisioning failed: {e}")

        
    print("[SYSTEM] Launching FastAPI Hot-Reload Server...\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

