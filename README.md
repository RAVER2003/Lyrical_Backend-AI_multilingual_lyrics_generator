# 🎵 Lyrical AI - Backend

The core API service for the Lyrical AI platform. This service handles database persistence, Keycloak auto-provisioning, and real-time streaming translations via Hugging Face.

## 🛠 Prerequisites
- **Python 3.10+**
- **Docker Desktop** (essential for the Keycloak SSO container)

## 🚀 Setup & Installation

1. **Clone and Enter**:
   ```bash
   cd Lyrical_Backend-AI_multilingual_lyrics_generator
   ```

2. **Initialize Virtual Environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **One-Time Keycloak Container Setup**:
   If you don't have the container yet, run this once:
   ```bash
   docker run -d --name keycloak-lyrical -p 8080:8080 -e KEYCLOAK_ADMIN=admin -e KEYCLOAK_ADMIN_PASSWORD=admin quay.io/keycloak/keycloak:latest start-dev
   ```

## 🏃 Running the Service
You only need one command! It will automatically start the Docker container and configure all security settings (realms, clients, and demo users) for you.

```bash
python main.py
```
- **API Documentation**: http://localhost:8000/docs
