import requests
import time
import logging
from core.config import settings

logger = logging.getLogger(__name__)

class KeycloakProvisioner:
    @staticmethod
    def get_admin_token():
        """Obtain an admin access token from Keycloak master realm."""
        url = f"{settings.KEYCLOAK_BASE_URL}/realms/master/protocol/openid-connect/token"
        data = {
            "client_id": "admin-cli",
            "username": settings.KEYCLOAK_ADMIN_USER,
            "password": settings.KEYCLOAK_ADMIN_PASS,
            "grant_type": "password"
        }
        
        # Wait for Keycloak to be responsive
        for i in range(10):
            try:
                resp = requests.post(url, data=data, timeout=5)
                if resp.status_code == 200:
                    return resp.json()["access_token"]
            except Exception as e:
                logger.info(f"Waiting for Keycloak API... ({i+1}/10)")
                time.sleep(3)
        return None

    @classmethod
    def run(cls):
        """Execute the provisioning sequence."""
        logger.info("Starting Keycloak auto-provisioning...")
        
        token = cls.get_admin_token()
        if not token:
            logger.error("Failed to connect to Keycloak API. Skipping provisioning.")
            return

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        # 1. Ensure Realm Exists
        realm_payload = {
            "realm": settings.KEYCLOAK_REALM,
            "enabled": True,
            "registrationAllowed": True
        }
        requests.post(f"{settings.KEYCLOAK_BASE_URL}/admin/realms", json=realm_payload, headers=headers)

        # 2. Ensure Client Exists
        client_payload = {
            "clientId": settings.KEYCLOAK_CLIENT_ID,
            "enabled": True,
            "publicClient": True,
            "redirectUris": ["http://localhost:5173/*"],
            "webOrigins": ["*"]
        }
        requests.post(f"{settings.KEYCLOAK_BASE_URL}/admin/realms/{settings.KEYCLOAK_REALM}/clients", json=client_payload, headers=headers)

        # 3. Ensure Default Test User Exists
        user_payload = {
            "username": "demo@lyrical.app",
            "email": "demo@lyrical.app",
            "firstName": "Demo",
            "lastName": "User",
            "enabled": True,
            "credentials": [{"type": "password", "value": "password123", "temporary": False}]
        }
        requests.post(f"{settings.KEYCLOAK_BASE_URL}/admin/realms/{settings.KEYCLOAK_REALM}/users", json=user_payload, headers=headers)

        logger.info("Keycloak auto-provisioning completed successfully.")
