import requests
import json

api_url = "https://kpatil092-poetic-verse-generator-api.hf.space/generate_verse"
headers = {
    "Content-Type": "application/json",
    "x-api-key": "gR8XuBNXCFz3WbJh0PmRxpuMeNmEAjqaVtNVjkteGM4nWicztmw6ctBzA6NiCMXHfLqQzBh4j7z7aNme1a2jQw6F1wZ5k4VyXzHG"
}
data = {
    "genre": "party",
    "rhyme_scheme": "AABB",
    "rough_lines": ["यह हिट, वह बर्फ की ठंड", "यहाँ बहुत लोग खुश हैं"]
}

try:
    resp = requests.post(api_url, json=data, headers=headers)
    print(resp.status_code)
    print(json.dumps(resp.json(), indent=2))
except Exception as e:
    print("Error:", e)
