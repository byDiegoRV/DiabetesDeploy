import sys
import requests

# Local por defecto. Para Render:  python client.py https://tu-app.onrender.com
base_url = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8000'

body = {
    "gender": "Male",
    "age": 62,
    "hypertension": 1,
    "heart_disease": 0,
    "smoking_history": "former",
    "bmi": 31.5,
    "HbA1c_level": 7.2,
    "blood_glucose_level": 200
}
response = requests.post(url=f'{base_url}/score', json=body, timeout=60)
print(response.status_code, response.json())
