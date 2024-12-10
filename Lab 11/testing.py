import requests

API_KEY = "i cant just give you my API key can i now"

url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=" + API_KEY

headers = {
    "Content-Type": "application/json"
}

data = {
    "contents": [
        {
            "parts": [
                {"text": "Explain super shortly why some idiots think 5g is bad"}
            ]
        }
    ]
}

response = requests.post(url, headers=headers, json=data)

if response.status_code == 200:
    print("Response:", response.json())
else:
    print(f"{response.status_code} - {response.text}")



