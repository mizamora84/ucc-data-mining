import requests

url = "http://127.0.0.1:5000/predict_probability"  # Asegúrate de que la URL coincide con la de tu servidor Flask
data = {
    "HomeTeam": "Everton",  # Sustituye con el nombre real del equipo local
    "AwayTeam": "Man City"  # Sustituye con el nombre real del equipo visitante
}

response = requests.post(url, json=data)

if response.status_code == 200:
    print("Probabilidad de victoria:", response.json())
else:
    print("Error:", response.json())
