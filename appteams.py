from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import logging

app = Flask(__name__)

# Configurar el registro de solicitudes
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Función para predecir la probabilidad de que el equipo local gane contra el equipo visitante
def predict_probability(HomeTeam, AwayTeam):
    # Generar características relevantes del partido
    match_features = generate_match_features(HomeTeam, AwayTeam)
    
    # Realizar la predicción de probabilidad
    probability = model.predict_proba([match_features])[0][1]
    return probability

# Función para generar características relevantes del partido
def generate_match_features(HomeTeam, AwayTeam):
    # Generar características del partido
    home_rank = get_team_rank(HomeTeam)
    away_rank = get_team_rank(AwayTeam)
    home_advantage = 1  # El equipo local tiene la ventaja de jugar en casa
    
    # Devolver las características como una lista
    return [home_rank, away_rank, home_advantage]

# Ejemplo de función para obtener el rango del equipo
def get_team_rank(team_name):
    # Lógica para obtener el rango del equipo desde una base de datos o API externa
    # Aquí simplemente se devuelve un valor aleatorio para fines de demostración
    return np.random.randint(1, 20)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_probability', methods=['POST'])
def predict_probability_route():
    try:
        # Obtener los datos de la solicitud
        data = request.get_json(force=True)
        
        # Validar y extraer los nombres de los equipos
        if 'HomeTeam' not in data or 'AwayTeam' not in data:
            return jsonify({'error': 'Nombres de equipos incompletos'}), 400

        HomeTeam = data['HomeTeam']
        AwayTeam = data['AwayTeam']
        
        # Realizar la predicción de probabilidad
        probability = predict_probability(HomeTeam, AwayTeam)

        return jsonify({'HomeTeam': HomeTeam, 'AwayTeam': AwayTeam, 'probability': probability})

    except Exception as e:
        app.logger.error(f"Error en la predicción de probabilidad: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
