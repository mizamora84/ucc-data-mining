import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

def train_model(data_path):
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Archivo no encontrado en {data_path}")
        return
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        return
    
    # Verifica si la columna 'Date' existe y ajusta según tus columnas reales
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')  # Ajusta el formato de fecha
    else:
        print("Error: la columna 'Date' no está presente en el archivo CSV.")
        return

    data.dropna(inplace=True)
    data['goal_difference'] = data['FTHG'] - data['FTAG']
    data['result'] = (data['FTHG'] > data['FTAG']).astype(int)

    # Solo incluir características relacionadas con el rendimiento del partido
    features = ['FTHG', 'FTAG', 'goal_difference']
    X = data[features]
    y = data['result']

    # Dividir los datos para entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Ajustar el modelo con menos características
    model = RandomForestClassifier(n_estimators=100, max_depth=50, random_state=42)
    model.fit(X_train, y_train)

    # Evaluación con validación cruzada
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print("Cross-Validation Scores:", cv_scores)
    print("Mean Cross-Validation Score:", cv_scores.mean())

    # Predicción en el conjunto de prueba
    y_pred = model.predict(X_test)

    # Métricas de evaluación
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Classification Report:\n', classification_report(y_test, y_pred))
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

    # Guardar el modelo entrenado
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)

if __name__ == "__main__":
    train_model('data/football_matches.csv')
