import json
import os

def load_theta():
    """Charge les paramètres theta depuis le fichier JSON"""
    filepath = 'theta.json'

    # Si le fichier n'existe pas, retourner valeurs par défaut
    if not os.path.exists(filepath):
        return 0, 0

    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
            return data['theta0'], data['theta1']
    except (json.JSONDecodeError, KeyError):
        raise ValueError("Error open file")

theta0, theta1 = load_theta()
print(f"θ0 = {theta0}, θ1 = {theta1}")
