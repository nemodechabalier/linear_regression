import json
import os


def load_theta():
    """Charge les paramètres theta depuis le fichier JSON"""
    filepath = 'theta.json'

    # Si le fichier n'existe pas, retourner valeurs par défaut
    if not os.path.exists(filepath):
        raise NameError("Path don't exist")
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
            return data['theta0'], data['theta1']
    except (json.JSONDecodeError, KeyError):
        raise ValueError("Warning: theta.json is corrupted or invalid.")


def estimate_price(mileage, theta0, theta1):
    """Calcule le prix estimé pour un kilométrage donné"""
    return theta0 + (theta1 * mileage)


def main():
    theta0, theta1 = load_theta()

    try:
        mileage = float(input("Enter a mileage: "))

        if mileage < 0:
            print("Error: mileage cannot be negative")
            return

        price = estimate_price(mileage, theta0, theta1)
        print(f"Estimated price: {price:.2f}")

    except ValueError:
        print("Error: please enter a valid number")


if __name__ == "__main__":
    main()
