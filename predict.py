import json
import os
from utils import estimate_price


def load_theta():
    """
    Load theta parameters from the JSON file.

    Reads the trained model parameters (theta0 and theta1) from theta.json.
    These parameters are used to make price predictions based on mileage.

    Returns:
        tuple: A tuple containing (theta0, theta1) values.

    Raises:
        NameError: If the theta.json file does not exist.
        ValueError: If the JSON file is corrupted or has invalid format.
    """
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


def main():
    """
    Main entry point for the price prediction program.

    Prompts the user to enter a car mileage and displays the estimated
    price based on the trained linear regression model. Handles input
    validation for negative values and non-numeric inputs.
    """
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
