# /home/nde-chab/Documents/Artificial_intelligence/Piscine-Pyhton/ex00/load_csv.py
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import json
import csv
from utils import estimate_price, calculate_r_squared


def load(path: str):
    """
    Load a CSV file and display its dimensions.

    Args:
        path: Path to the CSV file

    Returns:
        DataFrame if successful, None otherwise
    """
    try:
        mileages = []
        prices = []

        with open(path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                mileages.append(float(row['km']))
                prices.append(float(row['price']))

    except FileNotFoundError:
        raise NameError(f"Error: File '{path}' not found.")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Error: File '{path}' is empty.")
    except pd.errors.ParserError:
        raise ValueError(f"Error: File '{path}' has invalid format.")
    except Exception as e:
        ValueError(f"Error: {e}")

    return mileages, prices

def normalize_data(data):
    """Normalise les données entre 0 et 1"""
    min_val = min(data)
    max_val = max(data)
    range_val = max_val - min_val

    normalized = [(x - min_val) / range_val for x in data]
    return normalized, min_val, max_val


def denormalize_theta(theta0, theta1, km_min, km_max, price_min, price_max):
    """Dénormalise les theta pour les utiliser avec les vraies données"""
    km_range = km_max - km_min
    price_range = price_max - price_min

    # Formules de dénormalisation
    real_theta1 = theta1 * (price_range / km_range)
    real_theta0 = theta0 * price_range + price_min - real_theta1 * km_min

    return real_theta0, real_theta1


def train(mileages, prices, learning_rate=0.1, iteration=10000):
    """doc"""
    # Normaliser les données
    norm_mileages, km_min, km_max = normalize_data(mileages)
    norm_prices, price_min, price_max = normalize_data(prices)

    theta0 = 0.0
    theta1 = 0.0
    m = len(norm_mileages)
    print(f"Training with {m} data points...")
    print(f"Learning rate: {learning_rate}, Iterations: {iteration}")
    print(f"Mileage range: [{km_min:.0f}, {km_max:.0f}]")
    print(f"Price range: [{price_min:.0f}, {price_max:.0f}]\n")
    theta0_evolution = []
    theta1_evolution = []
    r2_evolution = []
    print(f"mileages: {mileages}\nnormalized mileages: {norm_mileages}\n")
    print(f"price {prices}\nnormalized prices: {norm_prices}\n")

    for i in range(iteration):
        sum_error_theta0 = 0.0
        sum_error_theta1 = 0.0

        for j in range(m):
            estimate_price_value = estimate_price(norm_mileages[j], theta0, theta1)
            error = estimate_price_value - norm_prices[j]

            sum_error_theta0 += error
            sum_error_theta1 += error * norm_mileages[j]

        tmp_theta0 = learning_rate * (1 / m) * sum_error_theta0
        tmp_theta1 = learning_rate * (1 / m) * sum_error_theta1

        theta0 = theta0 - tmp_theta0
        theta1 = theta1 - tmp_theta1

        # Debug: afficher tous les 200 itérations
        if (i + 1) % 2000 == 0:
            print(f"Iteration {i + 1}: θ0 = {theta0:.6f}, θ1 = {theta1:.6f}")
        if (i + 1) % 100 == 0:
            real_theta0, real_theta1 = denormalize_theta(
            theta0, theta1, km_min, km_max, price_min, price_max)
            theta0_evolution.append(real_theta0)
            theta1_evolution.append(real_theta1)
            r2 = calculate_r_squared(mileages, prices, real_theta0, real_theta1)
            r2_evolution.append(r2)

    # Dénormaliser les theta pour les sauvegarder
    real_theta0, real_theta1 = denormalize_theta(
        theta0, theta1, km_min, km_max, price_min, price_max
    )

    print(f"\nNormalized θ0 = {theta0:.6f}, θ1 = {theta1:.6f}")
    print(f"Denormalized θ0 = {real_theta0:.2f}, θ1 = {real_theta1:.6f}")

    return real_theta0, real_theta1, theta0_evolution, theta1_evolution, r2_evolution

def plot_training(theta0_evolution, theta1_evolution, r2_evolution):
    iters = [i * 100 for i in range(1, len(theta0_evolution) + 1)]

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(iters, theta0_evolution)
    axes[0].set_ylabel("theta0")

    axes[1].plot(iters, theta1_evolution)
    axes[1].set_ylabel("theta1")

    axes[2].plot(iters, r2_evolution)
    axes[2].set_ylabel("R²")
    axes[2].set_xlabel("Itération")

    plt.tight_layout()
    plt.show()

def save_theta(theta0, theta1, filepath='theta.json'):
    """Sauvegarde les paramètres theta dans le fichier JSON"""
    try :
        data = {
            "theta0": theta0,
            "theta1": theta1
        }
    except Exception as e:
        raise ValueError(f"Error {e}")
    with open(filepath, 'w') as file:
        json.dump(data, file, indent=2)

    print(f"\nParameters saved to {filepath}")

def main():
    mileages, prices = load("data.csv")
    theta0, theta1, theta0_evolution, theta1_evolution, r2_evolution = train(mileages, prices)
    save_theta(theta0, theta1)
    plot_training(theta0_evolution, theta1_evolution, r2_evolution)



if __name__ == "__main__":
    main()
