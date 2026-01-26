# /home/nde-chab/Documents/Artificial_intelligence/Piscine-Pyhton/ex00/load_csv.py
import matplotlib.pyplot as plt
import json
import csv
import os
import numpy as np


def load_theta():
    """Charge les paramètres theta depuis le fichier JSON"""
    filepath = 'theta.json'

    if not os.path.exists(filepath):
        raise NameError("Path don't exist")
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
            return data['theta0'], data['theta1']
    except (json.JSONDecodeError, KeyError):
        raise ValueError("Warning: theta.json is corrupted or invalid.")


def load(path: str):
    """
    Load a CSV file and display its dimensions.

    Args:
        path: Path to the CSV file

    Returns:
        mileages, prices lists
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
    except Exception as e:
        raise ValueError(f"Error: {e}")

    return mileages, prices


def estimate_price(mileage, theta0, theta1):
    """Calcule le prix estimé pour un kilométrage donné"""
    return theta0 + (theta1 * mileage)


def calculate_r_squared(mileages, prices, theta0, theta1):
    """
    Calcule le coefficient R² (coefficient de détermination)
    R² = 1 - (SS_res / SS_tot)

    R² proche de 1 = bon modèle
    R² proche de 0 = mauvais modèle
    """
    # Moyenne des prix réels
    mean_price = sum(prices) / len(prices)

    # Somme des carrés totaux (Total Sum of Squares)
    ss_tot = sum((price - mean_price) ** 2 for price in prices)

    # Somme des carrés des résidus (Residual Sum of Squares)
    ss_res = 0
    for i in range(len(mileages)):
        prediction = estimate_price(mileages[i], theta0, theta1)
        ss_res += (prices[i] - prediction) ** 2

    # R² = 1 - (SS_res / SS_tot)
    r_squared = 1 - (ss_res / ss_tot)

    return r_squared


def calculate_mse(mileages, prices, theta0, theta1):
    """Calcule l'erreur quadratique moyenne (Mean Squared Error)"""
    mse = 0
    for i in range(len(mileages)):
        prediction = estimate_price(mileages[i], theta0, theta1)
        error = prediction - prices[i]
        mse += error ** 2

    return mse / len(mileages)


def calculate_mae(mileages, prices, theta0, theta1):
    """Calcule l'erreur absolue moyenne (Mean Absolute Error)"""
    mae = 0
    for i in range(len(mileages)):
        prediction = estimate_price(mileages[i], theta0, theta1)
        error = abs(prediction - prices[i])
        mae += error

    return mae / len(mileages)


def plot_data_and_regression(mileages, prices, theta0, theta1):
    """
    Affiche les données et la ligne de régression
    """
    plt.figure(figsize=(12, 7))

    # 1. Scatter plot des données réelles
    plt.scatter(mileages, prices, color='blue', alpha=0.6, s=50, label='Real Data', edgecolors='black')

    # 2. Ligne de régression
    # Créer des points pour tracer la ligne
    min_km = min(mileages)
    max_km = max(mileages)

    # Générer des points régulièrement espacés
    km_line = np.linspace(min_km - 10000, max_km + 10000, 100)
    price_line = [estimate_price(km, theta0, theta1) for km in km_line]

    plt.plot(km_line, price_line, color='red', linewidth=2, label='Linear Regression', linestyle='--')

    # 3. Afficher quelques prédictions
    # Prendre 5 points espacés
    sample_indices = [0, len(mileages)//4, len(mileages)//2, 3*len(mileages)//4, len(mileages)-1]
    for i in sample_indices:
        km = mileages[i]
        real_price = prices[i]
        pred_price = estimate_price(km, theta0, theta1)

        # Ligne verticale pour montrer l'erreur
        plt.plot([km, km], [real_price, pred_price], color='green', alpha=0.3, linewidth=1)

    # Titre et labels
    plt.title('Car Price vs Mileage - Linear Regression', fontsize=16, fontweight='bold')
    plt.xlabel('Mileage (km)', fontsize=12)
    plt.ylabel('Price (€)', fontsize=12)

    # Grille
    plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    # Légende
    plt.legend(loc='upper right', fontsize=11)

    # Ajouter les équations et statistiques
    r_squared = calculate_r_squared(mileages, prices, theta0, theta1)
    mse = calculate_mse(mileages, prices, theta0, theta1)
    mae = calculate_mae(mileages, prices, theta0, theta1)

    textstr = f'Equation: Price = {theta0:.2f} + ({theta1:.6f} × km)\n'
    textstr += f'R² = {r_squared:.4f}\n'
    textstr += f'MSE = {mse:.2f}\n'
    textstr += f'MAE = {mae:.2f}'

    # Boîte de texte avec les stats
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.show()


def print_precision_metrics(mileages, prices, theta0, theta1):
    """Affiche les métriques de précision du modèle"""
    print("\n" + "="*60)
    print("MODEL PRECISION METRICS")
    print("="*60)

    # R² Score
    r_squared = calculate_r_squared(mileages, prices, theta0, theta1)
    print(f"\n📊 R² Score (Coefficient of Determination): {r_squared:.4f}")
    if r_squared > 0.9:
        print("   → Excellent model! Very high precision.")
    elif r_squared > 0.7:
        print("   → Good model. Acceptable precision.")
    elif r_squared > 0.5:
        print("   → Average model. Could be improved.")
    else:
        print("   → Poor model. Needs improvement.")

    # MSE
    mse = calculate_mse(mileages, prices, theta0, theta1)
    print(f"\n📉 MSE (Mean Squared Error): {mse:.2f}")
    print(f"   → RMSE (Root MSE): {mse**0.5:.2f} €")

    # MAE
    mae = calculate_mae(mileages, prices, theta0, theta1)
    print(f"\n📏 MAE (Mean Absolute Error): {mae:.2f} €")
    print(f"   → On average, predictions differ by {mae:.2f} € from reality")

    # Prédictions sur quelques exemples
    print("\n" + "-"*60)
    print("SAMPLE PREDICTIONS")
    print("-"*60)

    test_kms = [50000, 100000, 150000, 200000]
    print(f"{'Mileage (km)':<15} {'Predicted Price (€)':<20} {'Formula'}")
    print("-"*60)
    for km in test_kms:
        price = estimate_price(km, theta0, theta1)
        print(f"{km:<15} {price:<20.2f} {theta0:.2f} + ({theta1:.6f} × {km})")

    print("="*60 + "\n")


def main():
    """
    Visualise les données, la régression linéaire et calcule la précision
    """
    try:
        # Charger les données
        print("Loading data from data.csv...")
        mileages, prices = load("data.csv")
        print(f"✓ Loaded {len(mileages)} data points\n")

        # Charger les theta
        print("Loading theta parameters from theta.json...")
        theta0, theta1 = load_theta()
        print(f"✓ θ0 = {theta0:.2f}")
        print(f"✓ θ1 = {theta1:.6f}\n")

        # Afficher les métriques de précision
        print_precision_metrics(mileages, prices, theta0, theta1)

        # Afficher le graphique
        print("Displaying visualization...")
        plot_data_and_regression(mileages, prices, theta0, theta1)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
