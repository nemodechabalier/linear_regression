import matplotlib.pyplot as plt
import json
import csv
import os
import numpy as np


def load_theta():
    """
    Load theta parameters from the JSON file.

    Reads the trained model parameters (theta0 and theta1) from theta.json.

    Returns:
        tuple: A tuple containing (theta0, theta1) values.

    Raises:
        NameError: If the theta.json file does not exist.
        ValueError: If the JSON file is corrupted or has invalid format.
    """
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
    """
    Estimate the price of a car based on its mileage.

    Uses the linear regression formula: price = theta0 + (theta1 * mileage).

    Args:
        mileage (float): The car's mileage in kilometers.
        theta0 (float): The intercept parameter of the model.
        theta1 (float): The slope parameter of the model.

    Returns:
        float: The estimated price of the car.
    """
    return theta0 + (theta1 * mileage)


def calculate_r_squared(mileages, prices, theta0, theta1):
    """
    Calculate the R² (coefficient of determination) score.

    R² measures how well the regression line fits the data.
    Formula: R² = 1 - (SS_res / SS_tot)

    Args:
        mileages (list): List of car mileages.
        prices (list): List of actual car prices.
        theta0 (float): The intercept parameter of the model.
        theta1 (float): The slope parameter of the model.

    Returns:
        float: The R² score (1 = perfect fit, 0 = poor fit).
    """
    mean_price = sum(prices) / len(prices)

    ss_tot = sum((price - mean_price) ** 2 for price in prices)

    ss_res = 0
    for i in range(len(mileages)):
        prediction = estimate_price(mileages[i], theta0, theta1)
        ss_res += (prices[i] - prediction) ** 2

    r_squared = 1 - (ss_res / ss_tot)

    return r_squared


def calculate_mse(mileages, prices, theta0, theta1):
    """
    Calculate the Mean Squared Error (MSE) of the model.

    MSE is the average of squared differences between predicted
    and actual values. Lower values indicate better model performance.

    Args:
        mileages (list): List of car mileages.
        prices (list): List of actual car prices.
        theta0 (float): The intercept parameter of the model.
        theta1 (float): The slope parameter of the model.

    Returns:
        float: The mean squared error value.
    """
    mse = 0
    for i in range(len(mileages)):
        prediction = estimate_price(mileages[i], theta0, theta1)
        error = prediction - prices[i]
        mse += error ** 2

    return mse / len(mileages)


def calculate_mae(mileages, prices, theta0, theta1):
    """
    Calculate the Mean Absolute Error (MAE) of the model.

    MAE is the average of absolute differences between predicted
    and actual values. It represents the average prediction error in euros.

    Args:
        mileages (list): List of car mileages.
        prices (list): List of actual car prices.
        theta0 (float): The intercept parameter of the model.
        theta1 (float): The slope parameter of the model.

    Returns:
        float: The mean absolute error value in euros.
    """
    mae = 0
    for i in range(len(mileages)):
        prediction = estimate_price(mileages[i], theta0, theta1)
        error = abs(prediction - prices[i])
        mae += error

    return mae / len(mileages)


def plot_data_and_regression(mileages, prices, theta0, theta1):
    """
    Create a visualization of the data points and regression line.

    Displays a scatter plot of the actual data points with the fitted
    regression line overlaid. Includes error lines for sample points
    and a text box showing the model equation and metrics.

    Args:
        mileages (list): List of car mileages.
        prices (list): List of actual car prices.
        theta0 (float): The intercept parameter of the model.
        theta1 (float): The slope parameter of the model.
    """
    plt.figure(figsize=(12, 7))

    plt.scatter(mileages, prices, color='blue', alpha=0.6, s=50, label='Real Data', edgecolors='black')

    min_km = min(mileages)
    max_km = max(mileages)

    km_line = np.linspace(min_km - 10000, max_km + 10000, 100)
    price_line = [estimate_price(km, theta0, theta1) for km in km_line]

    plt.plot(km_line, price_line, color='red', linewidth=2, label='Linear Regression', linestyle='--')

    sample_indices = [0, len(mileages)//4, len(mileages)//2, 3*len(mileages)//4, len(mileages)-1]
    for i in sample_indices:
        km = mileages[i]
        real_price = prices[i]
        pred_price = estimate_price(km, theta0, theta1)

        plt.plot([km, km], [real_price, pred_price], color='green', alpha=0.3, linewidth=1)

    plt.title('Car Price vs Mileage - Linear Regression', fontsize=16, fontweight='bold')
    plt.xlabel('Mileage (km)', fontsize=12)
    plt.ylabel('Price (€)', fontsize=12)

    plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    plt.legend(loc='upper right', fontsize=11)

    r_squared = calculate_r_squared(mileages, prices, theta0, theta1)
    mse = calculate_mse(mileages, prices, theta0, theta1)
    mae = calculate_mae(mileages, prices, theta0, theta1)

    textstr = f'Equation: Price = {theta0:.2f} + ({theta1:.6f} × km)\n'
    textstr += f'R² = {r_squared:.4f}\n'
    textstr += f'MSE = {mse:.2f}\n'
    textstr += f'MAE = {mae:.2f}'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.05, 0.05, textstr, transform=plt.gca().transAxes, fontsize=10, ha='left', va='bottom', bbox=props)

    plt.tight_layout()
    plt.show()


def print_precision_metrics(mileages, prices, theta0, theta1):
    """
    Display comprehensive precision metrics for the model.

    Prints R² score, MSE, RMSE, and MAE with interpretations.
    Also shows sample predictions for common mileage values.

    Args:
        mileages (list): List of car mileages.
        prices (list): List of actual car prices.
        theta0 (float): The intercept parameter of the model.
        theta1 (float): The slope parameter of the model.
    """
    print("\n" + "="*60)
    print("MODEL PRECISION METRICS")
    print("="*60)

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

    mse = calculate_mse(mileages, prices, theta0, theta1)
    print(f"\n📉 MSE (Mean Squared Error): {mse:.2f}")
    print(f"   → RMSE (Root MSE): {mse**0.5:.2f} €")

    mae = calculate_mae(mileages, prices, theta0, theta1)
    print(f"\n📏 MAE (Mean Absolute Error): {mae:.2f} €")
    print(f"   → On average, predictions differ by {mae:.2f} € from reality")

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
    Main entry point for the visualization program.

    Loads the training data and model parameters, displays precision
    metrics in the console, and creates a graphical visualization
    of the linear regression results.
    """
    try:
        print("Loading data from data.csv...")
        mileages, prices = load("data.csv")
        print(f"✓ Loaded {len(mileages)} data points\n")

        print("Loading theta parameters from theta.json...")
        theta0, theta1 = load_theta()
        print(f"✓ θ0 = {theta0:.2f}")
        print(f"✓ θ1 = {theta1:.6f}\n")

        print_precision_metrics(mileages, prices, theta0, theta1)

        print("Displaying visualization...")
        plot_data_and_regression(mileages, prices, theta0, theta1)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
