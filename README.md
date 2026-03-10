# Linear Regression - Car Price Prediction

A simple linear regression implementation to predict car prices based on mileage using gradient descent algorithm.

## Overview

This project implements a univariate linear regression model from scratch to estimate the price of a car given its mileage. The model uses gradient descent optimization to find the optimal parameters (theta0 and theta1) that minimize the cost function.

### Linear Regression Formula

```
price = theta0 + (theta1 × mileage)
```

Where:
- `theta0`: The intercept (base price)
- `theta1`: The slope (price change per kilometer)

## Project Structure

```
linear_regression/
├── data.csv          # Training dataset (mileage, price)
├── train.py          # Model training script
├── predict.py        # Price prediction script
├── visualize.py      # Data visualization and metrics
├── utils.py          # Shared utility functions
├── theta.json        # Saved model parameters
└── README.md         # This file
```

## Requirements

- Python 3.x
- matplotlib
- numpy
- pandas

Install dependencies:
```bash
pip install matplotlib numpy pandas
```

## Usage

### 1. Train the Model

Train the linear regression model on the provided dataset:

```bash
python train.py
```

This will:
- Load data from `data.csv`
- Normalize the data for better convergence
- Train using gradient descent (10,000 iterations by default)
- Save the learned parameters to `theta.json`
- Display training progress plots

### 2. Make Predictions

Predict the price of a car based on its mileage:

```bash
python predict.py
```

Enter a mileage value when prompted, and the program will display the estimated price.

### 3. Visualize Results

View the regression line, data points, and model metrics:

```bash
python visualize.py
```

This displays:
- Scatter plot of actual data points
- Fitted regression line
- Model metrics (R², MSE, MAE)
- Sample predictions

## Model Metrics

The model provides several metrics to evaluate its performance:

| Metric | Description |
|--------|-------------|
| **R² (Coefficient of Determination)** | Measures how well the model fits the data (0-1, higher is better) |
| **MSE (Mean Squared Error)** | Average of squared prediction errors |
| **RMSE (Root Mean Squared Error)** | Square root of MSE, in price units (€) |
| **MAE (Mean Absolute Error)** | Average absolute prediction error in € |

## Algorithm Details

### Gradient Descent

The model uses batch gradient descent with data normalization:

1. **Normalize** mileage and price data to [0, 1] range
2. **Initialize** theta0 = 0, theta1 = 0
3. **Iterate** for each training iteration:
   - Calculate predictions for all data points
   - Compute gradients
   - Update theta values: `theta = theta - learning_rate × gradient`
4. **Denormalize** theta values for real-world predictions
5. **Save** final parameters to JSON

### Hyperparameters

- **Learning Rate**: 0.1 (default)
- **Iterations**: 10,000 (default)

## Dataset

The `data.csv` file contains 24 data points with:
- `km`: Car mileage in kilometers
- `price`: Car price in euros

## Example Output

```
Training with 24 data points...
Learning rate: 0.1, Iterations: 10000
Mileage range: [22899, 240000]
Price range: [3650, 8290]

Normalized θ0 = 1.008000, θ1 = -0.987650
Denormalized θ0 = 8499.59, θ1 = -0.021412

Parameters saved to theta.json
```

## License

This project is part of an AI/Machine Learning learning exercise.
