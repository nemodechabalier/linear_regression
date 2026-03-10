

def estimate_price(mileage, theta0, theta1):
    """
    Estimate the price of a car based on its mileage.

    Uses the linear regression formula: price = theta0 + (theta1 * mileage)
    where theta0 is the intercept and theta1 is the slope.

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

    Interpretation:
        - R² close to 1: Excellent model fit
        - R² close to 0: Poor model fit
        - R² can be negative if model performs worse than mean prediction

    Args:
        mileages (list): List of car mileages.
        prices (list): List of actual car prices.
        theta0 (float): The intercept parameter of the model.
        theta1 (float): The slope parameter of the model.

    Returns:
        float: The R² score between 0 and 1 (ideally).
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
