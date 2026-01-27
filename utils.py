

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
