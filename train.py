# /home/nde-chab/Documents/Artificial_intelligence/Piscine-Pyhton/ex00/load_csv.py
import pandas as pd
from pandas import DataFrame


def load(path: str) -> DataFrame:
    """
    Load a CSV file and display its dimensions.

    Args:
        path: Path to the CSV file

    Returns:
        DataFrame if successful, None otherwise
    """
    try:
        dataset = pd.read_csv(path)

        print(f"Loading dataset of dimensions {dataset.shape}")

        return dataset

    except FileNotFoundError:
        print(f"Error: File '{path}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File '{path}' is empty.")
        return None
    except pd.errors.ParserError:
        print(f"Error: File '{path}' has invalid format.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    print("test1")
    data = load("data.csv")
    print(data)


if __name__ == "__main__":
    main()
