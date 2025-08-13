import pandas as pd
import yaml
from pathlib import Path


def extract_columns_to_yaml(input_csv: str, output_yaml: str, columns: list, turbulence_intensity: float):
    """
    Load a CSV file, extract specific columns, and save them to a YAML file in the desired format.

    Args:
        input_csv (str): Path to the input CSV file.
        output_yaml (str): Path to the output YAML file.
        columns (list): List of column names to extract.
        turbulence_intensity (float): Constant turbulence intensity to include in the output.

    Returns:
        None
    """
    # Load the CSV file
    try:
        data = pd.read_csv(input_csv, header=2)
    except FileNotFoundError:
        print(f"Error: The file '{input_csv}' was not found.")
        return
    except Exception as e:
        print(f"Error: An error occurred while reading the file: {e}")
        return

    # Check if all desired columns exist in the file
    missing_columns = [col for col in columns if col not in data.columns]
    if missing_columns:
        print(f"Error: The following columns are missing in the input file: {missing_columns}")
        return

    # Extract the desired columns
    extracted_data = data[columns]

    # Prepare the YAML data structure
    yaml_data = {
        "wind_speeds": extracted_data[columns[1]].tolist(),
        "wind_directions": extracted_data[columns[2]].tolist(),
        "wind_turbulence_intensities": [turbulence_intensity] * len(extracted_data),
    }

    # Save the data to a YAML file
    try:
        with open(output_yaml, "w") as yaml_file:
            yaml.dump(yaml_data, yaml_file, default_flow_style=False)
        print(f"Extracted data saved to '{output_yaml}'")
    except Exception as e:
        print(f"Error: An error occurred while saving the file: {e}")


# Example usage
if __name__ == "__main__":
    input_file = "open-meteo-56.20N8.54E86m.csv"  # Replace with your input file path
    output_file = "open-meteo-56.20N8.54E86m.yaml"  # Replace with your desired output file path
    desired_columns = ["time", "wind_speed_100m (m/s)", "wind_direction_100m (°)"]

    extract_columns_to_yaml(input_file, output_file, desired_columns, turbulence_intensity=0.1)