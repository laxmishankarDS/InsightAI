
import os
import pandas as pd


def load_data(df, output_folder):
    """
    Save the transformed DataFrame as a CSV file.
    """

    print("\n========== LOAD STAGE ==========")

    if df is None or df.empty:
        raise ValueError("No data available to load.")

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    output_file = os.path.join(
        output_folder,
        "final_output.csv"
    )

    df.to_csv(output_file, index=False)

    print("Load Successful")
    print("Output File:", output_file)

    return output_file

