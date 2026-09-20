import os
import pandas as pd


def extract_data(folder_path):
    """
    Extract all CSV files from a folder.
    """

    print("\n========== EXTRACT STAGE ==========")

    dataframes = []

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    csv_files = [file for file in os.listdir(folder_path) if file.endswith(".csv")]

    if not csv_files:
        raise Exception("No CSV files found.")

    for file in csv_files:

        file_path = os.path.join(folder_path, file)

        print(f"Reading {file}")

        df = pd.read_excel(r"C:/Users/maury/Downloads/sales_data_sample.xlsx")

        df["Source_File"] = file

        dataframes.append(df)

    final_df = pd.concat(dataframes, ignore_index=True)

    print("\nExtraction Successful")
    print("Rows :", final_df.shape[0])
    print("Columns :", final_df.shape[1])

    return final_df
