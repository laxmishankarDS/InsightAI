import pandas as pd
from pathlib import Path


def extract_data(folder_path):

    print("\n========== EXTRACT STAGE ==========")

    folder_path = Path(folder_path)

    readers = {
        ".csv": pd.read_csv,
        ".xlsx": pd.read_excel,
        ".xls": pd.read_excel,
    }

    files = [f for f in folder_path.iterdir() if f.is_file()]

    if not files:
        raise FileNotFoundError(
            f"No files found in: {folder_path}"
        )

    dataframes = []

    for file_path in files:

        extension = file_path.suffix.lower()

        reader = readers.get(extension)

        if reader is None:
            print(f"Skipping unsupported file: {file_path.name}")
            continue

        print(f"Reading: {file_path.name}")

        df = reader(file_path)

        dataframes.append(df)

    if not dataframes:
        raise ValueError("No supported CSV/Excel files found.")

    final_df = pd.concat(dataframes, ignore_index=True)

    print("Rows:", final_df.shape[0])
    print("Columns:", final_df.shape[1])

    return final_df

