
import os
import pandas as pd


def extract_data(folder_path):
    """
    Extract Excel files from the given folder.
    """

    print("\n========== EXTRACT STAGE ==========")

    if not os.path.exists(folder_path):
        raise FileNotFoundError(
            f"Folder not found: {folder_path}"
        )

    # Find Excel files
    excel_files = [
        file for file in os.listdir(folder_path)
        if file.lower().endswith((".xlsx", ".xls"))
    ]

    if not excel_files:
        raise FileNotFoundError(
            f"No Excel files found in: {folder_path}"
        )

    dataframes = []

    for file in excel_files:

        file_path = os.path.join(folder_path, file)

        print(f"Reading: {file}")

        df = pd.read_excel(file_path)

        # Add source filename
        df["Source_File"] = file

        dataframes.append(df)

    # Combine all Excel files
    final_df = pd.concat(
        dataframes,
        ignore_index=True
    )

    print("\nExtraction Successful")
    print("Files:", len(excel_files))
    print("Rows:", final_df.shape[0])
    print("Columns:", final_df.shape[1])

    return final_df

