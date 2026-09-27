# ============================================================
# INSIGHTAI
# DYNAMIC DATA EXTRACTION
# ============================================================

import json
import pandas as pd
from pathlib import Path


# ============================================================
# SUPPORTED FILE TYPES
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".csv",
    ".xlsx",
    ".xls",
    ".json",
}


# ============================================================
# READ SINGLE FILE
# ============================================================

def read_file(file_path):
    """
    Read a supported dataset file and return a pandas DataFrame.
    """

    file_path = Path(file_path)
    extension = file_path.suffix.lower()

    print(f"Reading: {file_path.name}")

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    if extension == ".csv":

        return pd.read_csv(file_path)

    # --------------------------------------------------------
    # EXCEL
    # --------------------------------------------------------

    if extension in {".xlsx", ".xls"}:

        return pd.read_excel(file_path)

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    if extension == ".json":

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        # ----------------------------------------------------
        # JSON LIST OF RECORDS
        # ----------------------------------------------------

        if isinstance(data, list):

            return pd.DataFrame(data)

        # ----------------------------------------------------
        # JSON DICTIONARY
        # ----------------------------------------------------

        if isinstance(data, dict):

            # Try normal DataFrame construction first
            try:

                return pd.DataFrame(data)

            except Exception:

                # Fallback for nested record-style JSON
                return pd.json_normalize(data)

        raise ValueError(
            f"Unsupported JSON structure in: {file_path.name}"
        )

    # --------------------------------------------------------
    # UNSUPPORTED FILE
    # --------------------------------------------------------

    raise ValueError(
        f"Unsupported file type: {extension}"
    )


# ============================================================
# EXTRACT DATA
# ============================================================

def extract_data(folder_path):

    print("\n========== EXTRACT STAGE ==========")

    # --------------------------------------------------------
    # VALIDATE INPUT FOLDER
    # --------------------------------------------------------

    folder_path = Path(folder_path)

    if not folder_path.exists():

        raise FileNotFoundError(
            f"Input folder does not exist: {folder_path}"
        )

    if not folder_path.is_dir():

        raise NotADirectoryError(
            f"Input path is not a folder: {folder_path}"
        )

    # --------------------------------------------------------
    # FIND FILES
    # --------------------------------------------------------

    files = sorted(
        [
            file_path
            for file_path in folder_path.iterdir()
            if (
                file_path.is_file()
                and file_path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            )
        ]
    )

    # --------------------------------------------------------
    # NO FILES
    # --------------------------------------------------------

    if not files:

        raise FileNotFoundError(
            f"No supported dataset files found in: "
            f"{folder_path}"
        )

    print(
        f"Supported files found: {len(files)}"
    )

    # --------------------------------------------------------
    # READ FILES
    # --------------------------------------------------------

    dataframes = []

    for file_path in files:

        try:

            df = read_file(file_path)

            # -----------------------------------------------
            # BASIC VALIDATION
            # -----------------------------------------------

            if df is None:

                print(
                    f"WARNING: No data returned from "
                    f"{file_path.name}"
                )

                continue

            if df.empty:

                print(
                    f"WARNING: Empty file skipped: "
                    f"{file_path.name}"
                )

                continue

            print(
                f"Rows    : {df.shape[0]}"
            )

            print(
                f"Columns : {df.shape[1]}"
            )

            dataframes.append(df)

        except Exception as error:

            print(
                f"WARNING: Failed to read "
                f"{file_path.name}"
            )

            print(
                f"Reason: {error}"
            )

    # --------------------------------------------------------
    # CHECK SUCCESSFUL READS
    # --------------------------------------------------------

    if not dataframes:

        raise ValueError(
            "No valid dataset files could be read."
        )

    # --------------------------------------------------------
    # COMBINE DATASETS
    # --------------------------------------------------------

    print(
        "\nCombining datasets..."
    )

    final_df = pd.concat(
        dataframes,
        ignore_index=True,
        sort=False
    )

    # --------------------------------------------------------
    # BASIC INFORMATION
    # --------------------------------------------------------

    print(
        "\n========== EXTRACTION SUMMARY =========="
    )

    print(
        "Files processed:",
        len(dataframes)
    )

    print(
        "Rows:",
        final_df.shape[0]
    )

    print(
        "Columns:",
        final_df.shape[1]
    )

    print(
        "\nColumns detected:"
    )

    for column in final_df.columns:

        print(
            " -",
            column
        )

    print(
        "\nExtraction Successful"
    )

    return final_df


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    INPUT_FOLDER = (
        r"C:\Users\maury\Downloads\archive"
    )

    try:

        data = extract_data(
            INPUT_FOLDER
        )

        print(
            "\n========== TEST RESULT =========="
        )

        print(
            "Rows:",
            data.shape[0]
        )

        print(
            "Columns:",
            data.shape[1]
        )

        print(
            "\nFirst 5 rows:"
        )

        print(
            data.head()
        )

    except Exception as error:

        print(
            "\n========== EXTRACTION ERROR =========="
        )

        print(
            error
        )
