
from extract import extract_data
from transform import transform_data
from load import load_data


def main():

    # Input folder
    input_folder = r"C:\Users\maury\Downloads\archive"

    # Output folder
    output_folder = r"C:\Users\maury\Downloads\InsightAI_Output"

    # =========================
    # EXTRACT
    # =========================
    df = extract_data(input_folder)

    # =========================
    # TRANSFORM
    # =========================
    df = transform_data(df)

    # =========================
    # LOAD
    # =========================
    output = load_data(df, output_folder)

    print("\n========================================")
    print("       ETL PIPELINE COMPLETED")
    print("========================================")
    print("Final Rows:", df.shape[0])
    print("Final Columns:", df.shape[1])
    print("Output File:", output)


if __name__ == "__main__":
    main()
