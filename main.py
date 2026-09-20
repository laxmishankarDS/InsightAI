from extract import extract_data
from transform import transform_data
from load import load_data


def main():

    input_folder = r"C:\Users\maury\Downloads\sales_data_sample.xlsx"

    output_folder = r"C:/Users/bhola/Downloads/InsightAI_Output"

    # Extract
    df = extract_data(input_folder)

    # Transform
    df = transform_data(df)

    # Load
    output = load_data(df, output_folder)

    print("\n========== ETL PIPELINE COMPLETED ==========")

    print("Output File :", output)


if __name__ == "__main__":
    main()