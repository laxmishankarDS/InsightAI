import os


def load_data(df, output_folder):

    print("\n========== LOAD STAGE ==========")

    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(
        output_folder,
        "cleaned_sales_data.csv"
    )

    df.to_csv(output_path, index=False)

    print(f"File Saved Successfully\n{output_path}")

    return output_path