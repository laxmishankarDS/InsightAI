from ml.ml_engine import (
    load_dataset,
    train_regression_models
)


CSV_PATH = r"C:\Users\Ayush\Downloads\archive\processed\cleaned_product_sales.csv"


print("=" * 60)
print("INSIGHTAI ML TEST")
print("=" * 60)


# Load dataset
df = load_dataset(CSV_PATH)


# Train models
result = train_regression_models(
    df,
    target_column="Revenue"
)


print("\n")
print("=" * 60)
print("MODEL RESULTS")
print("=" * 60)


for model in result["models"]:

    print("\nModel:", model["model"])

    if "error" in model:

        print("Error:", model["error"])

    else:

        print("MAE:", model["mae"])
        print("RMSE:", model["rmse"])
        print("R²:", model["r2"])


print("\n")
print("=" * 60)
print("BEST MODEL")
print("=" * 60)

print(
    "Model:",
    result["best_model"]
)

print(
    "R²:",
    result["best_metrics"]["r2"]
)

print(
    "MAE:",
    result["best_metrics"]["mae"]
)

print(
    "RMSE:",
    result["best_metrics"]["rmse"]
)


print("\n")
print("=" * 60)
print("LEAKAGE DETECTION")
print("=" * 60)

print(
    "Removed leakage columns:",
    result["removed_leakage_columns"]
)

print(
    "Removed ID columns:",
    result["removed_id_columns"]
)