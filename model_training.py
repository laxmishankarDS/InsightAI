import pandas as pd
import json

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ========================================
# READ DATA
# ========================================

df = pd.read_csv(
    r"C:\Users\maury\Downloads\InsightAI_Output\feature_data.csv"
)

print("========== MODEL TRAINING ==========")


# ========================================
# READ SELECTION
# ========================================

with open("selection.json", "r") as file:
    selection = json.load(file)

target = selection["target"]
time = selection["time"]
group = selection["group"]
problem = selection["problem"]

print("Selected Target:", target)
print("Selected Problem:", problem)
print("Selected Time:", time)
print("Selected Group:", group)


# ========================================
# AUTOMATIC FEATURE SELECTION
# ========================================

print("\n========== FEATURE SELECTION ==========")

features = []

for col in df.columns:

    name = col.lower()

    # Target ko feature nahi banana
    if col == target:
        continue

    # ID columns ignore
    if name == "id" or name.endswith("_id"):
        continue

    # Date columns ignore
    if "date" in name:
        continue

    # Empty / constant columns ignore
    if df[col].nunique() <= 1:
        continue

    # Numeric columns
    if pd.api.types.is_numeric_dtype(df[col]):
        features.append(col)

    # Low-cardinality categorical columns
    elif pd.api.types.is_object_dtype(df[col]):
        if df[col].nunique() <= 20:
            features.append(col)


print("Selected Features:")

for feature in features:
    print("-", feature)

print("Total Features:", len(features))


# ========================================
# LEAKAGE CHECK
# ========================================

print("\n========== LEAKAGE CHECK ==========")

print("Target:", target)

print("\nFeatures used:")

for feature in features:
    print("-", feature)


# ========================================
# CORRELATION CHECK
# ========================================

print("\n========== CORRELATION CHECK ==========")

if pd.api.types.is_numeric_dtype(df[target]):

    numeric_features = [
        col for col in features
        if pd.api.types.is_numeric_dtype(df[col])
    ]

    if len(numeric_features) > 0:

        correlations = df[numeric_features].corrwith(df[target])

        correlations = correlations.abs().sort_values(
            ascending=False
        )

        for feature, correlation in correlations.items():

            print(
                f"{feature}: {correlation:.4f}"
            )

            if correlation >= 0.95:
                print(
                    f"WARNING: {feature} may contain target leakage."
                )

    else:
        print("No numeric features available.")

else:

    print(
        "Correlation check skipped because target is not numeric."
    )


# ========================================
# X AND Y
# ========================================

X = df[features]
y = df[target]


# ========================================
# TRAIN TEST SPLIT
# ========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# ========================================
# MODEL
# ========================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# ========================================
# TRAIN MODEL
# ========================================

model.fit(X_train, y_train)


# ========================================
# CROSS VALIDATION
# ========================================

cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

print("\n========== CROSS VALIDATION ==========")

print("Fold R2 Scores:", cv_scores)
print("Mean CV R2:", cv_scores.mean())
print("Std CV R2:", cv_scores.std())


# ========================================
# PREDICTION
# ========================================

y_pred = model.predict(X_test)


# ========================================
# EVALUATION
# ========================================

mae = mean_absolute_error(y_test, y_pred)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

r2 = r2_score(y_test, y_pred)


# ========================================
# ADJUSTED R2
# ========================================

n = len(y_test)
p = X_test.shape[1]

if n > p + 1:

    adjusted_r2 = 1 - (
        (1 - r2) * (n - 1) / (n - p - 1)
    )

else:

    adjusted_r2 = None


# ========================================
# MODEL RESULTS
# ========================================

print("\n========== MODEL RESULTS ==========")

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)
print("Adjusted R2:", adjusted_r2)


# ========================================
# FINAL
# ========================================

print("\n========== TRAINING COMPLETED ==========")

