import pandas as pd
import json

file = r"C:\Users\maury\Downloads\InsightAI_Output\feature_data.csv"

df = pd.read_csv(file)

print("========== TARGET SELECTION ==========")

targets = []

for col in df.columns:

    name = col.lower()
    unique = df[col].nunique()

    # Ignore ID columns
    if name == "id" or name.endswith("_id"):
        continue

    # Ignore date columns
    if "date" in name:
        continue

    # Ignore time features
    if name in ["year", "month", "quarter", "day_of_week"]:
        continue

    # Ignore constant columns
    if unique <= 1:
        continue

    # Classification
    if pd.api.types.is_string_dtype(df[col]):
        targets.append((col, "Classification"))

    # Numeric
    elif pd.api.types.is_numeric_dtype(df[col]):

        if unique <= 10:
            targets.append((col, "Classification"))
        else:
            targets.append((col, "Regression"))


# ========================================
# TARGET
# ========================================

print("\nPossible Targets:")

for i, (col, problem) in enumerate(targets, 1):
    print(f"{i}. {col} → {problem}")

choice = int(input("\nSelect target number: "))

if choice < 1 or choice > len(targets):
    raise ValueError("Invalid target selection.")

target, problem = targets[choice - 1]


# ========================================
# TIME
# ========================================

print("\n========== TIME SELECTION ==========")

date_columns = []

for col in df.columns:
    if "date" in col.lower():
        date_columns.append(col)

if date_columns:

    print("Date column:", date_columns[0])

    print("\n1. Day")
    print("2. Week")
    print("3. Month")
    print("4. Year")
    print("5. None")

    time_choice = int(input("\nSelect time: "))

    time_options = {
        1: "day",
        2: "week",
        3: "month",
        4: "year",
        5: "none"
    }

    if time_choice not in time_options:
        raise ValueError("Invalid time selection.")

    time = time_options[time_choice]

else:

    time = "none"

    print("No date column found.")
    print("Time selection skipped.")


# ========================================
# GROUP
# ========================================

print("\n========== GROUP SELECTION ==========")

groups = []

for col in df.columns:

    if col == target:
        continue

    name = col.lower()

    if name == "id" or name.endswith("_id"):
        continue

    if "date" in name:
        continue

    if pd.api.types.is_string_dtype(df[col]):
        groups.append(col)


if groups:

    for i, col in enumerate(groups, 1):
        print(f"{i}. {col}")

    print(f"{len(groups) + 1}. None")

    group_choice = int(input("\nSelect group: "))

    if group_choice == len(groups) + 1:
        group = "none"

    elif 1 <= group_choice <= len(groups):
        group = groups[group_choice - 1]

    else:
        raise ValueError("Invalid group selection.")

else:

    group = "none"

    print("No group available.")


# ========================================
# FINAL RESULT
# ========================================

print("\n========== FINAL SELECTION ==========")

print("Target:", target)
print("Problem:", problem)
print("Time:", time)
print("Group:", group)

#========selection========

selection = {
    "target": target,
    "problem": problem,
    "time": time,
    "group": group
}

with open("selection.json", "w") as file:
    json.dump(selection, file, indent=4)

print("\nSelection saved to: selection.json")






