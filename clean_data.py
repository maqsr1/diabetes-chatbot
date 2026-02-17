import pandas as pd

# Load dataset
df = pd.read_csv("diabetes.csv")

# Remove duplicates
df = df.drop_duplicates()

# Fill missing numeric values with column means
numeric_cols = df.select_dtypes(include=["number"]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Fill missing categorical values with mode (if any exist)
categorical_cols = df.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Save cleaned file
df.to_csv("clean_diabetes.csv", index=False)

print("Data cleaned successfully.")