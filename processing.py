import json
import pandas as pd

from main.classification.classifier import MessageClassifier


INPUT_FILE = "given_data/messages.csv"
OUTPUT_FILE = "outputs/classifications.json"


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df)} messages")


# --------------------------------------------------
# 2. Validate required columns
# --------------------------------------------------

required_columns = {
    "message_id",
    "timestamp",
    "sender",
    "message"
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )


# --------------------------------------------------
# 3. Convert timestamp and sort chronologically
# --------------------------------------------------

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

if df["timestamp"].isna().any():
    print("Warning: Some timestamps could not be parsed.")

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)


# --------------------------------------------------
# 4. Load classifier ONCE
# --------------------------------------------------

classifier = MessageClassifier()


# --------------------------------------------------
# 5. Classify every message
# --------------------------------------------------

results = []

for index, row in df.iterrows():

    result = classifier.classify(
        row["message"]
    )

    output = {
        "message_id": row["message_id"],
        "category": result["category"],
        "confidence": result["confidence"],
        "reason": result["reason"]
    }

    results.append(output)

    # Progress indicator
    if (index + 1) % 50 == 0:
        print(
            f"Processed {index + 1}/{len(df)} messages"
        )


# --------------------------------------------------
# 6. Save JSON
# --------------------------------------------------

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False
    )


print("\nClassification completed.")
print(f"Total messages: {len(results)}")
print(f"Output: {OUTPUT_FILE}")