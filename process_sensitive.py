import json
import pandas as pd

from main.sensitive.detector import SensitiveDetector


INPUT_FILE = "given_data/messages.csv"
OUTPUT_FILE = "outputs/sensitive_information.json"


df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df)} messages")

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


df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)


detector = SensitiveDetector()


results = []

detected_count = 0


for index, row in df.iterrows():

    result = detector.detect(
        message_id=row["message_id"],
        text=row["message"]
    )

    if result["detected"]:

        results.append({
            "message_id": result["message_id"],
            "sensitivity_type":
                result["sensitivity_type"],
            "risk": result["risk"],
            "masked_text":
                result["masked_text"],
            "recommended_action":
                result["recommended_action"]
        })

        detected_count += 1

    if (index + 1) % 50 == 0:

        print(
            f"Processed "
            f"{index + 1}/{len(df)} messages"
        )

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

print("\nSensitive detection completed.")
print(f"Total messages: {len(df)}")
print(f"Sensitive messages: {detected_count}")
print(
    f"Non-sensitive messages: "
    f"{len(df) - detected_count}"
)
print(f"Output: {OUTPUT_FILE}")