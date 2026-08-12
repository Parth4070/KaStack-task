import json
import pandas as pd

from main.extraction.task_event_extractor import (TaskEventExtractor)


INPUT_FILE = "given_data/messages.csv"
OUTPUT_FILE = "outputs/tasks_events.json"

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


extractor = TaskEventExtractor()

results = []

task_count = 0
event_count = 0
uncertain_count = 0


for index, row in df.iterrows():

    result = extractor.extract(
        message_id=row["message_id"],
        text=row["message"],
        message_timestamp=row["timestamp"]
    )

    if result is not None:
        if result["type"] == "task":

            task_count += 1
            result["item_id"] = (
                f"TASK_{task_count:03d}"
            )

        else:

            event_count += 1
            result["item_id"] = (
                f"EVENT_{event_count:03d}"
            )

        results.append(result)

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


print("\nExtraction completed.")
print(f"Total messages: {len(df)}")
print(f"Tasks extracted: {task_count}")
print(f"Events extracted: {event_count}")
print(f"Total extracted: {len(results)}")
print(f"Output: {OUTPUT_FILE}")