import json
import pandas as pd


MESSAGES_FILE = "given_data/messages.csv"
CLASSIFICATION_FILE = "outputs/classifications.json"
TASK_EVENT_FILE = "outputs/tasks_events.json"
SENSITIVE_FILE = "outputs/sensitive_information.json"

OUTPUT_FILE = "outputs/final_results.json"

messages = pd.read_csv(MESSAGES_FILE)

with open(CLASSIFICATION_FILE, "r", encoding="utf-8") as f:
    classifications = json.load(f)

with open(TASK_EVENT_FILE, "r", encoding="utf-8") as f:
    task_events = json.load(f)

with open(SENSITIVE_FILE, "r", encoding="utf-8") as f:
    sensitive_results = json.load(f)


classification_map = {
    item["message_id"]: item
    for item in classifications
}

task_event_map = {
    item["source_message_id"]: item
    for item in task_events
}

sensitive_map = {
    item["message_id"]: item
    for item in sensitive_results
}


final_results = []


for _, row in messages.iterrows():

    message_id = row["message_id"]

    classification = classification_map.get(
        message_id
    )

    task_event = task_event_map.get(
        message_id
    )

    sensitive = sensitive_map.get(
        message_id
    )


    classification_output = {
        "category": classification["category"],
        "confidence": classification["confidence"],
        "reason": classification["reason"]
    }


    if task_event:

        task_event_output = {
            "item_id": task_event["item_id"],
            "type": task_event["type"],
            "title": task_event["title"],
            "description": task_event["description"],
            "date": task_event["date"],
            "deadline": task_event["deadline"],
            "time": task_event["time"],
            "person": task_event["person"],
            "priority": task_event["priority"]
        }

    else:

        task_event_output = None



    if sensitive:

        sensitive_output = {
            "detected": True,
            "sensitivity_type":
                sensitive["sensitivity_type"],
            "risk": sensitive["risk"],
            "masked_text":
                sensitive["masked_text"],
            "recommended_action":
                sensitive["recommended_action"]
        }

    else:

        sensitive_output = {
            "detected": False
        }


    final_results.append({

        "message_id": message_id,

        "classification":
            classification_output,

        "task_event":
            task_event_output,

        "sensitive":
            sensitive_output
    })


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        final_results,
        f,
        indent=2,
        ensure_ascii=False
    )


print("Final integration completed.")
print(f"Messages: {len(final_results)}")
print(f"Output: {OUTPUT_FILE}")