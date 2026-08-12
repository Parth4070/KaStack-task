from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel

from main.classification.classifier import MessageClassifier
from main.extraction.task_event_extractor import TaskEventExtractor
from main.sensitive.detector import SensitiveDetector


app = FastAPI(
    title="Message Intelligence API",
    description="Local NLP pipeline for message classification, task/event extraction and sensitive information detection.",
    version="1.0.0"
)



print("Loading message classifier...")
classifier = MessageClassifier()

print("Loading task/event extractor...")
task_event_extractor = TaskEventExtractor()

print("Loading sensitive detector...")
sensitive_detector = SensitiveDetector()

print("All models loaded.")

class MessageRequest(BaseModel):

    message_id: str
    timestamp: datetime
    sender: str
    message: str


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }



@app.post("/analyze")
def analyze_message(
    request: MessageRequest
):

    classification = classifier.classify(
        request.message
    )

    task_event = task_event_extractor.extract(
        message_id=request.message_id,
        text=request.message,
        message_timestamp=request.timestamp
    )


    sensitive = sensitive_detector.detect(
        message_id=request.message_id,
        text=request.message
    )


    if sensitive["detected"]:

        sensitive_output = {
            "detected": True,
            "sensitivity_type":
                sensitive["sensitivity_type"],
            "risk":
                sensitive["risk"],
            "masked_text":
                sensitive["masked_text"],
            "recommended_action":
                sensitive["recommended_action"]
        }

    else:

        sensitive_output = {
            "detected": False
        }

    if task_event:

        task_event_output = {
            "type":
                task_event["type"],
            "title":
                task_event["title"],
            "description":
                task_event["description"],
            "date":
                task_event["date"],
            "deadline":
                task_event["deadline"],
            "time":
                task_event["time"],
            "person":
                task_event["person"],
            "priority":
                task_event["priority"]
        }

    else:

        task_event_output = None

    return {
        "message_id": request.message_id,

        "classification": {
            "category":
                classification["category"],
            "confidence":
                classification["confidence"],
            "reason":
                classification["reason"]
        },

        "task_event":
            task_event_output,

        "sensitive":
            sensitive_output
    }