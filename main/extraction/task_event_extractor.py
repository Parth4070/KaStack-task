import re

import dateparser
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class TaskEventExtractor:

    def __init__(self):

        print("Loading MiniLM for task/event detection...")

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.nlp = spacy.load(
            "en_core_web_sm"
        )
        self.intent_examples = {

            "task": [
                "Please complete this task.",
                "Please submit the required document.",
                "The recipient needs to perform an action.",
                "Someone is being asked to complete some work.",
                "Please review the document.",
                "You need to send something.",
                "The user must complete an activity.",
                "Someone is asked to pay or respond.",
                "Please finish the assigned work."
            ],

            "event": [
                "There is a scheduled meeting.",
                "An appointment is scheduled.",
                "Someone is invited to an event.",
                "There are details about a scheduled activity.",
                "A meeting will take place at a particular time.",
                "An interview is scheduled.",
                "A webinar or conference is taking place.",
                "Someone is invited to join an orientation.",
                "There is a scheduled dinner or gathering."
            ],

            "none": [
                "This message only provides information.",
                "This is a general informational message.",
                "The message does not require an action.",
                "The message does not describe a scheduled event.",
                "This is an update with no action required.",
                "The sender is simply providing information.",
                "This message expresses an opinion or preference.",
                "The message describes something that already happened.",
                "This is a general announcement."
            ]
        }

        self.intent_names = list(
            self.intent_examples.keys()
        )

        all_examples = []

        for intent in self.intent_names:
            all_examples.extend(
                self.intent_examples[intent]
            )

        self.example_embeddings = self.model.encode(
            all_examples,
            normalize_embeddings=True
        )

        self.example_intents = []

        for intent in self.intent_names:

            self.example_intents.extend([intent] * len(self.intent_examples[intent]))

    def detect_intent(self, text):

        embedding = self.model.encode(
            [text],
            normalize_embeddings=True
        )

        similarities = cosine_similarity(
            embedding,
            self.example_embeddings
        )[0]


        intent_scores = {}

        for intent in self.intent_names:

            scores = [
                similarities[i]
                for i, example_intent
                in enumerate(self.example_intents)
                if example_intent == intent
            ]

            intent_scores[intent] = max(scores)

        ranked = sorted(
            intent_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        best_intent = ranked[0][0]
        best_score = float(ranked[0][1])

        second_score = float(
            ranked[1][1]
        )

        margin = best_score - second_score

        if best_score < 0.40 or margin < 0.03:

            return {
                "intent": "none",
                "confidence": round(
                    best_score,
                    4
                ),
                "uncertain": True,
                "scores": {
                    key: round(
                        float(value),
                        4
                    )
                    for key, value
                    in intent_scores.items()
                }
            }

        return {
            "intent": best_intent,
            "confidence": round(
                best_score,
                4
            ),
            "uncertain": False,
            "scores": {
                key: round(
                    float(value),
                    4
                )
                for key, value
                in intent_scores.items()
            }
        }


    def extract_date(
        self,
        text,
        message_timestamp
    ):

        match = re.search(
            r"\b20\d{2}-\d{2}-\d{2}\b",
            text
        )

        if match:
            return match.group()

        match = re.search(
            r"\b\d{1,2}[/-]\d{1,2}[/-]20\d{2}\b",
            text
        )

        if match:

            parsed = dateparser.parse(
                match.group(),
                settings={
                    "DATE_ORDER": "DMY"
                }
            )

            if parsed:
                return parsed.strftime(
                    "%Y-%m-%d"
                )

        relative_pattern = (
            r"\b("
            r"today|tomorrow|"
            r"day after tomorrow|"
            r"next monday|next tuesday|"
            r"next wednesday|next thursday|"
            r"next friday|next saturday|"
            r"next sunday"
            r")\b"
        )

        match = re.search(
            relative_pattern,
            text,
            re.IGNORECASE
        )

        if match:

            parsed = dateparser.parse(
                match.group(),
                settings={
                    "RELATIVE_BASE":
                        message_timestamp.to_pydatetime(),
                    "PREFER_DATES_FROM":
                        "future"
                }
            )

            if parsed:
                return parsed.strftime(
                    "%Y-%m-%d"
                )

        return None

    def extract_time(self, text):

        match = re.search(
            r"\b([01]\d|2[0-3]):([0-5]\d)\b",
            text
        )

        if match:

            return (
                f"{int(match.group(1)):02d}:"
                f"{int(match.group(2)):02d}"
            )

        match = re.search(
            r"\b(1[0-2]|0?[1-9])"
            r"(?::([0-5]\d))?"
            r"\s*(AM|PM)\b",
            text,
            re.IGNORECASE
        )

        if match:

            hour = int(match.group(1))
            minute = int(
                match.group(2) or 0
            )

            period = match.group(3).upper()

            if period == "PM" and hour != 12:
                hour += 12

            if period == "AM" and hour == 12:
                hour = 0

            return f"{hour:02d}:{minute:02d}"

        return None

    def extract_person(self, text):

        doc = self.nlp(text)

        for entity in doc.ents:

            if entity.label_ == "PERSON":

                return entity.text.strip()

        return None


    def extract_title(self, text, intent):
        calendar_match = re.search(
            r"(?:calendar update|calendar)\s*:\s*([^,]+)",
            text,
            re.IGNORECASE
        )

        if calendar_match:
            return calendar_match.group(1).strip().title()

        if intent == "task":

            patterns = [
                r"please\s+(.+?)(?:\s+by\s+20\d{2}-\d{2}-\d{2}|[?.]|$)",

                r"can you\s+(.+?)(?:\s+by\s+20\d{2}-\d{2}-\d{2}|[?.]|$)",

                r"i need you to\s+(.+?)(?:\s+by\s+20\d{2}-\d{2}-\d{2}|[?.]|$)",

                r"don't forget to\s+(.+?)(?:;|\s+by\s+20\d{2}-\d{2}-\d{2}|[?.]|$)"
            ]

            for pattern in patterns:

                match = re.search(
                    pattern,
                    text,
                    re.IGNORECASE
                )

                if match:

                    title = match.group(1).strip()

                    return title[0].upper() + title[1:]

            return "Task"


        if intent == "event":
            match = re.search(
                r"(?:join|attend)\s+(?:the\s+)?"
                r"(.+?)(?:\s+on\s+20\d{2}-\d{2}-\d{2}"
                r"|\s+at\s+\d{1,2}(?::\d{2})?\s*(?:AM|PM)?"
                r"|[.,]|$)",
                text,
                re.IGNORECASE
            )

            if match:
                return match.group(1).strip().title()   
            event_words = [
                "meeting",
                "appointment",
                "interview",
                "webinar",
                "orientation",
                "workshop",
                "seminar",
                "event",
                "dinner",
                "session",
                "call"
            ]

            for word in event_words:

                if re.search(
                    rf"\b{re.escape(word)}\b",
                    text,
                    re.IGNORECASE
                ):
                    return word.title()

            return "Event"

        return None

    def extract_priority(self, text):

        lower = text.lower()

        high_words = [
            "urgent",
            "urgently",
            "asap",
            "immediately",
            "critical",
            "important",
            "deadline",
            "due today"
        ]

        if any(
            word in lower
            for word in high_words
        ):
            return "high"

        return "medium"

    def extract(
        self,
        message_id,
        text,
        message_timestamp
    ):

        intent_result = self.detect_intent(
            text
        )

        intent = intent_result["intent"]

  
        if intent == "none":

            return None

        if intent_result["uncertain"]:

            return None

        date = self.extract_date(
            text,
            message_timestamp
        )

        time = self.extract_time(
            text
        )

        person = self.extract_person(
            text
        )

        title = self.extract_title(
            text,
            intent
        )

        priority = self.extract_priority(
            text
        )

        return {
            "item_id": None,
            "type": intent,
            "title": title,
            "description": text,
            "date": (
                date
                if intent == "event"
                else None
            ),
            "deadline": (
                date
                if intent == "task"
                else None
            ),
            "time": time,
            "person": person,
            "priority": priority,
            "source_message_id": message_id,

            "confidence": intent_result[
                "confidence"
            ]
        }