import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .categories import CATEGORIES, CATEGORY_DESCRIPTIONS


class MessageClassifier:

    def __init__(self):

        print("Loading MiniLM...")

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.category_names = list(
            CATEGORY_DESCRIPTIONS.keys()
        )

        self.category_embeddings = self.model.encode(
            list(CATEGORY_DESCRIPTIONS.values()),
            normalize_embeddings=True
        )

        self.keywords = {

            "action_required": [
                "please submit",
                "please complete",
                "please provide",
                "please send",
                "please update",
                "please confirm",
                "submit",
                "complete",
                "fill out",
                "fill in",
                "respond",
                "reply",
                "pay",
                "payment required",
                "register",
                "upload",
                "verify",
                "required",
                "deadline",
                "due by",
                "must",
                "need to",
                "needs to",
                "action required"
            ],

            "meeting_or_event": [
                "meeting",
                "appointment",
                "interview",
                "webinar",
                "conference",
                "event",
                "workshop",
                "seminar",
                "scheduled",
                "schedule",
                "starts at",
                "held on",
                "join us",
                "calendar",
                "venue",
                "location"
            ],

            "promotional": [
                "discount",
                "offer",
                "sale",
                "deal",
                "limited time",
                "buy now",
                "shop now",
                "save",
                "cashback",
                "coupon",
                "promo",
                "promotion",
                "free trial",
                "subscribe",
                "exclusive offer",
                "% off"
            ],

            "sensitive_information": [
                "otp",
                "one time password",
                "one-time password",
                "password",
                "passcode",
                "pin",
                "cvv",
                "credit card",
                "debit card",
                "bank account",
                "account number",
                "authentication token",
                "verification code"
            ],

            "personal_information": [
                "my phone number",
                "my mobile number",
                "my email",
                "my personal email",
                "my address",
                "my home address",
                "my birthday",
                "my date of birth",
                "i live in",
                "i am staying",
                "my new number"
            ],

            "general_information": [
                "information",
                "announcement",
                "notice",
                "update",
                "will be closed",
                "will remain closed",
                "is closed",
                "has been closed",
                "has been announced",
                "please note",
                "inform you",
                "for your information",
                "regarding",
                "we would like to inform",
                "important information",
                "office will be",
                "library will be",
                "school will be",
                "college will be",
                "service will be",
                "system will be"
            ],
        }


    def classify(self, message):

        text = message.lower().strip()

        # --------------------------------------------------
        # 1. Strong rule-based detection
        # --------------------------------------------------

        scores = {
            category: 0
            for category in CATEGORIES
        }

        matched_keywords = {
            category: []
            for category in CATEGORIES
        }

        for category, keywords in self.keywords.items():

            for keyword in keywords:

                if keyword in text:

                    scores[category] += 1

                    matched_keywords[
                        category
                    ].append(keyword)

        # --------------------------------------------------
        # 2. Strong category overrides
        # --------------------------------------------------
        if scores["sensitive_information"] > 0:

            category = "sensitive_information"

            return {
                "category": category,
                "confidence": 0.95,
                "reason": self.generate_reason(
                    category,
                    matched_keywords[category]
                )
            }

        if scores["promotional"] >= 1:

            category = "promotional"

            return {
                "category": category,
                "confidence": min(
                    0.80 + 0.05 * scores[category],
                    0.98
                ),
                "reason": self.generate_reason(
                    category,
                    matched_keywords[category]
                )
            }

        if scores["action_required"] >= 1:

            category = "action_required"

            return {
                "category": category,
                "confidence": min(
                    0.80 + 0.04 * scores[category],
                    0.97
                ),
                "reason": self.generate_reason(
                    category,
                    matched_keywords[category]
                )
            }

        if scores["meeting_or_event"] >= 1:

            category = "meeting_or_event"

            return {
                "category": category,
                "confidence": min(
                    0.80 + 0.04 * scores[category],
                    0.97
                ),
                "reason": self.generate_reason(
                    category,
                    matched_keywords[category]
                )
            }

        if scores["personal_information"] >= 1:

            category = "personal_information"

            return {
                "category": category,
                "confidence": min(
                    0.80 + 0.04 * scores[category],
                    0.95
                ),
                "reason": self.generate_reason(
                    category,
                    matched_keywords[category]
                )
            }
        
        # --------------------------------------------------
        # General information
        # --------------------------------------------------
        if scores["general_information"] >= 1:

                category = "general_information"

                return {
                    "category": category,
                    "confidence": min(
                        0.80 + 0.04 * scores[category],
                        0.95
                    ),
                    "reason": self.generate_reason(
                        category,
                        matched_keywords[category]
                    )
                }

        # --------------------------------------------------
        # 3. MiniLM fallback
        # --------------------------------------------------

        embedding = self.model.encode(
            [message],
            normalize_embeddings=True
        )

        similarities = cosine_similarity(
            embedding,
            self.category_embeddings
        )[0]

        best_index = similarities.argmax()

        category = self.category_names[
            best_index
        ]

        similarity = float(
            similarities[best_index]
        )

        confidence = max(
            0.50,
            min(similarity, 0.85)
        )

        return {
            "category": category,
            "confidence": round(
                confidence,
                4
            ),
            "reason": self.generate_reason(
                category,
                []
            )
        }


    def generate_reason(
        self,
        category,
        matched_keywords
    ):

        if category == "action_required":

            if matched_keywords:

                return (
                    "The message asks the recipient "
                    "to perform an action, indicated by "
                    f"'{matched_keywords[0]}'."
                )

            return (
                "The message appears to require "
                "the recipient to take an action."
            )

        if category == "meeting_or_event":

            if matched_keywords:

                return (
                    "The message contains meeting or "
                    "event-related information, indicated "
                    f"by '{matched_keywords[0]}'."
                )

            return (
                "The message appears to describe "
                "a meeting or scheduled event."
            )

        if category == "promotional":

            if matched_keywords:

                return (
                    "The message contains promotional "
                    "content, indicated by "
                    f"'{matched_keywords[0]}'."
                )

            return (
                "The message appears to contain "
                "promotional content."
            )

        if category == "sensitive_information":

            if matched_keywords:

                return (
                    "The message contains potentially "
                    "sensitive information related to "
                    f"'{matched_keywords[0]}'."
                )

            return (
                "The message appears to contain "
                "sensitive information."
            )

        if category == "personal_information":

            if matched_keywords:

                return (
                    "The message contains personal "
                    "information, indicated by "
                    f"'{matched_keywords[0]}'."
                )

            return (
                "The message appears to contain "
                "personal information."
            )

        return (
            "The message provides general information "
            "without a clear action, event, or promotional intent."
        )