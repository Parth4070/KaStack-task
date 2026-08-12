import re

from gliner import GLiNER


class SensitiveDetector:

    def __init__(self):

        print("Loading GLiNER...")

        self.model = GLiNER.from_pretrained(
            "gliner-community/gliner_small-v2.5"
        )

        self.labels = [
            "password",
            "one time password",
            "PIN",
            "credit card number",
            "debit card number",
            "bank account number",
            "authentication token",
            "government identification number",
        ]

    def detect_patterns(self, text):

        patterns = {

            "one_time_password":
                r"\b(?:otp|one[- ]time password|verification code)"
                r"\s*(?:is|:|=|-)?\s*\d{4,8}\b",

            "password":
                r"\b(?:password|passwd|pwd|passcode)"
                r"\s*(?:is|:|=|-)\s*\S+",

            "pin":
                r"\b(?:pin|atm pin)"
                r"\s*(?:is|:|=|-)?\s*\d{4,6}\b",

            "credit_card":
                r"\b(?:\d{4}[- ]?){3}\d{4}\b",

            "bank_account":
                r"\b(?:account number|account no\.?|a/c number)"
                r"\s*(?:is|:|=|-)?\s*\d{6,18}\b",

            "pan_number":
                r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",

            "aadhaar_number":
                r"\b\d{4}[\s-]\d{4}[\s-]\d{4}\b",

            "phone_number":
                r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)",

            "email_address":
                r"\b[A-Za-z0-9._%+-]+"
                r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",

            "authentication_token":
                r"\b(?:token|api key|access token)"
                r"\s*(?:is|:|=|-)\s*[A-Za-z0-9._\-]{10,}\b",
            
            "private_address":
                r"\b\d{1,5}\s+"
                r"(?:[A-Za-z0-9]+\s+){1,5}"
                r"(?:Street|St|Road|Rd|Avenue|Ave|"
                r"Lane|Ln|Drive|Dr|Boulevard|Blvd|"
                r"Apartment|Apt|Colony|Nagar|Society)"
                r"\b"
        }

        detections = []

        for label, pattern in patterns.items():

            for match in re.finditer(
                pattern,
                text,
                re.IGNORECASE
            ):

                detections.append({
                    "type": label,
                    "start": match.start(),
                    "end": match.end(),
                    "score": 1.0
                })

        return detections

    def detect_gliner(self, text):

        entities = self.model.predict_entities(
            text,
            self.labels,
            threshold=0.65
        )

        results = []

        for entity in entities:

            results.append({
                "type": self.normalize_type(
                    entity["label"]
                ),
                "start": entity["start"],
                "end": entity["end"],
                "score": float(
                    entity["score"]
                )
            })

        return results

    def normalize_type(self, label):

        mapping = {

            "one time password":
                "one_time_password",

            "password":
                "password",

            "pin":
                "pin",

            "credit card number":
                "credit_card",

            "debit card number":
                "debit_card",

            "bank account number":
                "bank_account",

            "authentication token":
                "authentication_token",

            "government identification number":
                "government_id",

            "private address":
                "private_address",

            "phone number":
                "phone_number",

            "email address":
                "email_address"
        }

        return mapping.get(
            label.lower(),
            label.lower().replace(" ", "_")
        )

    def get_risk(self, sensitivity_type):

        high_risk = {
            "one_time_password",
            "password",
            "pin",
            "credit_card",
            "debit_card",
            "bank_account",
            "authentication_token",
            "government_id"
        }

        medium_risk = {
            "private_address",
            "phone_number",
            "email_address"
        }

        if sensitivity_type in high_risk:
            return "high"

        if sensitivity_type in medium_risk:
            return "medium"

        return "low"

    def get_action(self, risk):

        if risk == "high":
            return "do_not_store"

        if risk == "medium":
            return "ask_for_confirmation"

        return "safe_to_process_locally"

    def mask_text(self, text, detections):

        detections = sorted(
            detections,
            key=lambda x: x["start"]
        )

        filtered = []

        for detection in detections:

            if not filtered:

                filtered.append(detection)
                continue

            previous = filtered[-1]

            if detection["start"] < previous["end"]:

                if detection["score"] > previous["score"]:
                    filtered[-1] = detection

            else:

                filtered.append(detection)

        masked = text

        for detection in reversed(filtered):

            start = detection["start"]
            end = detection["end"]

            masked = (
                masked[:start]
                + "*" * (end - start)
                + masked[end:]
            )

        return masked

    def detect(self, message_id, text):

        pattern_results = self.detect_patterns(
            text
        )

        gliner_results = self.detect_gliner(
            text
        )

        detections = (
            pattern_results
            + gliner_results
        )

        if not detections:

            return {
                "message_id": message_id,
                "detected": False,
                "sensitivity_type": None,
                "risk": None,
                "masked_text": text,
                "recommended_action":
                    "safe_to_process_locally"
            }

        primary = max(
            detections,
            key=lambda x: x["score"]
        )

        sensitivity_type = primary["type"]

        risk = self.get_risk(
            sensitivity_type
        )

        masked_text = self.mask_text(
            text,
            detections
        )

        return {
            "message_id": message_id,
            "detected": True,
            "sensitivity_type": sensitivity_type,
            "risk": risk,
            "masked_text": masked_text,
            "recommended_action":
                self.get_action(risk)
        }