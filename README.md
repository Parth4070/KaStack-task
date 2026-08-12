# KaStack Task

This repository contains the solution for the KaStack message processing assignment.

## L1 Candidate Dataset Information
- `messages.csv` contains 900 fictional messages in chronological order. Columns: message_id, timestamp, sender, message. No answer labels are included.
- `mandatory_demo_ids.csv` lists the 15 message IDs that must be shown in the video.
- All sensitive-looking values are fictional but must still be masked as required by the assignment.

## How Message Classification Works

The message classification system uses a hybrid approach combining strong rule-based detection with a semantic search fallback mechanism:

1.  **Rule-Based Detection:** The system first scans the incoming message text for predefined keywords associated with specific categories (`action_required`, `meeting_or_event`, `promotional`, `sensitive_information`, `personal_information`, `general_information`). It assigns scores based on keyword matches.
2.  **Category Overrides:** If a message strongly matches specific rules (e.g., contains words like "otp", "password", or "discount"), it is immediately assigned to the corresponding category with high confidence and a generated reason.
3.  **Semantic Similarity Fallback (MiniLM):** If the rule-based approach does not yield a clear result, the system leverages a pre-trained SentenceTransformer model (`all-MiniLM-L6-v2`). It converts the message into an embedding and calculates the cosine similarity against predefined category descriptions to find the closest match.

## How Tasks and Events are Extracted

Task and event extraction relies on Intent Detection and Named Entity Recognition (NER):

1.  **Intent Detection:** Similar to classification, the system uses the `all-MiniLM-L6-v2` model to encode the message and compare it against a set of example sentences representing "task", "event", or "none". The closest match determines the intent.
2.  **Information Extraction:** If a task or event is detected, the system extracts relevant details:
    *   **Date & Time:** Uses Regular Expressions (Regex) and the `dateparser` library to parse both absolute dates (e.g., 2024-05-12) and relative dates (e.g., "tomorrow", "next monday") based on the message timestamp.
    *   **Person Name:** Utilizes the `spaCy` library (specifically the `en_core_web_sm` model) for Named Entity Recognition to identify people mentioned in the message.
    *   **Title & Priority:** Employs Regex patterns to infer a concise title or subject, and keyword matching to determine the priority level (e.g., "urgent", "asap" trigger high priority).

## How Sensitive Information is Detected and Masked

The sensitive information detection system uses a dual-layered approach to ensure high accuracy and recall:

1.  **Regex Pattern Matching:** Fast and precise detection for well-defined formats like emails, phone numbers (including Indian formats), PAN numbers, Aadhaar numbers, credit card numbers, OTPs, and generic PINs.
2.  **Zero-Shot NER (GLiNER):** For unstructured or novel sensitive entities, it uses the `GLiNER` (gliner-community/gliner_small-v2.5) model. This allows the system to contextually detect entities like "authentication token", "government identification number", or "private address" even if they don't follow a strict regex pattern.
3.  **Risk Assessment & Action:** Based on the type of sensitive information detected, the system assigns a risk level (`high`, `medium`, `low`) and recommends an action (`do_not_store`, `ask_for_confirmation`, `safe_to_process_locally`).
4.  **Masking:** The detected sensitive spans are sorted and merged to handle overlaps. The sensitive text is then replaced with asterisks (`*`) of the same length in the original message to ensure privacy.

## Assumptions and Limitations

*   **Language:** The models used (`SentenceTransformers`, `spaCy`, `GLiNER`) are primarily optimized for the English language. Non-English messages may not be processed accurately.
*   **Model Size Constraints:** To ensure the pipeline remains lightweight, small models are used (`all-MiniLM-L6-v2`, `en_core_web_sm`, `gliner_small-v2.5`). While efficient, they may occasionally miss highly complex or ambiguous entities compared to larger, more resource-intensive models.
*   **Format Specificity:** The regex patterns for certain sensitive information (like phone numbers, PAN, and Aadhaar) are tailored for the Indian context. International formats might not be caught by the regex rules and would rely solely on the GLiNER model.
*   **Date Parsing Ambiguity:** Relative date parsing depends heavily on the accuracy of the provided message timestamp and the `dateparser` library's interpretation of colloquial time expressions, which can sometimes be ambiguous.

## AI-Tool Usage Declaration

This project utilizes the following open-source AI models and tools:

*   **SentenceTransformers (`all-MiniLM-L6-v2`):** Used for generating sentence embeddings for intent detection and message classification.
*   **spaCy (`en_core_web_sm`):** A small English pipeline trained on web text, used for Named Entity Recognition (specifically, extracting Person names).
*   **GLiNER (`gliner-community/gliner_small-v2.5`):** A Generalist Model for Named Entity Recognition, used here in a zero-shot capacity to detect a wide variety of sensitive entities based on label names.
