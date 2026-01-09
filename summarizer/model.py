from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# -----------------------------
# Model & tokenizer
# -----------------------------
MODEL_NAME = "google/flan-t5-small"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# -----------------------------
# Style configuration
# -----------------------------
STYLE_CONFIG = {
    "short": {
        "prompt": "Summarize this in 1-2 concise sentences: ",
        "max_length": 50,
        "min_length": 20,
        "length_penalty": 1.8
    },
    "medium": {
        "prompt": "Summarize the following text in detail: ",
        "max_length": 120,
        "min_length": 60,
        "length_penalty": 1.0
    }
}

# -----------------------------
# Text chunking (for long input)
# -----------------------------
def split_text(text, max_words=350):
    words = text.split()
    return [
        " ".join(words[i:i + max_words])
        for i in range(0, len(words), max_words)
    ]

# -----------------------------
# Core summarization (NO chunking)
# -----------------------------
def summarize_batch(
    texts,
    style="medium",
    temperature=0.7,
    num_beams=4
):
    if style not in STYLE_CONFIG:
        raise ValueError("style must be 'short' or 'medium'")

    config = STYLE_CONFIG[style]

    prompts = [config["prompt"] + t for t in texts]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=config["max_length"],
            min_length=config["min_length"],
            num_beams=num_beams,
            length_penalty=config["length_penalty"],
            early_stopping=True
        )

    return [
        tokenizer.decode(o, skip_special_tokens=True)
        for o in outputs
    ]

# -----------------------------
# Long-text summarization (WITH chunking)
# -----------------------------
def summarize_long_text(
    text,
    style="medium",
    temperature=0.7,
    num_beams=4
):
    chunks = split_text(text)

    # 1 chunk → normal summary
    if len(chunks) == 1:
        return summarize_batch(
            [chunks[0]],
            style=style,
            temperature=temperature,
            num_beams=num_beams
        )[0]

    # Summarize each chunk
    chunk_summaries = summarize_batch(
        chunks,
        style=style,
        temperature=temperature,
        num_beams=num_beams
    )

    # Meta-summary
    combined = " ".join(chunk_summaries)

    final_summary = summarize_batch(
        [combined],
        style=style,
        temperature=temperature,
        num_beams=num_beams
    )[0]

    return final_summary
