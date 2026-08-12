from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


app = Flask(__name__)


# ============================================================
# Translation Model
# ============================================================

MODEL_NAME = "facebook/nllb-200-distilled-600M"

print("Loading translation model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

model.eval()

print("Translation model loaded successfully.")


# ============================================================
# Language Translation
# ============================================================

def translate_text(
    text: str,
    source_language: str,
    target_language: str
) -> str:

    tokenizer.src_lang = source_language

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():

        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(
                target_language
            ),
            max_length=512
        )

    translated_text = tokenizer.batch_decode(
        translated_tokens,
        skip_special_tokens=True
    )[0]

    return translated_text


# ============================================================
# Health Check
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok",
        "service": "Multilingual Translation API",
        "model": MODEL_NAME
    })


# ============================================================
# Translation Endpoint
# ============================================================

@app.route("/api/translate", methods=["POST"])
def translate():

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "error": "Request body is required."
            }), 400


        source_language = data.get("source_language")

        target_language = data.get("target_language")

        text = data.get("text")


        if not source_language:

            return jsonify({
                "error": "source_language is required."
            }), 400


        if not target_language:

            return jsonify({
                "error": "target_language is required."
            }), 400


        if not text or not text.strip():

            return jsonify({
                "error": "text is required."
            }), 400


        text = text.strip()


        translated_text = translate_text(
            text,
            source_language,
            target_language
        )


        return jsonify({

            "source_language": source_language,

            "target_language": target_language,

            "source_text": text,

            "translated_text": translated_text

        })


    except Exception as e:

        print("Translation error:", str(e))

        return jsonify({
            "error": "Translation failed.",
            "details": str(e)
        }), 500


# ============================================================
# Start API
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )