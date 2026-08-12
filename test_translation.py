from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_NAME = "facebook/nllb-200-distilled-600M"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print("Model loaded successfully.")


def translate(text, source_language, target_language):

    tokenizer.src_lang = source_language

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    with torch.no_grad():

        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(
                target_language
            ),
            max_length=256
        )

    result = tokenizer.batch_decode(
        translated_tokens,
        skip_special_tokens=True
    )

    return result[0]


text = "Hello, how are you today?"

translation = translate(
    text,
    "eng_Latn",
    "zul_Latn"
)

print()
print("English:")
print(text)

print()
print("isiZulu:")
print(translation)