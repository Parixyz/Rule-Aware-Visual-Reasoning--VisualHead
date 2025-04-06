import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    T5ForConditionalGeneration,
    T5Tokenizer,
    Blip2Processor,
    Blip2ForConditionalGeneration
)
import matplotlib.pyplot as plt

# ------------------- Setup -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load BLIP-2 model and processor
blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip2_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16
).to(device)

# Load T5 model (replaced with a SQuAD fine-tuned version)
T5_MODEL = "mrm8488/t5-base-finetuned-squadv2"
t5_tokenizer = T5Tokenizer.from_pretrained(T5_MODEL)
t5_model = T5ForConditionalGeneration.from_pretrained(T5_MODEL).to(device)

# Load RoBERTa for NLI
ROBERTA_MODEL = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL)
roberta_model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL).to(device)

# ------------------- Functions -------------------

def ask_blip_question(image_path, question):
    image = Image.open(image_path).convert("RGB")
    prompt = f"Question: {question} Answer:"
    inputs = blip2_processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
    ids = blip2_model.generate(**inputs)
    return blip2_processor.tokenizer.batch_decode(ids, skip_special_tokens=True)[0]

def classify_nli_t5(premise, hypothesis):
    input_text = f"premise: {premise} hypothesis: {hypothesis}"
    inputs = t5_tokenizer(input_text, return_tensors="pt", padding=True).to(device)
    outputs = t5_model.generate(**inputs)
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

def classify_nli_roberta(premise, hypothesis):
    inputs = roberta_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = roberta_model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    labels = ['entailment', 'neutral', 'contradiction']
    return labels[prediction]

def display_image_with_answers(image_path, qa_pairs):
    image = Image.open(image_path)
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.axis("off")
    text = "\n".join([f"{q} → {a}" for q, a in qa_pairs])
    plt.figtext(0.5, 0.01, text, wrap=True, horizontalalignment='center', fontsize=10)
    plt.tight_layout()
    plt.show()

# ------------------- Rule & Question Setup -------------------

rules = [
    "Objects must not be floating.",
    "There should be exactly one large sphere.",
    "All cubes must be red."
]

questions = [
    "Are there any objects floating?",
    "How many large spheres are in the image?",
    "What color are the cubes?"
]

# ------------------- Main Execution -------------------

image_dir = "CLEVR_v1.0/images/val/"
image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png")])[:1]

results = []

for image_path in tqdm(image_files, desc="Processing Images"):
    try:
        qa_pairs = [(q, ask_blip_question(image_path, q)) for q in questions]
        premise = " ".join([f"{q} {a}" for q, a in qa_pairs])

        for rule in rules:
            t5_result = classify_nli_t5(premise, rule)
            roberta_result = classify_nli_roberta(premise, rule)

            results.append({
                "image": os.path.basename(image_path),
                "premise": premise,
                "rule": rule,
                "t5_result": t5_result,
                "roberta_result": roberta_result
            })

        display_image_with_answers(image_path, qa_pairs)

    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# ------------------- Save or Display Results -------------------

df = pd.DataFrame(results)
