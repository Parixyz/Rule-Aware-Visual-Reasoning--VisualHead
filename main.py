import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    T5ForConditionalGeneration,
    T5Tokenizer,
    BlipProcessor,
    BlipForConditionalGeneration,
)
from PIL import Image
import pandas as pd
from tqdm import tqdm

# ------------------- Setup -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BLIP for image description and VQA
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-vqa-base").to(device)

# Load T5 for NLI
T5_MODEL = "ynie/t5-small-nli"
t5_tokenizer = T5Tokenizer.from_pretrained(T5_MODEL)
t5_model = T5ForConditionalGeneration.from_pretrained(T5_MODEL).to(device)

# Load RoBERTa for NLI
ROBERTA_MODEL = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL)
roberta_model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL).to(device)

# ------------------- Functions -------------------

def describe_image_with_blip(image_path, question=None):
    image = Image.open(image_path).convert('RGB')
    if question:
        inputs = blip_processor(image, question, return_tensors="pt").to(device)
    else:
        inputs = blip_processor(image, return_tensors="pt").to(device)
    output = blip_model.generate(**inputs)
    return blip_processor.decode(output[0], skip_special_tokens=True)

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

# ------------------- Rule and Question Setup -------------------

rules = [
    "Objects must not be floating.",
    "There should be exactly one large sphere.",
    "All cubes must be red."
]

# Convert each rule into a question for BLIP
questions = [
    "Are there any objects floating?",
    "How many large spheres are in the image?",
    "What color are the cubes?"
]

# ------------------- Core Evaluation Loop -------------------

image_dir = "CLEVR_v1.0/images/val/"
image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png")])[:10]

results = []

for image_path in tqdm(image_files, desc="Processing Images"):
    try:
        # Gather multiple answers to rule-specific questions to form a better premise
        answers = []
        for q in questions:
            answer = describe_image_with_blip(image_path, question=q)
            answers.append(f"{q} {answer}")
        
        # Combine all question-answer pairs into one premise
        premise = " ".join(answers)

        # Evaluate against each rule (used directly as hypothesis here)
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
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# Convert to DataFrame for review
df = pd.DataFrame(results)
import ace_tools as tools; tools.display_dataframe_to_user(name="CLEVR Inference Results", dataframe=df)

