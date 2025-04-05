🚦 Visual-NLI Contradiction Detection on CLEVR
This project evaluates logical consistency in images by:

Describing image content using BLIP (for VQA),

Applying NLI (Natural Language Inference) models (T5 and RoBERTa) to compare the description (premise) with a set of logical rules (hypotheses),

Reporting whether the image contradicts any of the domain-specific rules.

🔍 Use Case
Imagine an AI assistant reviewing scenes to catch visual inconsistencies based on predefined logical constraints, such as:

"All cubes must be red"

"There should be exactly one large sphere"

This pipeline can verify such rules automatically.

🧠 Models Used
Component	Model Name
VQA / Description	Salesforce/blip-vqa-base
NLI (Approach 1)	ynie/t5-small-nli
NLI (Approach 2)	ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli
📂 Directory Structure
bash
Copy
Edit
project_root/
├── main.py                # Core script to evaluate contradictions
├── CLEVR_v1.0/            # CLEVR image dataset directory
│   └── images/val/        # Folder with .png images
├── requirements.txt       # Python dependencies
└── README.md              # You're here!
🛠️ Setup
1. Clone this repo and install dependencies:
bash
Copy
Edit
git clone https://github.com/yourusername/clevr-nli-checker.git
cd clevr-nli-checker
pip install -r requirements.txt
2. Download CLEVR images
You can download the CLEVR_v1.0 dataset from Stanford's official site and place the validation images in:

swift
Copy
Edit
CLEVR_v1.0/images/val/
🚀 Run the Script
bash
Copy
Edit
python main.py
You’ll get a DataFrame output showing:

Image name

Premise (based on VQA)

Rule (hypothesis)

Classification results from T5 and RoBERTa

📊 Example Output
image	premise	rule	t5_result	roberta_result
CLEVR_001.png	Are there any objects floating? No. ...	All cubes must be red.	entailment	neutral
CLEVR_002.png	How many large spheres? Two. ...	There should be exactly one ...	contradiction	contradiction
📌 Customization
Add your own rules in the rules list.

Add more questions to extract better premises from BLIP.

Extend to other datasets (e.g., real-world scenes) by adjusting rules/questions accordingly.

