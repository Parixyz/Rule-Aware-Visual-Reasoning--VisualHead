# 🚦 Visual-NLI Contradiction Detection on CLEVR

This project evaluates **logical consistency in images** by:
- Describing image content using **BLIP** (for VQA),
- Applying **NLI (Natural Language Inference)** models (**T5** and **RoBERTa**) to compare the description (premise) with a set of logical **rules (hypotheses)**,
- Reporting whether the image contradicts any of the domain-specific rules.

---

## 🔍 Use Case

Imagine an AI assistant reviewing scenes to catch visual inconsistencies based on predefined logical constraints, such as:
- _"All cubes must be red"_
- _"There should be exactly one large sphere"_

This pipeline can verify such rules automatically.

---

## 🧠 Models Used

| Component           | Model Name                                                 |
|---------------------|------------------------------------------------------------|
| **VQA / Description** | `Salesforce/blip-vqa-base`                                |
| **NLI (Approach 1)** | `ynie/t5-small-nli`                                       |
| **NLI (Approach 2)** | `ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli`    |

---

## 📂 Directory Structure


---

## 🛠️ Setup

### 1. Clone this repo and install dependencies:

```bash
git clone https://github.com/yourusername/clevr-nli-checker.git
cd clevr-nli-checker
pip install -r requirements.txt
