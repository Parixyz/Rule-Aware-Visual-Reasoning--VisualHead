# 🚦 Visual-Models Trained on 3DNLI

This project evaluates **logical consistency in images** by:
- Describing image content using **BLIP** (for VQA),
- Applying **NLI (Natural Language Inference)** models (**T5** and **RoBERTa**) to compare the description (premise) with a set of logical **rules (hypotheses)**,
- Reporting whether the image contradicts any of the domain-specific rules.

---
![scene_00003_angle_0_scene_summary](https://github.com/user-attachments/assets/76c03588-46d7-4ed2-88b2-e0ba7b8e2d9e)

## 🔍 Use Case

Imagine an AI assistant reviewing scenes to catch visual inconsistencies based on predefined logical constraints, such as:
- _"All cubes must be red"_
- _"There should be exactly one large sphere"_

This pipeline can verify such rules automatically.

---


