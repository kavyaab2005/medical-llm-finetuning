# 🩺 Medical LLM Fine-Tuning — Mistral-7B + QLoRA

Fine-tuned **Mistral-7B** on a Medical Q&A dataset using **QLoRA** 
(4-bit quantization) and **Unsloth** for 2x faster training — 
entirely on a **free T4 GPU** via Google Colab.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)
![LoRA](https://img.shields.io/badge/QLoRA-4bit-green)
![GPU](https://img.shields.io/badge/GPU-T4%20Free-orange)

---

## 🔗 Links
- 🤗 **Model on HF Hub**: [your-username/medical-chatdoctor-mistral7b](https://huggingface.co/kavyaab2005/medical-chatdoctor-mistral7b)
- 📓 **Colab Notebook**: [Open in Colab](https://colab.research.google.com/github/kavyaab2005/medical-llm-finetuning/blob/main/notebook/medical_llm_finetuning.ipynb)

---

## 🧠 What This Project Does

Demonstrates full LLM fine-tuning pipeline:

```
Dataset → Base Model → QLoRA Adapters → Training → Deploy
```

Instead of calling an API, this project **trains a model** — showing 
practical knowledge of how LLMs actually learn.

---

## ⚙️ Tech Stack

| Component | Tool |
|---|---|
| Base Model | Mistral-7B-Instruct-v0.2 |
| Fine-tuning Method | QLoRA (4-bit) + LoRA rank-8 |
| Training Framework | Unsloth + HuggingFace TRL |
| Dataset | ChatDoctor-HealthCareMagic-100k |
| Hardware | Google Colab T4 GPU (free) |
| Deployment | Hugging Face Hub + Gradio |

---

## 📊 Training Details

| Parameter | Value |
|---|---|
| Training samples | 2,000 |
| Epochs | 1 |
| Batch size (effective) | 8 |
| Learning rate | 2e-4 |
| Trainable parameters | ~42M of 7.2B (0.58%) |
| Training time | ~45 mins on free T4 |

---

## 💡 Key Concepts Demonstrated

- **QLoRA**: 4-bit quantization of frozen base model weights to fit 7B 
  model in 15GB VRAM
- **LoRA**: Low-rank adapter matrices trained instead of full weights — 
  99.4% of parameters stay frozen
- **Instruction tuning**: Custom prompt template teaching the model 
  input/output structure
- **SFTTrainer**: Supervised fine-tuning with gradient accumulation

---

## 🚀 Run Inference

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name   = "your-username/medical-chatdoctor-mistral7b",
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)
```

Or clone and run locally:

```bash
git clone https://github.com/your-username/medical-llm-finetuning
cd medical-llm-finetuning
pip install -r requirements.txt
python src/inference.py
```

---

## ⚠️ Disclaimer

For educational purposes only. Not a substitute for professional 
medical advice.
