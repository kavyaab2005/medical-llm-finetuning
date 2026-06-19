"""
Inference script for Medical Q&A Fine-tuned Mistral-7B
Load from Hugging Face Hub and run locally
"""
from unsloth import FastLanguageModel

# ── Config ─────────────────────────────────────────────
HF_REPO     = "kavyaab2005/medical-chatdoctor-mistral7b"
MAX_SEQ_LEN = 1024

# ── Load model ─────────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = HF_REPO,
    max_seq_length = MAX_SEQ_LEN,
    load_in_4bit   = True,
)
FastLanguageModel.for_inference(model)

# ── Inference function ──────────────────────────────────
def ask(question: str) -> str:
    prompt = f"""### Instruction:
You are a helpful and accurate medical assistant. Answer the patient's question clearly and concisely.

### Patient:
{question}

### Doctor:
"""
    inputs  = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.7, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Doctor:")[-1].strip()


# ── Test ────────────────────────────────────────────────
if __name__ == "__main__":
    questions = [
        "I have had a persistent cough for 3 weeks with mild fever. What could this be?",
        "I feel dizzy when I stand up quickly. Is this serious?",
        "I have been having lower back pain for a week. What should I do?",
    ]
    for q in questions:
        print(f"Patient : {q}")
        print(f"Doctor  : {ask(q)}")
        print("-" * 60)
