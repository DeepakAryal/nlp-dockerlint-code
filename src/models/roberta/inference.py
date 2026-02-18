import os
import time
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from joblib import load

MODEL_TYPE = "roberta"

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

STAGE1_MODEL_DIR = f"saved_models/stage1_{MODEL_TYPE}"
STAGE1_CHECKPOINT_DIR = os.path.join(STAGE1_MODEL_DIR, "last_checkpoint")
STAGE1_MODEL_PATH = os.path.join(STAGE1_CHECKPOINT_DIR, "pytorch_model.bin")

STAGE2_MODEL_DIR = f"saved_models/stage2_{MODEL_TYPE}"
STAGE2_ENCODER_PATH = f"data/{MODEL_TYPE}/tokenized_chunks-stage2/label_encoder.joblib"

BASE_MODEL_NAMES = {
    "codebert": "microsoft/codebert-base",
    "roberta": "roberta-base"
}

MAX_LEN = 256

print(f"[INFO] Using device: {DEVICE}")
print(f"[INFO] Model type: {MODEL_TYPE.upper()}")


print("\n[INFO] Loading Stage 1 model (binary classification)...")
if not os.path.exists(STAGE1_CHECKPOINT_DIR):
    raise FileNotFoundError(
        f"[ERROR] Stage 1 checkpoint directory not found: {STAGE1_CHECKPOINT_DIR}\n"
        f"Please make sure you have trained and saved the Stage 1 model."
    )

if not os.path.exists(STAGE1_MODEL_PATH):
    raise FileNotFoundError(
        f"[ERROR] Stage 1 model file not found: {STAGE1_MODEL_PATH}\n"
        f"Please make sure you have trained and saved the Stage 1 model."
    )

base_model_name = BASE_MODEL_NAMES[MODEL_TYPE]
stage1_model = AutoModelForSequenceClassification.from_pretrained(
    base_model_name, 
    num_labels=2
)
stage1_model.load_state_dict(torch.load(STAGE1_MODEL_PATH, map_location=DEVICE, weights_only=False))
stage1_model.to(DEVICE)
stage1_model.eval()

stage1_tokenizer = AutoTokenizer.from_pretrained(STAGE1_CHECKPOINT_DIR, local_files_only=True)
print(f"[INFO] Stage 1 model loaded from: {STAGE1_CHECKPOINT_DIR}")

print("\n[INFO] Loading Stage 2 model (rule classification)...")
if not os.path.exists(STAGE2_MODEL_DIR):
    raise FileNotFoundError(
        f"[ERROR] Stage 2 model directory not found: {STAGE2_MODEL_DIR}\n"
        f"Please make sure you have trained and saved the Stage 2 model."
    )

stage2_tokenizer = AutoTokenizer.from_pretrained(STAGE2_MODEL_DIR, local_files_only=True)
stage2_model = AutoModelForSequenceClassification.from_pretrained(STAGE2_MODEL_DIR, local_files_only=True)
stage2_model.to(DEVICE)
stage2_model.eval()
print(f"[INFO] Stage 2 model loaded: {STAGE2_MODEL_DIR}")

if os.path.exists(STAGE2_ENCODER_PATH):
    print(f"[INFO] Loading label encoder from {STAGE2_ENCODER_PATH}")
    label_encoder = load(STAGE2_ENCODER_PATH)
    print(f"[INFO] Label encoder loaded. Number of rules: {len(label_encoder.classes_)}")
else:
    raise FileNotFoundError(f"[ERROR] Label encoder not found at {STAGE2_ENCODER_PATH}")


def predict_stage1(text):
    """
    Stage 1: Predict if Dockerfile line is correct (1) or wrong (0).
    Returns: (prediction, confidence, inference_time)
    """
    start_time = time.time()
    
    encodings = stage1_tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    
    input_ids = encodings["input_ids"].to(DEVICE)
    attention_mask = encodings["attention_mask"].to(DEVICE)
    
    with torch.no_grad():
        outputs = stage1_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
    
    pred = torch.argmax(logits, dim=1).item()
    confidence = probs[0][pred].item()
    
    inference_time = time.time() - start_time
    
    return pred, confidence, inference_time


def predict_stage2(text):
    """
    Stage 2: Predict which rule is violated (only called if Stage 1 predicts wrong).
    Returns: (rule_name, confidence, inference_time)
    """
    start_time = time.time()
    
    encodings = stage2_tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    
    input_ids = encodings["input_ids"].to(DEVICE)
    attention_mask = encodings["attention_mask"].to(DEVICE)
    
    with torch.no_grad():
        outputs = stage2_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
    
    pred_class_id = torch.argmax(logits, dim=1).item()
    confidence = probs[0][pred_class_id].item()
    
    rule_name = label_encoder.inverse_transform([pred_class_id])[0]
    
    inference_time = time.time() - start_time
    
    return rule_name, confidence, inference_time


def infer(dockerfile_line):
    """
    Complete inference pipeline:
    1. Stage 1: Check if configuration is correct
    2. If wrong, Stage 2: Identify which rule is violated
    """
    print("\n" + "="*80)
    print("DOCKERFILE LINE ANALYSIS")
    print("="*80)
    print(f"Input: {dockerfile_line}")
    print("-"*80)
    
    print("\n[STAGE 1] Checking if configuration is correct...")
    stage1_pred, stage1_conf, stage1_time = predict_stage1(dockerfile_line)
    
    if stage1_pred == 1:
        result = "CORRECT"
        status_emoji = "✅"
    else:
        result = "WRONG"
        status_emoji = "❌"
    
    print(f"  Prediction: {status_emoji} {result}")
    print(f"  Confidence: {stage1_conf:.4f} ({stage1_conf*100:.2f}%)")
    print(f"  Inference time: {stage1_time*1000:.2f} ms")
    
    if stage1_pred == 0:
        print("\n[STAGE 2] Identifying violated rule...")
        rule_name, stage2_conf, stage2_time = predict_stage2(dockerfile_line)
        
        print(f"  Violated Rule: {rule_name}")
        print(f"  Confidence: {stage2_conf:.4f} ({stage2_conf*100:.2f}%)")
        print(f"  Inference time: {stage2_time*1000:.2f} ms")
        
        total_time = stage1_time + stage2_time
        print(f"\n  Total inference time: {total_time*1000:.2f} ms")
        
        return {
            "status": "WRONG",
            "rule": rule_name,
            "stage1_confidence": stage1_conf,
            "stage2_confidence": stage2_conf,
            "stage1_time_ms": stage1_time * 1000,
            "stage2_time_ms": stage2_time * 1000,
            "total_time_ms": total_time * 1000
        }
    else:
        print("\n[STAGE 2] Skipped (configuration is correct)")
        return {
            "status": "CORRECT",
            "rule": None,
            "stage1_confidence": stage1_conf,
            "stage2_confidence": None,
            "stage1_time_ms": stage1_time * 1000,
            "stage2_time_ms": 0,
            "total_time_ms": stage1_time * 1000
        }


if __name__ == "__main__":
    examples = [
        "RUN apt-get update && apt-get install -y python3",
    ]
    
    print("\n" + "="*80)
    print("RUNNING INFERENCE EXAMPLES ROBERTA")
    print("="*80)
    
    for example in examples:
        result = infer(example)
        print("\n" + "-"*80)
    
    print("\n" + "="*80)
    print("INTERACTIVE MODE")
    print("="*80)
    print("Enter Dockerfile lines to analyze (type 'quit' to exit):\n")
    
    while True:
        user_input = input("Dockerfile line: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n[INFO] Exiting...")
            break
        if user_input:
            infer(user_input)
            print()
