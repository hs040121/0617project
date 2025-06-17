import pandas as pd
import torch
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 모델 로딩 및 GPU로 이동
model_path = "./mobilebert-finetuned"
tokenizer = MobileBertTokenizer.from_pretrained(model_path)
model = MobileBertForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

# 데이터 로딩
df = pd.read_csv("filtered1_comments21960.csv")
texts = df['text'].fillna("").tolist()

# 배치 단위로 추론 (메모리 절약)
batch_size = 32
predictions = []

for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]
    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().tolist())

# 결과 저장
df['label'] = predictions
df.to_csv("predicted_comments.csv", index=False)

print("✅ GPU 추론 완료. 결과는 'predicted_comments.csv'로 저장됨.")
