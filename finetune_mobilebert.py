import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# 데이터 로드
df = pd.read_csv("labeled_by_keywords5001.csv")  # 파일 경로 조정 가능
df = df.dropna(subset=['text', 'label'])

# 텍스트 및 라벨 준비
texts = df['text'].tolist()
labels = df['label'].astype(int).tolist()

# Tokenizer & Dataset
tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

dataset = Dataset.from_pandas(pd.DataFrame({"text": texts, "label": labels}))
dataset = dataset.train_test_split(test_size=0.2, seed=42)
dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# 모델 로딩
model = MobileBertForSequenceClassification.from_pretrained("google/mobilebert-uncased", num_labels=2)

# 트레이너 설정
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="./logs"
)

# 정확도 평가 지표
def compute_metrics(p):
    preds = torch.argmax(torch.tensor(p.predictions), dim=1)
    labels = torch.tensor(p.label_ids)
    acc = (preds == labels).float().mean()
    return {"accuracy": acc.item()}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model("./mobilebert-finetuned")
tokenizer.save_pretrained("./mobilebert-finetuned")
