import pandas as pd
import matplotlib.pyplot as plt

# 추론 결과 파일 로딩
df = pd.read_csv("merged_labeled_data.csv")  # 또는 해당 경로

label_counts = df['label'].value_counts().sort_index()
label_names = ['Negative', 'Positive']

plt.figure(figsize=(6, 4))
plt.bar(label_names, label_counts, color=['red', 'green'])
plt.title("Label Distribution (Negative vs Positive)")
plt.xlabel("Label")
plt.ylabel("Count")
plt.grid(axis='y')
plt.savefig("label_distribution.png")
plt.show()

print("📊 라벨 분포:")
print(label_counts)
