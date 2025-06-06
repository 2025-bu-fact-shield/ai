import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# ✅ 설정
MODEL_NAME = "beomi/KcELECTRA-base"
EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 2e-5
MAX_LEN = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Dataset 정의
class NewsDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.samples = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        encoding = self.tokenizer(
            row["newTitle"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(row["clickbaitClass"])
        }

# ✅ CSV 로딩 최적화
def load_csv_data(csv_folder):
    all_data = []
    for fname in os.listdir(csv_folder):
        if fname.endswith(".csv"):
            path = os.path.join(csv_folder, fname)
            try:
                df = pd.read_csv(
                    path,
                    usecols=["newTitle", "clickbaitClass"],
                    dtype={"newTitle": str, "clickbaitClass": int}
                ).dropna()
                all_data.append(df)
            except Exception as e:
                print(f"❌ {fname} 로딩 실패: {e}")
    return pd.concat(all_data, ignore_index=True)

# ✅ 메인
def main():
    print(f"🚀 현재 장치: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"🖥️ CUDA 디바이스 이름: {torch.cuda.get_device_name(0)}")

    # ✅ 데이터 로딩
    train_df = load_csv_data("D:/MyProjects/Fact-Shield/dev_backend/DataSets/Training")
    val_df = load_csv_data("D:/MyProjects/Fact-Shield/dev_backend/DataSets/Validation")
    print(f"📊 학습 샘플 수: {len(train_df)}, 검증 샘플 수: {len(val_df)}")

    # ✅ 토크나이저 및 데이터셋
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = NewsDataset(train_df, tokenizer)
    val_dataset = NewsDataset(val_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # ✅ 모델 구성
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0,
        num_training_steps=EPOCHS * len(train_loader)
    )
    scaler = GradScaler()

    # ✅ 학습 루프
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        print(f"\n================== Epoch {epoch+1}/{EPOCHS} 시작 ==================")
        for batch in tqdm(train_loader, desc=f"[{epoch+1}에폭] 학습 중"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} 완료 - 평균 손실: {avg_loss:.4f}")

    # ✅ 저장
    save_path = "./kcelectra_model_from_csv.pt"
    torch.save(model.state_dict(), save_path)
    print(f"🎉 전체 학습 완료! 모델 저장됨: {save_path}")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
# This code is designed to train a KcELECTRA model using a CSV dataset.
# It includes optimizations for data loading, model training, and GPU utilization.
# The model is saved after training for later use.
# The code uses PyTorch and Hugging Face Transformers library for model handling.