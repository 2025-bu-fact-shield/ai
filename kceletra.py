import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, get_scheduler
from transformers.models.electra.modeling_electra import ElectraForSequenceClassification
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# 설정
MODEL_NAME = "beomi/KcELECTRA-base"
EPOCHS = 4
BATCH_SIZE = 64
LEARNING_RATE = 2e-5
MAX_LEN = 64
WARMUP_STEPS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CSV 로딩 함수 (단일 파일용)
def load_csv_data(csv_path):
    try:
        df = pd.read_csv(
            csv_path,
            usecols=["newsTitle", "clickbaitClass"],
            dtype={"newsTitle": str, "clickbaitClass": int}
        ).dropna()
        return df
    except Exception as e:
        print(f"❌ {csv_path} 로딩 실패: {e}")
        return pd.DataFrame()

# Dataset 클래스
class NewsDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.samples = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        encoding = self.tokenizer(
            row["newsTitle"],
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

# 검증 함수
def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy

# 사용자 정의 모델 클래스
class CustomElectra(ElectraForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.classifier.dropout.p = 0.3

# 메인 실행
def main():
    print(f"\U0001F680 현재 장치: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"\U0001F5A5️ CUDA 디바이스 이름: {torch.cuda.get_device_name(0)}")

    train_path = "/content/drive/MyDrive/training_clickbait.csv"
    val_path = "/content/drive/MyDrive/validation_clickbait.csv"

    train_df = load_csv_data(train_path)
    val_df = load_csv_data(val_path)
    print(f"\U0001F4CA 학습 샘플 수: {len(train_df)}, 검증 샘플 수: {len(val_df)}")

    if len(train_df) == 0 or len(val_df) == 0:
        print("❗ 데이터가 비어 있습니다. 학습을 중단합니다.")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = NewsDataset(train_df, tokenizer)
    val_dataset = NewsDataset(val_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.num_labels = 2
    model = CustomElectra.from_pretrained(MODEL_NAME, config=config)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=EPOCHS * len(train_loader)
    )
    scaler = GradScaler()

    best_accuracy = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        print(f"\n================== Epoch {epoch+1}/{EPOCHS} ==================")
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

        avg_train_loss = total_loss / len(train_loader)
        val_loss, val_accuracy = evaluate(model, val_loader, DEVICE)

        print(f"✅ Epoch {epoch+1} 결과:")
        print(f"   - 평균 학습 손실: {avg_train_loss:.4f}")
        print(f"   - 검증 손실: {val_loss:.4f}")
        print(f"   - 검증 정확도: {val_accuracy:.4f}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), "/content/drive/MyDrive/kcelectra_best_model.pt")
            print(f"📥 모델 저장됨 (정확도 향상): {best_accuracy:.4f}")

    print(f"🏁 최종 최고 검증 정확도: {best_accuracy:.4f}")

if __name__ == "__main__":
    main()
