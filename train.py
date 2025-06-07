import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, get_scheduler
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt

# 설정
MODEL_NAME = "beomi/KcELECTRA-base-v2022"  # 🔁 여기서 Electra 또는 KoBERT 교체 가능
EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
MAX_LEN = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_losses = []
val_losses = []
val_accuracies = []
# 에폭별 기록 저장을 위한 리스트

# CSV 로딩 함수
def load_csv_data(csv_folder):
    all_data = []
    for fname in os.listdir(csv_folder):
        if fname.endswith(".csv"):
            path = os.path.join(csv_folder, fname)
            try:
                df = pd.read_csv(
                    path,
                    usecols=["newsTitle", "clickbaitClass"],
                    dtype={"newsTitle": str, "clickbaitClass": int}
                ).dropna()
                all_data.append(df)
            except Exception as e:
                print(f"❌ {fname} 로딩 실패: {e}")
    return pd.concat(all_data, ignore_index=True)

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

# 메인 실행
def main():
    print(f"🚀 현재 장치: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"🖥️ CUDA 디바이스 이름: {torch.cuda.get_device_name(0)}")

    train_path = "./DataSets/training"
    val_path = "./DataSets/validation"

    train_df = load_csv_data(train_path)
    # 🔻 학습 데이터 40%
    train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)  
    val_df = load_csv_data(val_path)
    # 🔻 검증 데이터 40%
    val_df = val_df.sample(frac=1.0, random_state=42).reset_index(drop=True)  

    print(f"📊 학습 샘플 수: {len(train_df)}, 검증 샘플 수: {len(val_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = NewsDataset(train_df, tokenizer)
    val_dataset = NewsDataset(val_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.num_labels = 2
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=2)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0,
        num_training_steps=EPOCHS * len(train_loader)
    )
    scaler = GradScaler()

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

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"✅ Epoch {epoch+1} 결과:")
        print(f"   - 평균 학습 손실: {avg_train_loss:.4f}")
        print(f"   - 검증 손실: {val_loss:.4f}")
        print(f"   - 검증 정확도: {val_accuracy:.4f}")

    save_path = "./kcelectra_model_dropout1.0.pt"  # 또는 절대 경로 지정
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 경로 보장
    torch.save(model.state_dict(), save_path)
    print(f"🎉 전체 학습 완료! 모델 저장됨: {save_path}")

    epochs = range(1, EPOCHS + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    plt.plot(epochs, val_losses, marker='s', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch-wise Loss')
    plt.legend()

    # 검증 정확도
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, marker='^', color='green', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Epoch-wise Validation Accuracy')
    plt.ylim(0, 1)
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_results1.0.png")
    plt.show()

    print("📈 학습 결과 시각화 완료! 'training_results1.0.png'로 저장됨.")

if __name__ == "__main__":
    main()
