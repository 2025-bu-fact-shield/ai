import os
import json
import csv
import re
from tqdm import tqdm

# 텍스트 전처리 함수
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = text.replace('\\"', '"')
    text = text.replace("...", "…")
    text = re.sub(r'\[[^\]]+\]', '', text)
    text = re.sub(r'[“”‘’]', '"', text)
    text = re.sub(r"[^가-힣a-zA-Z0-9\s.,!?\"']", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 폴더 내 JSON 파일 처리 함수
def convert_folder_json_to_csv(json_folder_path, csv_output_path):
    rows = []

    for fname in tqdm(os.listdir(json_folder_path), desc=f"📂 {os.path.basename(json_folder_path)} 처리 중"):
        if not fname.endswith(".json"):
            continue

        file_path = os.path.join(json_folder_path, fname)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            source = data.get("sourceDataInfo", {})
            labeled = data.get("labeledDataInfo", {})
            refer_map = {int(d["sentenceNo"]): d["referSentenceyn"] for d in labeled.get("referSentenceInfo", [])}

            for s in source.get("sentenceInfo", []):
                rows.append({
                    "newsID": source.get("newsID"),
                    "newsCategory": source.get("newsCategory"),
                    "newsSubcategory": source.get("newsSubcategory"),
                    "newsTitle": clean_text(source.get("newsTitle")),
                    "newsSubTitle": clean_text(source.get("newsSubTitle")),
                    "processLevel": source.get("processLevel"),
                    "sentenceNo": s.get("sentenceNo"),
                    "sentenceContent": clean_text(s.get("sentenceContent")),
                    "newTitle": clean_text(labeled.get("newTitle")),
                    "clickbaitClass": labeled.get("clickbaitClass"),
                    "referSentenceyn": refer_map.get(s.get("sentenceNo"), "N")
                })

        except Exception as e:
            print(f"❌ 오류 발생: {fname} - {e}")
            continue

    # CSV 저장
    with open(csv_output_path, "w", newline="", encoding="utf-8-sig") as csvfile:
        fieldnames = [
            "newsID", "newsCategory", "newsSubcategory", "newsTitle", "newsSubTitle",
            "processLevel", "sentenceNo", "sentenceContent",
            "newTitle", "clickbaitClass", "referSentenceyn"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ 저장 완료: {csv_output_path} (총 {len(rows)}개 문장)")

# 상위 Training 폴더 내 모든 하위 폴더 처리
def convert_all_subfolders(training_root):
    for folder in os.listdir(training_root):
        sub_path = os.path.join(training_root, folder)
        if not os.path.isdir(sub_path):
            continue

        csv_name = f"{folder}.csv"
        csv_path = os.path.join(training_root, csv_name)
        convert_folder_json_to_csv(sub_path, csv_path)

# 사용 예시
convert_all_subfolders("D:\MyProjects\Fact-Shield\dev_backend\DataSets\Validation")
