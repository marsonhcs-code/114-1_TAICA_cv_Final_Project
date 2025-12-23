import pandas as pd
import os
from collections import defaultdict


class DatasetParser:
    def __init__(self, target_sequences=['Water', 'Fat', 'T1', 'T2']):
        self.target_sequences = target_sequences

    def parse_csv(self, csv_path):
        df = pd.read_csv(csv_path, sep=',') 
        parsed_data = defaultdict(dict)
        warnings = []

        for index, row in df.iterrows():
            subject_id = row['MRI_sample']
            sequence = row['MRI_Sequence']
            label_path = row['detailed_label_3D_file']
            img_path = row['image_3D_file']
            
            
            # 過濾不需要的序列 (如 STIR)
            if sequence not in self.target_sequences:
                continue
            
            # 簡單的重複檢查
            if sequence in parsed_data[subject_id]:
                warnings.append(f"[Duplicate] {subject_id} - {sequence}")
                continue

            parsed_data[subject_id][sequence]["label"] = label_path
            parsed_data[subject_id][sequence]["image"] = img_path
            
        print(f"Parsed {len(parsed_data)} subjects from CSV.")

        return parsed_data, warnings

    def validate_completeness(self, parsed_data):
        valid_subjects = {}
        incomplete_subjects = []

        for sub_id, sequences in parsed_data.items():
            missing = [seq for seq in self.target_sequences if seq not in sequences]
            if not missing:
                valid_subjects[sub_id] = sequences # 只保留完整的病人資料
            else:
                incomplete_subjects.append((sub_id, missing))
        
        return valid_subjects, incomplete_subjects

# --- 使用範例 ---
if __name__ == "__main__":
    # 讀取檔案
    csv_file = "/home/n26141826/114-1_TAICA_cv_Final_Project/data/metadata_3D.csv"  # <--- 請改成你的真實檔案路徑"

    # 1. 初始化解析器
    parser = DatasetParser(target_sequences=['Water', 'Fat', 'T1', 'T2'])
    
    # 2. 執行解析
    data, warns = parser.parse_csv(csv_file)

    # 3. 顯示警告 (重複檔案)
    print("--- Warnings ---")
    for w in warns:
        print(w)

    # 4. 驗證完整性
    valid_subs, incomplete_subs = parser.validate_completeness(data)

    print("\n--- Validation Report ---")
    print(f"Valid Subjects (Ready for training): {len(valid_subs)}")
    print(f"Incomplete Subjects: {len(incomplete_subs)}")
    
    for sub, missing in incomplete_subs:
        print(f"  {sub} is missing: {missing}")

    # 5. 查看結果結構 (以 THIGH_017 為例)
    if 'THIGH_017' in data:
        print("\n--- Sample Structure (THIGH_017) ---")
        print(data['THIGH_017'])