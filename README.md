# 114-1_TAICA_cv_Final_Project
Thigh muscle segmentation


# 大腿肌肉分割專題 - 完整實作指南

## 📋 目錄
1. [環境設定](#環境設定)
2. [資料準備](#資料準備)
3. [模型訓練](#模型訓練)
4. [評估與分析](#評估與分析)
5. [時程規劃](#時程規劃)
6. [期末報告結構](#期末報告結構)

---

## 環境設定

### 1. 安裝基礎套件

```bash
# 建立 venv 環境
    curl -fsSL https://pyenv.run | bash
    export PATH="$HOME/.pyenv/bin:$PATH"
    pyenv install 3.12.7
    pyenv local 3.12.7
    python -m venv .venv
    source .venv/bin/activate

# 安裝PyTorch (根據你的CUDA版本)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安裝nnU-Net V2 、其他必要套件
pip install -r requirements.txt
```

### 2. 設定nnU-Net環境變量

```bash
# 在 ~/.bashrc 或 ~/.zshrc 中加入：
export nnUNet_raw="/home/n26141826/114-1_TAICA_cv_Final_Project/data/nnUNet_raw"
export nnUNet_preprocessed="/home/n26141826/114-1_TAICA_cv_Final_Project/data/nnUNet_preprocessed"
export nnUNet_results="/home/n26141826/114-1_TAICA_cv_Final_Project/data/nnUNet_results"

# 重新載入
source ~/.bashrc
```

### 3. 建立資料夾結構

```bash
mkdir -p ./data/nnUNet_raw
mkdir -p ./data/nnUNet_preprocessed
mkdir -p ./data/nnUNet_results
```

---

## 資料準備

### Step 1: 執行資料預處理腳本

```bash
cd .
# 使用之前提供的 data_prep_nnunet.py
python data_prep_nnunet.py

# 或是使用互動式方式
python -c "
from data_prep_nnunet import *

SOURCE_DIR = './data'  # 你的data資料夾
TARGET_DIR = './data/nnUNet_raw'

prep = ThighMuscleDataPreparation(SOURCE_DIR, TARGET_DIR)
prep.create_directory_structure()
prep.convert_to_nnunet_format(use_detailed_label=True)
prep.create_dataset_json(use_detailed_label=True)
prep.split_train_validation(val_ratio=0.2)
prep.verify_dataset()
"
```

### Step 2: 驗證資料

```bash
# nnU-Net內建的驗證工具
nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity
```

---

## 模型訓練

### 快速實驗 (適合11/20進度報告)

```bash
# 使用提供的training_script.py
python training_script.py --mode quick --dataset_id 001
```

這會：
1. 自動預處理資料
2. 訓練一個fold的3D Low Resolution U-Net (M3，論文最佳模型)
3. 大約需要4-8小時（取決於GPU）

### 完整訓練 (期末報告用)

```bash
# 訓練所有配置
python training_script.py --mode full --dataset_id 001
```

這會：
1. 訓練5-fold cross validation的3D Low Res
2. 訓練3D Full Res (M2)
3. 訓練2D (M1)
4. 自動找出最佳配置
5. 需要數天時間

### 手動訓練特定配置

```bash
# 訓練M3 (3D Low Resolution) - 論文表現最好
nnUNetv2_train 001 3d_lowres 0 --npz

# 訓練所有5 folds
for fold in {0..4}; do
    nnUNetv2_train 001 3d_lowres $fold --npz
done

# 訓練M2 (3D Full Resolution)
nnUNetv2_train 001 3d_fullres 0 --npz

# 訓練M1 (2D)
nnUNetv2_train 001 2d 0 --npz
```

### 預測

```bash
# 使用訓練好的模型預測
python training_script.py --mode predict --dataset_id 001

# 或手動執行
nnUNetv2_predict \
    -i $nnUNet_raw/Dataset001_ThighMuscle/imagesTs \
    -o ./predictions \
    -d 001 \
    -c 3d_lowres \
    -f all
```

---

## 評估與分析

### 執行完整評估

```bash
# 使用提供的evaluation_script.py
python evaluation_script.py
```

這會計算：
- DSC (Dice Similarity Coefficient)
- IOU (Intersection over Union)
- NSD (Normalized Surface Dice)
- HD95 (Hausdorff Distance 95)
- 性別差異分析

### 查看訓練過程

```bash
# nnU-Net會自動生成訓練曲線
# 結果位於：
cd $nnUNet_results/Dataset001_ThighMuscle/nnUNetTrainer__nnUNetPlans__3d_lowres/fold_0

# 查看validation結果
cat validation_results.json
```

---

## 時程規劃

### 第一階段：11/10 - 11/20 (進度報告準備)

**目標：完成初步實驗並準備進度報告**

#### Week 1 (11/10-11/13)
- [x] 資料整理完成（已完成）
- [ ] 環境設定與安裝套件
- [ ] 執行資料預處理腳本
- [ ] 驗證資料格式正確

#### Week 2 (11/13-11/20)
- [ ] 訓練quick experiment (單fold)
- [ ] 在測試集上預測
- [ ] 計算基本評估指標 (DSC, IOU)
- [ ] 準備進度報告投影片

**11/20進度報告內容：**
```
1. 專題介紹
   - 大腿肌肉分割的臨床意義
   - 資料集來源與特性
   - 11類肌肉的解剖位置

2. 方法介紹
   - nnU-Net架構說明
   - 論文中的最佳配置 (M3)
   - 訓練參數設定

3. 初步結果
   - 單fold的DSC結果
   - 視覺化分割結果
   - 與論文結果初步比較

4. 後續規劃
   - 完整5-fold訓練
   - 嘗試ensemble方法
   - 疾病分類系統開發
```

### 第二階段：11/20 - 12/10 (完整實驗)

#### Week 3 (11/20-11/27)
- [ ] 訓練完整5-fold cross validation
- [ ] 實作ensemble方法
- [ ] 性別差異分析

#### Week 4 (11/27-12/04)
- [ ] 測試不同配置 (3D fullres, 2D)
- [ ] 比較CNN vs Transformer (如果有時間)
- [ ] 開發肌肉評分系統

#### Week 5 (12/04-12/10)
- [ ] 完整評估與統計分析
- [ ] 製作所有視覺化圖表
- [ ] 撰寫期末報告

### 第三階段：12/10 - 期末 (報告完善)

- [ ] 準備期末報告投影片
- [ ] 錄製demo影片
- [ ] 整理程式碼與文件
- [ ] (Optional) 準備論文投稿

---

## 期末報告結構

### 1. 專題介紹 (5-7分鐘)

```
內容重點：
✓ 臨床背景與動機
  - 肌肉疾病的影像特徵
  - T1油花 vs T2水腫
  - 為什麼需要自動分割

✓ 資料集介紹
  - 三個來源的MRI資料
  - 影像modalities (T1, T2, STIR, FAT, WATER)
  - 11類肌肉標籤

✓ 挑戰與目標
  - 多類別小目標分割困難
  - 需要高精度支援臨床應用
```

### 2. 模型介紹 (8-10分鐘)

```
內容重點：
✓ 為什麼選擇nnU-Net
  - 論文證實的效果
  - 自動化配置優勢
  - 醫學影像分割的標準baseline

✓ 架構細節
  - U-Net encoder-decoder結構
  - 3D vs 2D的差異
  - Low resolution vs Full resolution

✓ 訓練策略
  - 論文的最佳參數
  - 5-fold cross validation
  - 資料增強方法

✓ 實作細節
  - 硬體環境
  - 訓練時間
  - 遇到的問題與解決
```

### 3. 結果分析與討論 (10-12分鐘)

```
內容重點：
✓ 定量結果
  - 各肌肉的DSC, IOU表現
  - 與論文結果對比
  - 哪些肌肉容易/困難分割

✓ 定性結果
  - 視覺化分割結果
  - 成功案例分析
  - 失敗案例分析

✓ 消融實驗 (如果有做)
  - 粗略 vs 詳細標籤
  - 不同模型配置比較
  - Ensemble效果

✓ 性別差異分析
  - 模仿論文的發現
  - 討論生理學意義

✓ 應用展示
  - 肌肉油花評分系統
  - 水腫檢測
  - 疾病分類潛力

✓ 限制與未來工作
  - 資料量限制
  - 需要多中心驗證
  - 臨床應用的挑戰
```

---

## 論文方法對應表

| 論文方法 | 你的實作 | 狀態 |
|---------|---------|------|
| nnU-Net M3 (3D Low Res) | ✅ 主要模型 | Ready |
| 5-fold cross validation | ✅ 完整訓練 | Ready |
| DSC, IOU評估 | ✅ 評估腳本 | Ready |
| NSD, HD95評估 | ✅ 評估腳本 | Ready |
| 性別差異分析 | ✅ 如有metadata | Ready |
| Ensemble方法 | ⚠️ Optional | Todo |
| Transformer比較 | ⚠️ Optional | Todo |

---

## 關鍵指標目標

根據論文結果，你應該達到：

| 組織 | 論文DSC | 目標DSC | 難度 |
|-----|---------|---------|------|
| Sartorius | 0.93 | > 0.85 | 容易 |
| Quadriceps群 | 0.92 | > 0.85 | 容易 |
| Hamstring群 | 0.92 | > 0.85 | 容易 |
| Gracilis | 0.93 | > 0.80 | 中等 |
| Adductor | - | > 0.75 | 困難 |

---

## 常見問題處理

### Q1: 訓練時GPU記憶體不足
```bash
# 解決方法：
# 1. 減少batch size（nnU-Net會自動調整）
# 2. 使用3d_lowres而不是3d_fullres
# 3. 使用2D模型
```

### Q2: 某些肌肉DSC很低
```bash
# 可能原因：
# 1. 肌肉體積太小（如gracilis）
# 2. 標籤品質問題
# 3. 需要更多訓練資料

# 解決建議：
# - 針對小目標增加權重
# - 使用focal loss
# - 增加該類別的資料增強
```

### Q3: 如何加速訓練
```bash
# 方法：
# 1. 使用--npz參數（已包含）
# 2. 降低epoch數（測試用）
# 3. 只訓練單個fold先測試
# 4. 使用預訓練模型（如果有）
```

---

## 額外資源

### 論文原始碼參考
```bash
# 論文使用的nnU-Net版本
git clone https://github.com/MIC-DKFZ/nnUNet.git
```

### 視覺化工具
```python
# ITK-SNAP - 查看分割結果
http://www.itksnap.org/

# 3D Slicer - 更強大的視覺化
https://www.slicer.org/
```

### 相關論文
- nnU-Net原始論文：Isensee et al., Nature Methods 2021
- 你參考的論文：Deep learning models for fat and muscle mass

---

## 成功標準

### 最低要求（Pass）
- ✅ 完成資料準備與預處理
- ✅ 至少訓練一個模型配置
- ✅ 計算基本評估指標 (DSC, IOU)
- ✅ 完整的報告與程式碼

### 良好水準（Good）
- ✅ 完整5-fold cross validation
- ✅ 多種評估指標分析
- ✅ 視覺化結果與討論
- ✅ 與論文結果比較

### 優秀水準（Excellent）
- ✅ 嘗試ensemble或新方法
- ✅ 深入的性別/疾病分析
- ✅ 開發應用系統（評分）
- ✅ 具備發表潛力

---

## 聯絡與協作

### Git協作建議
```bash
# 建立專案repository
git init
git add .
git commit -m "Initial commit"

# 分工建議：
# - 資料準備：Person A
# - 模型訓練：Person B  
# - 評估分析：Person C
# - 報告撰寫：共同
```
