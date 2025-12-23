import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob
from tqdm import tqdm
from collections import defaultdict
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# 1. 基本參數設定
NUM_CLASSES_DETAILED = 12 # Detailed (12類): 0:BG, 1:SA, 2:RF, 3:VL, 4:VI, 5:VM, 6:AM, 7:GR, 8:BFL, 9:ST, 10:SM, 11:BFS
NUM_CLASSES_ROUGH = 5 # Rough (5類): 0:BG, 1:SA, 2:QF, 3:GR, 4:HS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0
PIN_MEMORY = True

# 2. 資料路徑
PROJECT_ROOT = "/home/n26141826/114-1_TAICA_cv_Final_Project"
TRAIN_DATA_DIR = os.path.join(PROJECT_ROOT,"data", "npy_2D_dataset_with_Embedding", "train") 
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "npy_2D_dataset_with_Embedding", "test")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 3. 模型存檔路徑 (Model Checkpoints)
PATH_MODEL_ROUGH = os.path.join(CHECKPOINT_DIR, "model_rough_best.pth")
PATH_MODEL_WARMUP = os.path.join(CHECKPOINT_DIR, "model_warmup.pth")
PATH_MODEL_DETAIL = os.path.join(CHECKPOINT_DIR, "model_detail_best.pth")
EVAL_MODEL_PATH = PATH_MODEL_DETAIL
EVAL_CSV_OUTPUT = os.path.join(PROJECT_ROOT, "evaluation_metrics_per_sequence.csv")

# 模型架構參數 (Model Architecture)
IN_CHANNELS = 1        # 輸入影像通道數 (灰階)
EMBEDDING_DIM = 64     # 條件向量 (Type/Pos) 的維度
NUM_MRI_TYPES = 5      # MRI 序列總數 (Water, Fat, T1, T2, STIR)

# 視覺化設定
VIZ_INTERVAL = 50

print(f"✅ Configuration Loaded!")
print(f"   - Device: {DEVICE}")
print(f"   - Checkpoints: {CHECKPOINT_DIR}")

TARGET_SIZE = 512
# 幾何變換 (Joint Transform: Image + Label)
AUG_P_FLIP = 0.5          # 左右翻轉的機率
AUG_P_SCALE = 0.5         # 隨機縮放的機率
AUG_LIMIT_SCALE = 0.1     # 縮放幅度 (0.1 代表 0.9x ~ 1.1x)

# 像素變換 (Independent Transform: Image Only)
AUG_P_BRIGHTNESS = 0.5    # 亮度/對比度調整機率
AUG_LIMIT_BRIGHT = 0.2    # 亮度調整幅度 (+-20%)
AUG_LIMIT_CONTRAST = 0.2  # 對比度調整幅度 (+-20%)

# Pre-train (Rough)
ROUGH_BATCH_SIZE = 32
ROUGH_LR = 1e-3
ROUGH_EPOCHS = 30

# Hierarchical Warm-up (Detail Head Warm-up)
WARMUP_BATCH_SIZE = 32    # 使用 Detail Data，通常量較少或需較小 Batch
WARMUP_LR = 1e-3          # 只訓練 Head，可以用大一點的 LR
WARMUP_EPOCHS = 10        # 短暫熱身即可
# 控制 Student(Detail) 模仿 Teacher(Rough) 的強度
# 建議範圍: 0.1 ~ 1.0
CONSISTENCY_WEIGHT = 0.5

# Fine-tune (Detail)
DETAIL_BATCH_SIZE = 24
DETAIL_EPOCHS = 50
DETAIL_LR_ENCODER = 1e-5 # Encoder 慢慢修 (微整形)
DETAIL_LR_DECODER = 1e-4 # Decoder 正常學

# Mappings & Definitions
ROUGH_MAP = [0, 1, 2, 2, 2, 2, 0, 3, 4, 4, 4, 4]  # 0:BG, 1:SA, 2:QF, 3:GR, 4:HS
# MRI 序列映射表 (Modality Mapping)
TYPE_MAP = {
    'Water': 0,
    'FATFRACTION': 1, # 通常將 Fat Fraction 視為 Fat 類別，或依你需求改為獨立 ID
    'Fat': 1,
    'T1': 2,
    'T2': 3,
    'STIR': 4
}
ID_TO_TYPE = {v: k for k, v in TYPE_MAP.items()}

# 定義映射矩陣 (12類 -> 5類)
# 0:BG, 1:SA, 2:RF, 3:VL, 4:VI, 5:VM, 6:AM, 7:GR, 8:BFL, 9:ST, 10:SM, 11:BFS
# Map to: 0:BG, 1:SA, 2:QF, 3:GR, 4:HS
MAPPING_MATRIX = torch.tensor([
    [1, 0, 0, 0, 0], # 0->0
    [0, 1, 0, 0, 0], # 1->1
    [0, 0, 1, 0, 0], # 2->2
    [0, 0, 1, 0, 0], # 3->2
    [0, 0, 1, 0, 0], # 4->2
    [0, 0, 1, 0, 0], # 5->2
    [1, 0, 0, 0, 0], # 6->0 (AM -> BG)
    [0, 0, 0, 1, 0], # 7->3
    [0, 0, 0, 0, 1], # 8->4
    [0, 0, 0, 0, 1], # 9->4
    [0, 0, 0, 0, 1], # 10->4
    [0, 0, 0, 0, 1]  # 11->4
], dtype=torch.float32).to(DEVICE)

# Evaluation Configuration
EVAL_BATCH_SIZE = 1
# 2. 肌肉名稱對照表 (1-11)
MUSCLE_NAMES = {
    1: 'Sartorius',
    2: 'Rectus Femoris',
    3: 'Vastus Lateralis',
    4: 'Vastus Intermedius',
    5: 'Vastus Medialis',
    6: 'Adductor Magnus',
    7: 'Gracilis',
    8: 'Biceps Femoris LH',
    9: 'Semitendinosus',
    10: 'Semimembranosus',
    11: 'Biceps Femoris SH'
}
# 3. 視覺化顏色設定 (Visualization Colors)
VIZ_COLORS = [
    '#000000', '#e6194b', '#006400', '#228B22', '#32CD32', '#7CFC00', 
    '#911eb4', '#46f0f0', '#00008B', '#0000CD', '#4169E1', '#87CEEB'
]
VIZ_CMAP = mcolors.ListedColormap(VIZ_COLORS)
VIZ_NORM = mcolors.BoundaryNorm(boundaries=np.arange(13)-0.5, ncolors=12)

# --- Metrics (為了跟組員比對，使用標準 Dice) ---
def dice_score(preds, targets, num_classes):
    # 簡單的 Dice 計算範例，組員可能有更複雜的版本，可替換
    dice_list = []
    preds = torch.argmax(preds, dim=1)
    
    for cls in range(1, num_classes): # Skip background
        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()
        
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        
        score = (2. * intersection + 1e-6) / (union + 1e-6)
        dice_list.append(score.item())
        
    return np.mean(dice_list)

class SliceMasterDataset(Dataset):
    def __init__(self, file_list, mode='rough', transform=None):
        """
        Args:
            file_list (list): List of .npy file paths
            mode (str): 'rough' or 'detail'
            transform: Albumentations transform
        """
        self.file_list = file_list
        self.mode = mode
        self.transform = transform
        # Rough Map (12 -> 5)
        self.rough_map = np.array(ROUGH_MAP, dtype=np.uint8)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # 1. Load Dictionary
        data = np.load(self.file_list[idx], allow_pickle=True).item()
        
        image = data['image'] # (H, W) float32
        z_pos = data['z_pos']
        type_idx = data['type_idx']

        # 2. Select Label
        if self.mode == 'rough':
            label = data['rough_label']
        elif self.mode == 'detail':
            label = data['detail_label']
            # 如果是 Detail 模式但沒有標註，這張圖應該被過濾或 Loss Masking
            # 這裡簡單處理：如果全黑則視為背景
        
        # 3. Augmentation
        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask']
        
        # 4. To Tensor
        if isinstance(image, np.ndarray):
            if image.ndim == 2: 
                image = image[np.newaxis, ...] # (1, H, W)
            image = image.copy()
            
        if isinstance(label, np.ndarray):
            label = label.copy()

        return (
            torch.from_numpy(image).float(),
            torch.from_numpy(label).long(),
            torch.tensor(z_pos).float(),
            torch.tensor(type_idx).long()
        )

train_transform = A.Compose([
    # 1. Independent Transform (只改 Image 不影響座標)
    A.RandomBrightnessContrast(
        brightness_limit=AUG_LIMIT_BRIGHT,  # 亮度
        contrast_limit=AUG_LIMIT_CONTRAST,  # 對比度
        p=AUG_P_BRIGHTNESS
    ),
    # (可選) 高斯雜訊
    # A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),

    # 2. Joint Transform (Image 與 Label 同步)
    A.HorizontalFlip(p=AUG_P_FLIP),   # 左右翻轉 (Horizontal Flip)
    A.RandomScale(scale_limit=AUG_LIMIT_SCALE, p=AUG_P_SCALE), # 隨機縮放 (Zoom In/Out)
    
    # 3. Resize
    A.Resize(height=TARGET_SIZE, width=TARGET_SIZE, interpolation=1),
])

# 驗證集：只做 Resize，不做任何隨機增強
val_transform = A.Compose([
    A.Resize(height=TARGET_SIZE, width=TARGET_SIZE, interpolation=1)
])

class ConditionedUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes_rough=5, n_classes_detail=12, embed_dim=64, num_mri_types=5):
        super().__init__()
        
        # --- Standard Encoder (Simplified) ---
        # 建議：如果公版用 segmentation_models_pytorch (smp)，我們可以繼承它並修改 forward
        # 這裡手寫一個簡單版示意
        self.inc = self._block(n_channels, 64)
        self.down1 = self._block(64, 128)
        self.down2 = self._block(128, 256)
        self.down3 = self._block(256, 512)
        self.down4 = self._block(512, 1024) 
        
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # --- Embedding Layers (Key Difference) ---
        self.type_emb = nn.Embedding(num_embeddings=num_mri_types, embedding_dim=embed_dim)
        self.pos_emb = nn.Sequential(
            nn.Linear(1, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )
        
        # --- Fusion Layer ---
        # 將 Image Features (1024) + Type (64) + Pos (64) 融合回 1024
        self.fusion = nn.Sequential(
            nn.Conv2d(1024 + embed_dim*2, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        # --- Decoder ---
        self.up1 = self._block(1024 + 512, 512)
        self.up2 = self._block(512 + 256, 256)
        self.up3 = self._block(256 + 128, 128)
        self.up4 = self._block(128 + 64, 64)
        
        # --- [關鍵] Dual Heads ---
        self.head_rough = nn.Conv2d(64, n_classes_rough, kernel_size=1)   # Phase 1 Target
        self.head_detail = nn.Conv2d(64, n_classes_detail, kernel_size=1) # Phase 2 Target

    def _block(self, in_c, out_c):
        """Standard Double Conv Block"""
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
        )

    def forward(self, x, type_idx, z_pos, return_mode='both'):
        """
        Args:
            return_mode: 'rough', 'detail', or 'both'
        """
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(self.pool(x1))
        x3 = self.down2(self.pool(x2))
        x4 = self.down3(self.pool(x3))
        x5 = self.down4(self.pool(x4)) # (B, 1024, H/16, W/16)
        
        # Injection
        # 1. 取得條件向量
        t_vec = self.type_emb(type_idx) # (B, 64)
        p_vec = self.pos_emb(z_pos.unsqueeze(1)) # (B, 64)
        
        # 2. 串接條件
        cond = torch.cat([t_vec, p_vec], dim=1) # (B, 128)
        
        # 3. 空間廣播 (Expand to Spatial dims)
        cond = cond.unsqueeze(2).unsqueeze(3).expand(-1, -1, x5.shape[2], x5.shape[3])
        
        # 4. 特徵融合
        x5 = torch.cat([x5, cond], dim=1)
        x5 = self.fusion(x5)
        
        # Decoder
        x = self.up1(torch.cat([self.up(x5), x4], dim=1))
        x = self.up2(torch.cat([self.up(x), x3], dim=1))
        x = self.up3(torch.cat([self.up(x), x2], dim=1))
        x = self.up4(torch.cat([self.up(x), x1], dim=1))
        
        # --- Heads ---
        if return_mode == 'rough':
            return self.head_rough(x)
        elif return_mode == 'detail':
            return self.head_detail(x)
        elif return_mode == 'both':
            return self.head_rough(x), self.head_detail(x)
        else:
            raise ValueError(f"Invalid return_mode: {return_mode}")
        
    def load_pretrained_encoder(self, weight_path):
        """載入權重，自動過濾不匹配的層"""
        if not os.path.exists(weight_path):
            print(f"⚠️ Path {weight_path} not found. Skipping.")
            return
            
        print(f"🔄 Loading weights from: {weight_path}")
        # map_location='cpu' 避免 GPU 記憶體激增
        pretrained_dict = torch.load(weight_path, map_location='cpu')
        model_dict = self.state_dict()
        
        # 過濾形狀不符的層
        filtered_dict = {k: v for k, v in pretrained_dict.items() 
                         if k in model_dict and v.shape == model_dict[k].shape}
        
        model_dict.update(filtered_dict)
        self.load_state_dict(model_dict)
        print(f"✅ Weights loaded! ({len(filtered_dict)} layers matched)")
        
    def freeze_encoder_and_rough(self):
        """Phase 1.5: Freeze everything except detail head"""
        print("🔒 Freezing Encoder layers...")
        for param in self.parameters(): param.requires_grad = False
        for param in self.head_detail.parameters(): param.requires_grad = True
        print("🔒 Frozen Encoder & Rough Head. Only 'head_detail' is trainable.")
                
        print("✅ Encoder frozen.")
    
    def unfreeze_all(self):
        """Phase 2: Unfreeze all"""
        for param in self.parameters(): param.requires_grad = True
        print("🔓 All layers unfrozen.")
        
        
test_files = glob.glob(os.path.join(TEST_DATA_DIR, "*.npy"))
if len(test_files) == 0:
    raise FileNotFoundError(f"❌ No .npy files found in {TEST_DATA_DIR}. Please check the path!")

print(f"Found {len(test_files)} slices in Test Set.")
test_detail = []
test_without_detail = []
for i in test_files:
    data = np.load(i, allow_pickle=True).item()
    if data.get('has_detail'):
        test_detail.append(i)
    else:
        test_without_detail.append(i)
print(f"  - With Detailed Labels: {len(test_detail)} slices")
print(f"  - Without Detailed Labels: {len(test_without_detail)} slices")
        
# 建立 Dataset 與 Loader
# 注意: mode='detail', transform=None (測試集不做增強)
test_ds = SliceMasterDataset(test_detail, mode='detail', transform=val_transform)
test_loader = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# ==========================================
# 2. 載入模型
# ==========================================
model = ConditionedUNet(
    n_channels=IN_CHANNELS, 
    n_classes_rough=NUM_CLASSES_ROUGH, 
    n_classes_detail=NUM_CLASSES_DETAILED, 
    embed_dim=EMBEDDING_DIM,
    num_mri_types=NUM_MRI_TYPES
).to(DEVICE)

if os.path.exists(EVAL_MODEL_PATH):
    print(f"🔄 Loading weights from: {EVAL_MODEL_PATH}")
    state_dict = torch.load(EVAL_MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state_dict)
else:
    raise FileNotFoundError(f"❌ Model weight not found at {EVAL_MODEL_PATH}")

model.eval()

# ================= 3. 推論與數據收集 =================
# 儲存結構: metrics[Sequence][Muscle_ID] = [dice1, dice2, ...]
metrics_data = defaultdict(lambda: defaultdict(list))
viz_results = []  # 儲存要畫圖的資料
print("🚀 Starting Inference on Test Set...")

with torch.no_grad():
    for idx, (images, labels, z_pos, type_idx) in enumerate(tqdm(test_loader)):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        z_pos = z_pos.to(DEVICE)
        type_idx = type_idx.to(DEVICE)
        
        # 取得當前的 MRI Type ID
        current_type_id = type_idx.item()
        
        # 推論 (return_mode='detail')
        with torch.amp.autocast("cuda"):
            out_det = model(images, type_idx, z_pos, return_mode='detail')
        
        pred_det = torch.argmax(out_det, dim=1)
        
        # --- 計算每個肌肉的 Dice 並存起來 ---
        slice_dices = []
        for c in range(1, NUM_CLASSES_DETAILED): # 1~11 (Skip BG)
            pred_mask = (pred_det == c)
            true_mask = (labels == c)
            
            inter = (pred_mask & true_mask).sum().item()
            union = (pred_mask.sum() + true_mask.sum()).item()
            
            # 只有當 GT 有該肌肉時才納入統計
            if true_mask.sum() > 0:
                dice_val = 2 * inter / (union + 1e-6)
                metrics_data[current_type_id][c].append(dice_val)
                slice_dices.append(dice_val)
            
        # --- 收集視覺化資料 (隨機抽樣) ---
        if idx % VIZ_INTERVAL == 0:
            avg_slice_dice = np.mean(slice_dices) if slice_dices else 0.0
            type_name = ID_TO_TYPE.get(current_type_id, str(current_type_id))
            
            viz_results.append({
                'type_name': type_name,
                'z': z_pos.item(),
                'img': images[0, 0].cpu().numpy(),
                'gt': labels[0].cpu().numpy(),
                'pred': pred_det[0].cpu().numpy(),
                'dice': avg_slice_dice
            })

# ================= 4. 產生報表 =================
print("\n" + "="*40)
print("   Test Set Evaluation Report (Dice Score)")
print("="*40)

final_table = {}
all_types_in_data = sorted(metrics_data.keys())

for c in range(1, NUM_CLASSES_DETAILED):
    muscle_name = MUSCLE_NAMES.get(c, f"Muscle_{c}")
    row_data = {}
    for t_id in all_types_in_data:
        dices = metrics_data[t_id][c]
        mean_dice = np.mean(dices) if dices else 0.0
        
        col_name = ID_TO_TYPE.get(t_id, str(t_id))
        row_data[col_name] = mean_dice
    final_table[muscle_name] = row_data

df_metrics = pd.DataFrame(final_table).T 
df_metrics = df_metrics.sort_index()
target_order = ['Fat', 'STIR', 'T1', 'T2', 'Water']
df_metrics = df_metrics.reindex(columns=target_order)
# 加入平均
if not df_metrics.empty:
    df_metrics.loc['AVERAGE'] = df_metrics.mean()

pd.options.display.float_format = '{:.4f}'.format
print("Mean Dice Score per Muscle per Sequence:")
display(df_metrics)

# 存檔
df_metrics.to_csv(EVAL_CSV_OUTPUT)
print(f"\nReport saved to: {EVAL_CSV_OUTPUT}")

# ================= 5. 視覺化展示 =================
if viz_results:
    print("\n" + "="*40)
    print(f"   Visualization (Sampled every {VIZ_INTERVAL} slices)")
    print("="*40)

    for item in viz_results:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        title_fs = 11
        
        # Raw Image
        axs[0].imshow(item['img'], cmap='gray')
        axs[0].set_title(f"{item['type_name']} | Z={item['z']:.2f}", fontsize=title_fs)
        axs[0].axis('off')
        
        # Ground Truth
        axs[1].imshow(item['img'], cmap='gray')
        axs[1].imshow(item['gt'], cmap=VIZ_CMAP, norm=VIZ_NORM, alpha=0.6, interpolation='nearest')
        axs[1].set_title("Ground Truth", fontsize=title_fs)
        axs[1].axis('off')
        
        # Prediction
        axs[2].imshow(item['img'], cmap='gray')
        axs[2].imshow(item['pred'], cmap=VIZ_CMAP, norm=VIZ_NORM, alpha=0.6, interpolation='nearest')
        axs[2].set_title(f"Prediction (Dice: {item['dice']:.2f})", fontsize=title_fs)
        axs[2].axis('off')
        
        plt.tight_layout()
        plt.show()
else:
    print("No visualization samples generated. Check VIZ_INTERVAL or data size.")