import nibabel as nib
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_similarity(file1, file2):
    # 1. 載入影像
    img1 = nib.load(file1)
    img2 = nib.load(file2)
    
    # 取得數據並轉為浮點數 (避免溢位)
    data1 = img1.get_fdata().astype(np.float32)
    data2 = img2.get_fdata().astype(np.float32)

    # 檢查尺寸是否一致
    if data1.shape != data2.shape:
        print(f"錯誤: 影像尺寸不同 {data1.shape} vs {data2.shape}，無法直接比對。")
        return

    # --- 指標 1: MSE (均方誤差) ---
    mse = np.mean((data1 - data2) ** 2)
    
    # --- 指標 2: 相關係數 (Pearson Correlation) ---
    # 將 3D 陣列拉平為 1D 進行計算
    flat1 = data1.flatten()
    flat2 = data2.flatten()
    correlation = np.corrcoef(flat1, flat2)[0, 1]

    # --- 指標 3: SSIM (結構相似性) ---
    # 注意: SSIM 計算較慢，且需要指定 data_range
    # 這裡我們假設影像是 3D 的
    data_range = data1.max() - data1.min()
    ssim_val = ssim(data1, data2, data_range=data_range, channel_axis=None)

    print(f"--- 相似度分析報告 ---")
    print(f"檔案 1: {file1}")
    print(f"檔案 2: {file2}")
    print(f"-" * 30)
    print(f"1. MSE (誤差，越低越好):      {mse:.6f}")
    print(f"2. SSIM (結構，接近 1 越好):   {ssim_val:.6f}")
    print(f"3. Corr (相關，接近 1 越好):   {correlation:.6f}")

    # --- 額外: 如果是 Mask (只有 0 和 1 的數據)，加算 Dice ---
    unique_vals = np.unique(data1)
    if len(unique_vals) <= 5:  # 簡單判斷是否為 Mask
        intersection = np.sum((data1 > 0) * (data2 > 0))
        sum_pixels = np.sum(data1 > 0) + np.sum(data2 > 0)
        dice = (2.0 * intersection) / sum_pixels if sum_pixels > 0 else 1.0
        print(f"4. Dice (重疊，接近 1 越好):   {dice:.6f}")

# 使用範例
calculate_similarity('/home/n26141826/114-1_TAICA_cv_Final_Project/data2/data_fixed/3D_dataset/train/rough_label/510_HV009_1_WATER.nii.gz'
                     , '/home/n26141826/114-1_TAICA_cv_Final_Project/data2/data_fixed/3D_dataset/train/rough_label/508_HV009_1_FATFRACTION.nii.gz')