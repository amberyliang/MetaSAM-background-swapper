import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image

# ✅ 設定目前工作目錄為此 Python 檔案所在資料夾
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# ✅ 設定背景選項（請先下載並放到 backgrounds 資料夾）
background_options = {
    "galaxy": "backgrounds/galaxy.jpg",
    "beach": "backgrounds/beach.jpg",
    "party": "backgrounds/party.jpg",
    "moon": "backgrounds/moon.jpg",
    "library": "backgrounds/library.jpg"
}

# ✅ 使用者互動式選擇背景風格
print("請選擇背景風格：")
for i, key in enumerate(background_options.keys()):
    print(f"{i + 1}. {key}")

choice = int(input("請輸入數字編號："))
selected_style = list(background_options.keys())[choice - 1]
background_path = background_options[selected_style]

# ✅ 載入圖片（主體）
image_path = r"C:/Users/user/human_example.png"  # ← 絕對路徑
image_bgr = cv2.imread(image_path)
if image_bgr is None:
    raise FileNotFoundError(f"❌ 找不到主體圖片：{image_path}")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# ✅ 載入背景圖，resize 成主圖尺寸
bg_img = cv2.imread(background_path)
if bg_img is None:
    raise FileNotFoundError(f"❌ 找不到背景圖片：{background_path}")
bg_img = cv2.resize(bg_img, (image_rgb.shape[1], image_rgb.shape[0]))
bg_rgb = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)

# ✅ 載入 SAM 模型（本機路徑）
sam_checkpoint = "C:/Users/user/sam_vit_h_4b8939.pth"  # ← 改成你的模型權重
model_type = "vit_h"
device = "cpu"  # 強制使用 CPU 避免 NMS CUDA 不支援問題


sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

# ✅ 自動分割出遮罩
masks = mask_generator.generate(image_rgb)
largest_mask = max(masks, key=lambda x: x['area'])['segmentation']

# ✅ 背景替換：用遮罩疊圖
composite = np.where(largest_mask[:, :, None], image_rgb, bg_rgb)

# ✅ 顯示結果
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(image_rgb)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Mask")
plt.imshow(largest_mask, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title(f"Stylized: {selected_style}")
plt.imshow(composite)
plt.axis("off")
plt.tight_layout()
plt.show()

# ✅ 儲存結果
cv2.imwrite(f"output_{selected_style}.jpg", cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
