import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image
from ultralytics import YOLO  # ✅ 新增 YOLOv8 匯入

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

print("請選擇背景風格：")
for i, key in enumerate(background_options.keys()):
    print(f"{i + 1}. {key}")
choice = int(input("請輸入數字編號："))
selected_style = list(background_options.keys())[choice - 1]
background_path = background_options[selected_style]

# ✅ 載入圖片（主體）
image_path = r"C:/Users/user/O200596276.jpg"
image_bgr = cv2.imread(image_path)
if image_bgr is None:
    raise FileNotFoundError(f"❌ 找不到主體圖片：{image_path}")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# ✅ 載入背景圖
bg_img = cv2.imread(background_path)
if bg_img is None:
    raise FileNotFoundError(f"❌ 找不到背景圖片：{background_path}")
bg_img = cv2.resize(bg_img, (image_rgb.shape[1], image_rgb.shape[0]))
bg_rgb = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)

# ✅ 載入 SAM 模型
sam_checkpoint = "C:/Users/user/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

# ✅ SAM 自動分割所有遮罩
masks = mask_generator.generate(image_rgb)

# ✅ 使用 YOLOv8 偵測人物區域
yolo_model = YOLO("yolov8n.pt").to("cpu")  # 使用最輕量化模型即可
yolo_results = yolo_model(image_rgb)

# 取得第一個人 (class 0 = person) 的 bounding box
person_box = None
for box in yolo_results[0].boxes.data:
    cls = int(box[-1])  # class ID
    if cls == 0:  # person
        x1, y1, x2, y2 = map(int, box[:4])
        person_box = (x1, y1, x2, y2)
        break

if person_box is None:
    raise ValueError("❌ YOLOv8 無法偵測到人物")

# ✅ 篩選與人框重疊最多的遮罩
x1, y1, x2, y2 = person_box
max_iou = -1
best_mask = None
for mask in masks:
    seg = mask['segmentation'].astype(np.uint8)
    mask_box = cv2.boundingRect(seg)
    mx, my, mw, mh = mask_box
    mx2, my2 = mx + mw, my + mh

    # 計算交集面積
    inter_x1 = max(x1, mx)
    inter_y1 = max(y1, my)
    inter_x2 = min(x2, mx2)
    inter_y2 = min(y2, my2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box_area = (x2 - x1) * (y2 - y1)
    iou = inter_area / box_area if box_area > 0 else 0

    if iou > max_iou:
        max_iou = iou
        best_mask = mask

if best_mask is None:
    raise ValueError("❌ 找不到與人物重疊的遮罩")
final_mask = best_mask['segmentation']

# ✅ 合成背景替換結果
composite = np.where(final_mask[:, :, None], image_rgb, bg_rgb)

# ✅ 顯示結果
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(image_rgb)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("YOLO+SAM Mask")
plt.imshow(final_mask, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title(f"Stylized: {selected_style}")
plt.imshow(composite)
plt.axis("off")
plt.tight_layout()
plt.show()

# ✅ 儲存結果
cv2.imwrite(f"output_{selected_style}.jpg", cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
