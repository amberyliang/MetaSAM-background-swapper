# 🎨 Meta SAM Subject Background Swapper

This project uses Meta's **Segment Anything Model (SAM)** and **YOLOv8** to automatically segment the *human subject* in an image and seamlessly place it onto a custom background. Perfect for generating creative, stylized portraits or effects like moonwalk cats, cosmic party animals, or professional-style cutouts. 🚀

---

## 🧰 Tools & Technologies
- **Segment Anything Model (SAM)** by Meta AI
- **YOLOv8** for accurate person detection
- **Python** for scripting and automation
- **PyTorch** for loading models
- **OpenCV** for image processing
- **Matplotlib** for visualization
- **NumPy**, **Pillow** for data handling

SAM generates pixel-level masks of all visible objects. We enhance it using YOLOv8 to accurately locate the human subject and filter the SAM masks to avoid misidentification in complex or dark backgrounds.

---

## ⚙️ Features
✅ Detects and segments **only the human subject** using YOLOv8 + SAM mask filtering  
✅ Supports multiple creative background styles  
✅ Interactive CLI to choose a background  
✅ Auto-resizes and composites subject into selected background  
✅ Results are visualized and saved automatically  

### 🎨 Available Background Styles:
- `galaxy` – cosmic deep space
- `beach` – summer & sand
- `party` – colorful gradients and lights
- `moon` – lunar surface
- `library` – study room aesthetic

---

## 📦 Project Structure
```
project/
├── Meta_SAM.py                  # Main script
├── sam_vit_h_4b8939.pth         # SAM model checkpoint (download separately)
├── yolov8n.pt                   # YOLOv8 model weights (download separately)
├── backgrounds/                 # Background images
│   ├── galaxy.jpg
│   ├── beach.jpg
│   ├── party.jpg
│   ├── moon.jpg
│   └── library.jpg
├── output/                      # Output images
│   ├── mask.png
│   ├── original.png
│   └── styled.png
└── input_image.jpg              # Your subject image
```

---

## 🚀 How to Run

### 1. Install dependencies:
```bash
pip install torch torchvision opencv-python matplotlib pillow ultralytics
```

### 2. Download weights:
- [SAM weights](https://github.com/facebookresearch/segment-anything)
- [YOLOv8 weights](https://github.com/ultralytics/ultralytics)

### 3. Run the script:
```bash
python Meta_SAM.py
```
Choose your background style when prompted.

---

## ✨ Sample Results
Below are sample outputs showing original image, YOLO-filtered segmentation mask, and final result:

| Original | Segmentation Mask | Final Output |
|----------|-------------------|---------------|
| ![](https://github.com/amberyliang/MetaSAM-background-swapper/blob/main/output/original.png) | ![](https://github.com/amberyliang/MetaSAM-background-swapper/blob/main/output/mask.png) | ![](https://github.com/amberyliang/MetaSAM-background-swapper/blob/main/output/styled.png) |

---

## 📌 Notes
- The script runs on **CPU by default** to avoid torchvision CUDA NMS issues.
- If using GPU, make sure PyTorch and Torchvision are **CUDA-compatible**.
- You can add your own `.jpg` backgrounds by placing them in the `backgrounds/` folder.
- YOLOv8 is used to ensure only human subjects are segmented from images where SAM alone may fail.


## 🙏 Acknowledgments

* [Meta AI - SAM](https://github.com/facebookresearch/segment-anything)
* Image resources: Pexels, Freepik, Pixabay (please check licensing)

---

## 👩‍💻 Author

Crafted by Yu-Jung,Liang — feel free to fork or star ⭐ the project!
