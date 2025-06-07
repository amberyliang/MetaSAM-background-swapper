# 🎨 Meta SAM Subject Background Swapper

This project uses Meta's **Segment Anything Model (SAM)** to automatically segment the subject in an image and seamlessly place it onto a custom background. Perfect for generating creative, stylized portraits or effects like portrait cutouts, moonwalk cats, or cosmic party animals 🚀.

---

## 🧰 Tools & Technologies

* **[Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)** by Meta AI
* **Python** for scripting and automation
* **PyTorch** for loading and executing the segmentation model
* **OpenCV** for image manipulation
* **Matplotlib** for visualization
* **NumPy**, **Pillow** for data handling

The SAM model generates pixel-level segmentation **masks**, from which we extract the largest segmented area — assumed to be the subject — and use it to blend the subject into a new background.

---

## ⚙️ Features

* ✅ Auto-detects and segments the main subject using Meta SAM
* ✅ Supports multiple creative background styles
* ✅ User-interactive CLI to choose a background
* ✅ Automatically resizes and composites subject over selected background
* ✅ Supports saving the result

### Available Background Styles:

* `galaxy` – cosmic deep space
* `beach` – summer & sand
* `party` – colorful gradients and lights
* `moon` – lunar surface
* `library` – study room aesthetic

---

## 📦 Project Structure

```
project/
├── Meta_SAM.py                  # Main script
├── sam_vit_h_4b8939.pth         # SAM model checkpoint (download separately)
├── backgrounds/                 # Background images
│   ├── galaxy.jpg
│   ├── beach.jpg
│   ├── party.jpg
│   ├── moon.jpg
│   └── library.jpg
└── input_image.jpg              # Your subject image
```

---

## 🚀 How to Run

1. Install dependencies:

```bash
pip install torch torchvision opencv-python matplotlib pillow
```

2. Download SAM weights from Meta:
   [https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)

3. Run the script:

```bash
python Meta_SAM.py
```

4. Choose your preferred background by typing the number prompted.

---

## ✨ Sample Results

Below are sample outputs showing original image, segmentation mask, and final result:

| Original Image             | Segmentation Mask      | Final Stylized Output           |
| -------------------------- | ---------------------- | ------------------------------- |
| ![](https://github.com/amberyliang/MetaSAM-background-swapper/blob/main/output/original.png)  | ![](https://github.com/amberyliang/MetaSAM-background-swapper/blob/main/output/mask.png)    | ![](https://github.com/amberyliang/MetaSAM-background-swapper/blob/main/output/styled.png)         |

---

## 📌 Notes

* The script defaults to **CPU execution** to avoid issues with torchvision NMS on CUDA.
* To use CUDA, ensure your `torch` and `torchvision` installations are CUDA-compatible.
* You can add your own backgrounds by placing `.jpg` files into the `backgrounds/` folder and updating the `background_options` dictionary.

---

## 🙏 Acknowledgments

* [Meta AI - SAM](https://github.com/facebookresearch/segment-anything)
* Image resources: Pexels, Freepik, Pixabay (please check licensing)

---

## 👩‍💻 Author

Crafted by Yu-Jung,Liang — feel free to fork or star ⭐ the project!
