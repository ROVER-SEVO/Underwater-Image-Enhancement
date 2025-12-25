这是一个整合了所有信息（包括 UIEB 数据集、项目介绍、安装步骤、Python/C++ 双版本说明）的完整 `README.md` 文件内容。

你可以直接**全选复制**下面的内容，保存为 `README.md` 文件，然后上传到你的 GitHub 仓库。

---

```markdown
# Underwater Image Enhancement Tool / 水下图像增强工具

[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-blue.svg)](https://opencv.org/)
[![Python](https://img.shields.io/badge/Python-3.x-yellow.svg)](https://www.python.org/)
[![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green.svg)](https://pypi.org/project/PyQt5/)
[![C++](https://img.shields.io/badge/C++-11%2B-red.svg)](https://isocpp.org/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

**A real-time, interactive tool for enhancing underwater imagery using physics-based color restoration and dehazing techniques.**

This project provides a complete pipeline to correct color casts, remove haze, and improve contrast in underwater photos. It includes both a user-friendly **Python (PyQt5) GUI** version and a high-performance **C++ (OpenCV)** version.

这是一个基于物理模型的水下图像增强工具，支持颜色还原和去雾。本项目同时提供了 **Python (PyQt5) 图形界面版** 和 **C++ (OpenCV) 高性能版**，方便研究人员和开发者使用。

---

## 🖼️ Dataset / 数据集

This project uses the **UIEB (Underwater Image Enhancement Benchmark)** dataset for testing and validation. The raw dataset preserves the original underwater lighting conditions, making it ideal for evaluating restoration algorithms.

本项目推荐使用 **UIEB (Underwater Image Enhancement Benchmark)** 数据集进行测试。该数据集包含真实的原始水下图像，非常适合用于评估增强算法的效果。

* **Dataset Name**: UIEB Dataset (Raw)
* **Download Link**: [Kaggle - UIEB Dataset Raw](https://www.kaggle.com/datasets/larjeck/uieb-dataset-raw)
* **Description**: Contains raw underwater images covering various water types (greenish, bluish, turbid).

**How to use:**
1.  Download the dataset from the Kaggle link above.
2.  Extract the images into a local folder.
3.  Use the "Load Image" button in the Python tool to test these images.

---

## ✨ Features / 功能特性

The enhancement pipeline consists of 6 modular stages designed to tackle specific underwater degradation issues:
本增强流水线包含 6 个模块化步骤，专门解决水下图像的特定退化问题：

1.  **White Balance Correction (LAB)**: Corrects color casts by adjusting A (Green-Red) and B (Blue-Yellow) channels in LAB space.
    * *白平衡修正*：在 LAB 空间修正色偏。
2.  **Red Channel Restoration**: Compensates for the severe absorption of red light using histogram equalization.
    * *红通道恢复*：补偿水下严重被吸收的红光分量。
3.  **CLAHE**: Contrast-Limited Adaptive Histogram Equalization for improving local contrast without amplifying noise.
    * *CLAHE*：限制对比度自适应直方图均衡化，增强局部细节。
4.  **Dark Channel Dehazing**: Removes underwater haze/fog using the Dark Channel Prior (DCP) algorithm.
    * *暗通道去雾*：基于暗通道先验去除水体雾霭。
5.  **Adaptive Sharpening**: Unsharp masking to enhance edge details.
    * *自适应锐化*：增强边缘细节。
6.  **Gamma Correction**: Brightens shadow regions to reveal hidden details.
    * *Gamma 校正*：提亮暗部区域，显示更多细节。

---

## 🛠️ Quick Start / 快速开始

### Option 1: Python Version (GUI)
*Best for interactive testing and parameter tuning.*

**Requirements:**
```bash
pip install opencv-python numpy PyQt5

```

**Run:**

```bash
python underwater_enhance.py

```

**User Guide:**

1. Click **"Load Image"** to select a file (supports Chinese paths).
2. Adjust sliders (Omega, CLAHE, Shifts) to see real-time changes.
3. Click **"Save Result"** to export the enhanced image.

### Option 2: C++ Version (High Performance)

*Best for batch processing or embedded integration.*

**Requirements:**

* C++ Compiler (GCC, Clang, MSVC)
* OpenCV 4.x (Development headers)

**Compile (Linux/Mac with g++):**

```bash
g++ underwater_enhance.cpp -o EnhanceApp `pkg-config --cflags --libs opencv4`

```

**Compile (Windows with CMake):**
A standard `CMakeLists.txt` is recommended for linking OpenCV on Windows.

**Run:**

```bash
./EnhanceApp path/to/image.jpg

```

*(Press 'S' to save result, 'Q' to quit)*

---

## 📂 Project Structure / 项目结构

```text
Underwater-Image-Enhancement/
├── underwater_enhance.py    # Python source code with PyQt5 GUI
├── underwater_enhance.cpp   # C++ source code with OpenCV HighGUI
├── README.md                # Project documentation
└── requirements.txt         # Python dependencies

```

## 📊 Pipeline Comparison / 效果对比

| Stage | Visual Effect | Description |
| --- | --- | --- |
| **Original** | Hazy, Blue/Green Cast | Low visibility and color distortion. |
| **WB + Red** | Color Corrected | Red component restored, color cast removed. |
| **Dehaze** | Clearer | Haze/Fog removed, transmission map applied. |
| **Final** | Sharp & Natural | Local contrast enhanced and edges sharpened. |

## 🤝 Contributing / 贡献

Contributions are welcome! If you find a bug or have an idea for improvement (e.g., Deep Learning integration), please feel free to submit an issue or a pull request.

欢迎提交 Issue 或 Pull Request！如果你有改进建议（例如集成深度学习模型），请随时联系。

## 📄 License

This project is open source and available under the [MIT License](https://www.google.com/search?q=LICENSE).

```

```
