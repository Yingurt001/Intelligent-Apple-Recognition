<!-- Banner -->
<div align="center">
  <img src="Assets/Banner_1.jpg" width="100%" alt="Project Banner">
</div>

<!-- Badges -->
<div align="center">
  <img src="https://img.shields.io/github/stars/你的用户名/你的项目名" alt="Stars">
  <img src="https://img.shields.io/github/license/你的用户名/你的项目名" alt="License">
  <img src="https://img.shields.io/badge/python-3.8+-blue" alt="Python Version">
</div>

<!-- Social Media -->
<div align="center">
  <a href="https://github.com/yourname">
    <img src="github图标链接" width="3%">
  </a>
  <a href="https://space.bilibili.com/xxxxxx">
    <img src="bilibili图标链接" width="3%">
  </a>
</div>

---

# 🍎 Intelligent Apple Recognition System | 苹果智能识别系统

A computer vision and deep learning-based system for identifying, classifying, and evaluating apples in orchard images.  
基于计算机视觉与深度学习的苹果图像识别系统，实现苹果计数、定位、成熟度评估与品质识别。

---

## 🗂️ Project Overview | 项目概览

This project implements a complete apple recognition pipeline using Python, OpenCV, and neural networks. It includes:

- 🍏 Apple counting and localization using image processing  
- 🔥 Heatmap visualization of apple density  
- 🧠 Ripeness classification via CNN  
- 📊 Weight estimation via Monte Carlo simulation  
- 🏷️ Quality classification using pre-trained ResNet50  

本项目包含以下核心模块：

- 图像分割与苹果计数定位  
- 热力图展示密度分布  
- CNN 成熟度分类模型  
- 蒙特卡洛法进行重量估计  
- 基于 ResNet50 的品质识别  

---


## 📦 Installation | 安装方式

```bash
git clone https://github.com/yourname/apple-segmentation
cd apple-segmentation
pip install -r requirements.txt

---
##  Dataset Description | 数据集说明

The project uses image datasets of apples and mixed fruits provided by the 2023 APMCM Problem A.

###  Attachment 1: Apple Orchard Images
- 200 RGB images of harvest-ready apples in natural orchard environments
- Image size: 270 × 180 pixels
- Apples are presented with various occlusion types: leaf, branch, fruit, and mixed occlusion
- Tasks: apple counting, positioning, maturity estimation, and mass estimation

###  Attachment 2: Labeled Fruit Images
- 20,705 labeled images of five fruit types: apple, carambola, pear, plum, tomato
- Each image has a size of 270 × 180 pixels
- Used for training a fruit classifier model (e.g., ResNet50)

###  Attachment 3: Unlabeled Fruit Images
- 20,705 unlabeled fruit images with identical format to Attachment 2
- The task is to classify and identify apples among them using the trained model



项目使用了 2023 APMCM A 题所提供的苹果图像与混合果蔬图像数据集：

### 附件1：果园苹果图像
- 共 200 张 RGB 图像，拍摄于自然果园环境
- 图像尺寸：270 × 180 像素
- 苹果存在多种遮挡情况：叶片遮挡、枝干遮挡、果实遮挡与混合遮挡
- 用于执行苹果计数、定位、成熟度与质量估计等任务

### 附件2：标注水果图像
- 共 20705 张已标注的果蔬图像，包括苹果、杨桃、梨、李子和番茄共五类
- 每张图像大小为 270 × 180 像素
- 用于训练水果识别模型，如 ResNet50 网络

### 附件3：未标注水果图像
- 共 20705 张与附件2相同格式的未标注图像
- 目标为使用训练好的模型识别其中的苹果图像
✅

---
## 🧱 Project Structure | 项目结构

```bash
notebooks/
├── 1_count_location.ipynb          # Apple Detection & Counting | 图像处理与苹果定位
├── 2_location_scatter_heatmap.ipynb # Heatmap Visualization | 热力图与坐标可视化
├── 3_maturity_cnn.ipynb            # Maturity Classification (CNN) | 成熟度分类模型
├── 4_mass_montecarlo.ipynb         # Weight Estimation (Monte Carlo) | 重量估计
├── 5_resnet50_fruit_classification.ipynb # Quality Classification (ResNet50) | 品质识别
├── README_Q1.md                    # Q1 Documentation | Q1说明文档

data/
├── Attachment 1/                   # Apple orchard images (200 files)
├── Attachment 2/                   # Labeled fruit dataset (20,705 files)
└── Attachment 3/                   # Unlabeled fruit dataset (20,705 files)

results/
└── Q1_results/                     # Q1 processing outputs

Previous Code/                      # Legacy implementation for reference


### Result
![3_detected](https://github.com/user-attachments/assets/df885f4d-6179-47b9-86fa-006c4b363ad8)


### Project Maintainers 

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/your-github">
        <img src="https://avatars.githubusercontent.com/u/你的ID?v=4" width="100px;" alt=""/>
        <br />
        <sub><b>你的名字</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/friend-github">
        <img src="https://avatars.githubusercontent.com/u/朋友ID?v=4" width="100px;" alt=""/>
        <br />
        <sub><b>朋友的名字</b></sub>
      </a>
    </td>
  </tr>
</table>



Paper Reference | 背景论文

Author: Ying Zhang, Tianhao Hua, Yuzhi Zheng
GitHub: @Yingurt001
Email: millionyogurt@gmail.com, 

欢迎 star / fork 本项目，也欢迎提出改进建议与合作 🙌
