# 🍎 Intelligent Apple Recognition System | 苹果智能识别系统

A computer vision and deep learning-based system for identifying, classifying, and evaluating apples in orchard images.  
基于计算机视觉与深度学习的苹果图像识别与分析系统，实现苹果的计数、定位、成熟度评估与品质识别。

---

## 📌 Project Overview | 项目概览

This project implements a complete apple recognition pipeline using Python, OpenCV, and neural networks. It includes:

- ✅ Apple counting and localization using image processing
- ✅ Heatmap visualization of apple density across regions
- ✅ Ripeness classification using a custom CNN
- ✅ Weight estimation using Monte Carlo simulation
- ✅ Quality classification using pre-trained ResNet50

本项目使用 Python 与深度学习技术，完成苹果图像的全流程处理与分析，具体包括：

- 🍏 图像分割与苹果计数定位
- 🗺 热力图可视化展示苹果密度分布
- 🎯 使用 CNN 进行成熟度分类
- ⚖️ 运用蒙特卡洛方法估算苹果重量
- 🧠 利用 ResNet50 进行品质识别与分类

---

## 🧱 Project Structure | 项目结构

```bash
notebooks/
├── 1_count_location.ipynb          # 图像处理与苹果定位
├── 2_location_scatter_heatmap.ipynb # 热力图与坐标可视化
├── 3_maturity_cnn.ipynb            # 成熟度分类模型（CNN）
├── 4_mass_montecarlo.ipynb         # 重量估计（Monte Carlo 方法）
├── 5_resnet50_fruit_classification.ipynb # 品质识别（ResNet50）


Paper Reference | 背景论文

Author: Ying Zhang
GitHub: @Yingurt001
Email: millionyogurt@gmail.com

欢迎 star / fork 本项目，也欢迎提出改进建议与合作 🙌
