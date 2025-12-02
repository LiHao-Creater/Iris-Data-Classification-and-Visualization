# Iris Classification and Visualization

基于经典 **Iris 鸢尾花数据集**，本项目实现了一套完整的监督学习实验与可视化流程，涵盖二维决策边界、多维决策超平面、概率分布可视化以及四维特征空间上的交叉验证评估。配套可复现的 Python 代码与图像结果。

---

## 🌟 项目亮点

- 使用 **6 种经典分类器** 对 Iris 三分类任务进行对比分析（Logistic Regression、Linear SVM、k-NN、Decision Tree、Random Forest、Gaussian NB）:contentReference[oaicite:1]{index=1}  
- 在 **3D 特征空间** 中构建逻辑回归决策超平面，并进行交互式可视化与概率流形展示:contentReference[oaicite:2]{index=2}  
- 在 **4D 全特征空间** 上执行分层五折交叉验证，结合 **混淆矩阵 + ROC 曲线** 对模型进行定量评估:contentReference[oaicite:3]{index=3}  
- 单脚本端到端实现，便于一键复现实验流程和图表生成:contentReference[oaicite:4]{index=4}  

---

## 📁 项目结构（示例）

```text
.
├── classifier3d.py        # 3D 决策超平面与概率流形绘制脚本
├── report.pdf             # 实验报告（中英混合，接近论文格式）
├── figures/               # 生成的可视化结果（二维/三维图像）
│   ├── task1_logistic_regression.png
│   ├── task1_linear_svm.png
│   ├── task1_k-nn_(k=5).png
│   ├── task1_decision_tree.png
│   ├── task1_random_forest.png
│   ├── task1_gaussian_nb.png
│   ├── task2_3d_boundary.png
│   ├── task3_3d_probability_map.png
│   ├── task4_cv_bar.png
│   ├── task4_confusion_matrix.png
│   └── task4_roc_binary.png
└── README.md              # 项目说明（本文件）
