import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    RocCurveDisplay,
)

RANDOM_STATE = 42


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def plot_decision_regions_2d(
    clf,
    X,
    y,
    feature_names,
    title,
    save_path,
    h=0.02,
):
    """
    画二维决策边界 + 散点（多分类）
    clf: 已经 fit 的分类器（支持 predict）
    X: (n_samples, 2)
    y: (n_samples,)
    feature_names: [name_x, name_y]
    """
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(grid)
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 5))
    # 决策边界区域
    im = ax.pcolormesh(xx, yy, Z, shading="auto", alpha=0.3)
    # 样本点
    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        edgecolors="k",
        s=40,
        alpha=0.9,
    )
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title(title)
    ax.grid(True, ls="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def task1_many_classifiers_2d(out_dir):
    """
    Task 1:
    三分类 / 两个特征
    可视化不同分类器的二维决策边界
    """
    iris = load_iris()
    # 使用花瓣长度/宽度，更容易分
    X = iris.data[:, 2:4]
    y = iris.target
    feature_names = [iris.feature_names[2], iris.feature_names[3]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    classifiers = {
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=500,
                        multi_class="auto",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "LinearSVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LinearSVC(random_state=RANDOM_STATE)),
            ]
        ),
        "RBFSVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "KNN(k=5)": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=4, random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=150, random_state=RANDOM_STATE
        ),
        "GaussianNB": GaussianNB(),
    }

    results = []
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        results.append((name, test_acc))
        filename = f"task1_{name}.png"
        save_path = os.path.join(out_dir, filename)
        plot_decision_regions_2d(
            clf,
            X,
            y,
            feature_names,
            f"{name} (test acc={test_acc:.3f})",
            save_path,
        )
        print(f"[Task1] {name:15s} test accuracy = {test_acc:.3f}, figure -> {save_path}")

    return results

def task2_3d_boundary(out_dir):
    """
    Task 2:
    两分类 / 三个特征
    可视化 3D 决策边界（Logistic Regression 的平面）
    """
    iris = load_iris()
    X_all = iris.data
    y_all = iris.target
    # 选择两类：Versicolor(1) vs Virginica(2)
    mask = (y_all != 0)
    X_bin = X_all[mask]
    y_bin = y_all[mask]
    # 重新映射为 0/1
    y_bin = (y_bin == 2).astype(int)

    # 三个特征：sepal length, sepal width, petal length
    feat_idx = [0, 1, 2]
    X3d = X_bin[:, feat_idx]
    feat_names = [iris.feature_names[i] for i in feat_idx]

    # 标准化 + 逻辑回归
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000, random_state=RANDOM_STATE
                ),
            ),
        ]
    )
    pipe.fit(X3d, y_bin)

    # 决策平面参数（在标准化后的空间是线性的）
    X_scaled = pipe.named_steps["scaler"].transform(X3d)
    clf = pipe.named_steps["clf"]
    w = clf.coef_.ravel()  # (3,)
    b = clf.intercept_[0]

    # 画标准化后的坐标系
    x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
    y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 40),
        np.linspace(y_min, y_max, 40),
    )
    if abs(w[2]) < 1e-8:
        zz = np.zeros_like(xx)
    else:
        zz = -(w[0] * xx + w[1] * yy + b) / w[2]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    # 散点（用不同 marker 表示两类）
    ax.scatter(
        X_scaled[y_bin == 0, 0],
        X_scaled[y_bin == 0, 1],
        X_scaled[y_bin == 0, 2],
        marker="o",
        s=35,
        alpha=0.9,
        label="Versicolor (0)",
    )
    ax.scatter(
        X_scaled[y_bin == 1, 0],
        X_scaled[y_bin == 1, 1],
        X_scaled[y_bin == 1, 2],
        marker="^",
        s=40,
        alpha=0.9,
        label="Virginica (1)",
    )
    # 决策平面
    ax.plot_surface(xx, yy, zz, alpha=0.3)

    ax.set_xlabel(feat_names[0] + " (scaled)")
    ax.set_ylabel(feat_names[1] + " (scaled)")
    ax.set_zlabel(feat_names[2] + " (scaled)")
    ax.set_title("Task 2: 3D Decision Boundary (Logistic Regression)")
    ax.legend()
    plt.tight_layout()
    save_path = os.path.join(out_dir, "task2_3d_boundary.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Task2] 3D boundary figure -> {save_path}")


def task3_3d_probability_map(out_dir):
    """
    Task 3:
    两分类 / 三个特征
    可视化 3D Probability Map
    做法：在多个 z 切片上画 p(class=1) 的等高线投影
    """
    iris = load_iris()
    X_all = iris.data
    y_all = iris.target
    mask = (y_all != 0)
    X_bin = X_all[mask]
    y_bin = y_all[mask]
    y_bin = (y_bin == 2).astype(int)

    feat_idx = [0, 1, 2]
    X3d = X_bin[:, feat_idx]
    feat_names = [iris.feature_names[i] for i in feat_idx]

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
    pipe.fit(X3d, y_bin)

    # 标准化后的坐标
    X_scaled = pipe.named_steps["scaler"].transform(X3d)

    # 几个 z 切片
    z_values = np.quantile(X_scaled[:, 2], [0.2, 0.5, 0.8])

    x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
    y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 60),
        np.linspace(y_min, y_max, 60),
    )

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    for z in z_values:
        grid_scaled = np.c_[xx.ravel(), yy.ravel(), np.full_like(xx.ravel(), z)]
        # 注意：predict_proba 的输入需要还原回原始尺度（inverse_transform）
        proba = pipe.predict_proba(
            pipe.named_steps["scaler"].inverse_transform(grid_scaled)
        )[:, 1]
        proba = proba.reshape(xx.shape)
        ax.contourf(
            xx,
            yy,
            proba,
            zdir="z",
            offset=z,
            levels=12,
            alpha=0.7,
        )

    # 叠加样本点
    ax.scatter(
        X_scaled[y_bin == 0, 0],
        X_scaled[y_bin == 0, 1],
        X_scaled[y_bin == 0, 2],
        marker="o",
        s=18,
        alpha=0.9,
        label="Versicolor (0)",
    )
    ax.scatter(
        X_scaled[y_bin == 1, 0],
        X_scaled[y_bin == 1, 1],
        X_scaled[y_bin == 1, 2],
        marker="^",
        s=22,
        alpha=0.9,
        label="Virginica (1)",
    )

    ax.set_xlabel(feat_names[0] + " (scaled)")
    ax.set_ylabel(feat_names[1] + " (scaled)")
    ax.set_zlabel(feat_names[2] + " (scaled)")
    ax.set_title("Task 3: 3D Probability Map (p(class=1) slices)")
    ax.legend()
    plt.tight_layout()
    save_path = os.path.join(out_dir, "task3_3d_probability_map.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Task3] 3D probability map figure -> {save_path}")


def task4_bonus_evaluation(out_dir):
    """
    Task 4:
    能提高分数的任何事情 —— 这里给出一个建议实现：
    1) 7 个模型在 4 维特征上的 5 折交叉验证对比（条形图）
    2) 选取 CV 最好的模型，画 2D 特征上的混淆矩阵
    3) 二分类（Versicolor vs Virginica）的 ROC 曲线
    """
    iris = load_iris()
    X_all = iris.data
    y_all = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    models_full = {
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=500, random_state=RANDOM_STATE
                    ),
                ),
            ]
        ),
        "LinearSVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LinearSVC(random_state=RANDOM_STATE)),
            ]
        ),
        "RBFSVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "KNN(k=5)": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=4, random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=150, random_state=RANDOM_STATE
        ),
        "GaussianNB": GaussianNB(),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    model_names = []
    cv_means = []
    cv_stds = []
    for name, model in models_full.items():
        scores = cross_val_score(model, X_all, y_all, cv=cv)
        model_names.append(name)
        cv_means.append(scores.mean())
        cv_stds.append(scores.std())
        print(f"[Task4-CV] {name:15s}: mean={scores.mean():.4f}, std={scores.std():.4f}")

    # 条形图
    fig, ax = plt.subplots(figsize=(7, 5))
    y_pos = np.arange(len(model_names))
    ax.barh(y_pos, cv_means, xerr=cv_stds, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_names)
    ax.set_xlabel("Accuracy (5-fold CV mean ± std)")
    ax.set_title("Task 4: Model comparison on all 4 features")
    ax.grid(axis="x", ls="--", alpha=0.4)
    plt.tight_layout()
    bar_path = os.path.join(out_dir, "task4_cv_bar.png")
    fig.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Task4] CV bar plot -> {bar_path}")

    # 选 CV 最好的模型，画二维特征上的混淆矩阵
    best_idx = int(np.argmax(cv_means))
    best_name = model_names[best_idx]
    best_model = models_full[best_name]

    # 仍然用花瓣长度/宽度（与 Task1 一致）
    X2d = X_all[:, 2:4]
    X2d_train, X2d_test, y_train, y_test = train_test_split(
        X2d, y_all, test_size=0.3, random_state=RANDOM_STATE, stratify=y_all
    )
    best_model.fit(X2d_train, y_train)
    y_pred = best_model.predict(X2d_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=target_names)
    disp.plot(ax=ax, colorbar=True)
    ax.set_title(f"Confusion Matrix ({best_name}, 2D split)")
    plt.tight_layout()
    cm_path = os.path.join(out_dir, "task4_confusion_matrix.png")
    fig.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Task4] Confusion matrix -> {cm_path}")

    # 二分类 ROC（Versicolor vs Virginica，三维特征）
    mask = (y_all != 0)
    X_bin = X_all[mask][:, :3]  # 前 3 个特征
    y_bin = y_all[mask]
    y_bin = (y_bin == 2).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X_bin,
        y_bin,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=y_bin,
    )
    clf_bin = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
    clf_bin.fit(X_train, y_train)
    y_score = clf_bin.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="LogReg(3 feats)").plot(ax=ax)
    ax.set_title("Task 4: ROC (Versicolor vs Virginica)")
    plt.tight_layout()
    roc_path = os.path.join(out_dir, "task4_roc_binary.png")
    fig.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Task4] ROC curve -> {roc_path}")


def main(output_dir="project3_results"):
    ensure_dir(output_dir)
    print("输出目录:", output_dir)
    print("========== Task 1: 2D 多分类决策边界 ==========")
    task1_many_classifiers_2d(output_dir)
    print("========== Task 2: 3D Boundary ==========")
    task2_3d_boundary(output_dir)
    print("========== Task 3: 3D Probability Map ==========")
    task3_3d_probability_map(output_dir)
    print("========== Task 4: Bonus Evaluation ==========")
    task4_bonus_evaluation(output_dir)
    print("全部任务完成，图像已保存到:", output_dir)


if __name__ == "__main__":
    main()
