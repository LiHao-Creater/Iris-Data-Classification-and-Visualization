import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 定义背景颜色常量，增强对比度 ---
BG_COLOR_SCENE = "rgb(180, 180, 180)" # 整体场景背景色
BG_COLOR_WALL = "rgb(180, 180, 180)"  # 轴墙壁背景色
GRID_COLOR = "rgba(100, 100, 100, 0.5)" # 网格线颜色
TITLE_FONT_COLOR = "black" # 标题字体颜色在深色背景下保持黑色


# ==========================================
# 1. 数据准备与特征工程 (Data Preparation and Feature Engineering)
# ==========================================
# 加载Iris数据集
iris = load_iris()
X = iris.data[:, :3] # 选择前三个特征
y = iris.target

# 筛选难分类别 (Versicolor vs Virginica)
mask = y != 0
X_filtered = X[mask]
y_filtered = np.where(y[mask] == 1, 0, 1)

# 标准化处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filtered)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_filtered, test_size=0.3, random_state=42
)

# ==========================================
# 2. 建模与流形计算 (Modeling and Manifold Calculation)
# ==========================================
# 训练逻辑回归模型
model = LogisticRegression(solver='lbfgs', C=10.0)
model.fit(X_train, y_train)

# 提取超平面参数
w = model.coef_[0]
b = model.intercept_[0]

# 生成高分辨率网格 (100x100)
x_range = np.linspace(X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1, 100)
y_range = np.linspace(X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1, 100)
xx, yy = np.meshgrid(x_range, y_range)

# 计算决策平面 Z 坐标
zz = -(w[0] * xx + w[1] * yy + b) / w[2]

# 计算概率流形 (Task 3 + Task 4 融合)
grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
probs = model.predict_proba(grid_points)[:, 1].reshape(xx.shape)

# ==========================================
# 3. 绘图渲染 (高对比度与高级感) (High Contrast Rendering)
# ==========================================
fig = go.Figure()

# --- A. 绘制数据点 (尺寸减小一半) ---
# Class 0: Versicolor (鲜艳蓝色)
fig.add_trace(go.Scatter3d(
    x=X_train[y_train == 0, 0], y=X_train[y_train == 0, 1], z=X_train[y_train == 0, 2],
    mode='markers',
    name='Class 0 (Versicolor)',
    # 数据点尺寸减小到 4
    marker=dict(size=4, color='rgba(0, 0, 255, 1)', line=dict(width=1.5, color='white'), opacity=1.0)
))

# Class 1: Virginica (鲜红色)
fig.add_trace(go.Scatter3d(
    x=X_train[y_train == 1, 0], y=X_train[y_train == 1, 1], z=X_train[y_train == 1, 2],
    mode='markers',
    name='Class 1 (Virginica)',
    # 数据点尺寸减小到 4
    marker=dict(size=4, color='rgba(255, 0, 0, 1)', line=dict(width=1.5, color='white'), symbol='diamond', opacity=1.0)
))

# --- B. 绘制决策平面与概率场 (更换高对比度色谱) ---
# 【修复点】使用渐变色，避免中间出现纯白色宽带，改用高亮中性色
CUSTOM_COLORSCALE = [
    [0.0, 'rgb(0, 50, 150)'],    # P=0.0: 深蓝
    [0.4, 'rgb(150, 200, 255)'], # P=0.4: 亮蓝
    [0.5, 'rgb(255, 255, 150)'], # P=0.5: 柔和的亮黄色（突出边界且保持色彩）
    [0.6, 'rgb(255, 150, 150)'], # P=0.6: 亮红
    [1.0, 'rgb(150, 0, 0)']      # P=1.0: 深红
]

fig.add_trace(go.Surface(
    x=xx, y=yy, z=zz,
    surfacecolor=probs,
    colorscale=CUSTOM_COLORSCALE, # 使用修复后的渐变色谱
    cmin=0, cmax=1,
    opacity=0.8, # 增加不透明度，使平面更实体化
    lighting=dict(ambient=0.5, diffuse=0.5, specular=0.8, fresnel=0.1),
    colorbar=dict(
        title=dict(text='Probability P(y=1)', side='right', font=dict(size=14, color=TITLE_FONT_COLOR)),
        thickness=15, len=0.6,
        bgcolor=BG_COLOR_WALL 
    ),
    name='Decision Manifold'
))

# --- C. 底部阴影投影 ---
z_floor = X_scaled[:, 2].min() - 1.5
fig.add_trace(go.Scatter3d(
    x=X_scaled[:, 0], y=X_scaled[:, 1], z=np.full_like(X_scaled[:, 0], z_floor),
    mode='markers',
    marker=dict(size=2, color='rgba(50, 50, 50, 0.5)'),
    hoverinfo='skip',
    showlegend=False
))

# --- D. 布局美化 (增强对比度布局) ---
fig.update_layout(
    title=dict(
        text=r'<b>3D Linear Decision Boundary & Probability Landscape</b><br><span style="font-size:12px;color:dimgray">Logistic Regression in Feature Space $\mathbb{R}^3$</span>',
        y=0.95, font=dict(color=TITLE_FONT_COLOR)
    ),
    scene=dict(
        xaxis=dict(
            title=dict(text='Sepal Length (std)', font=dict(size=16, color=TITLE_FONT_COLOR)), 
            backgroundcolor=BG_COLOR_WALL, gridcolor=GRID_COLOR, showbackground=True, 
            tickfont=dict(size=12, color=TITLE_FONT_COLOR)),
        yaxis=dict(
            title=dict(text='Sepal Width (std)', font=dict(size=16, color=TITLE_FONT_COLOR)), 
            backgroundcolor=BG_COLOR_WALL, gridcolor=GRID_COLOR, showbackground=True,
            tickfont=dict(size=12, color=TITLE_FONT_COLOR)),
        zaxis=dict(
            title=dict(text='Petal Length (std)', font=dict(size=16, color=TITLE_FONT_COLOR)), 
            backgroundcolor=BG_COLOR_WALL, gridcolor=GRID_COLOR, showbackground=True, 
            range=[z_floor, X_scaled[:, 2].max() + 0.5],
            tickfont=dict(size=12, color=TITLE_FONT_COLOR)),
        
        bgcolor=BG_COLOR_SCENE, # 整体场景背景色
        aspectmode='cube',
        camera=dict(eye=dict(x=1.6, y=1.6, z=1.4))
    ),
    margin=dict(l=0, r=0, b=0, t=60),
    font=dict(family="Arial, sans-serif", size=12, color=TITLE_FONT_COLOR)
)

fig.show()