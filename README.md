

# 克拉尼版控制
---

# 变厚度自由边界薄板受迫振动方程推导

### 1. 基本假设与几何关系
基于 Kirchhoff-Love 薄板理论，假设中面内无变形，且变形前垂直于中面的直线在变形后仍为直线且垂直于变形后的中面。设薄板厚度为变量 $h = h(x,y)$，材料密度为 $\rho$，杨氏模量为 $E$，泊松比为 $\nu$。

板的挠度（中面的法向位移）为 $w = w(x,y,t)$。
根据几何方程，板的曲率与扭率可表示为：
$$\kappa_x = -\frac{\partial^2 w}{\partial x^2}, \quad \kappa_y = -\frac{\partial^2 w}{\partial y^2}, \quad \kappa_{xy} = -\frac{\partial^2 w}{\partial x \partial y}$$

### 2. 本构方程与内力
由于板厚 $h(x,y)$ 是空间坐标的函数，板的抗弯刚度 $D$ 也是变参数：
$$D(x,y) = \frac{E h^3(x,y)}{12(1-\nu^2)}$$

对厚度方向积分，可以得到弯矩 $M_x, M_y$ 和扭矩 $M_{xy}$：
$$\begin{aligned} M_x &= -D(x,y) \left( \frac{\partial^2 w}{\partial x^2} + \nu \frac{\partial^2 w}{\partial y^2} \right) \\ M_y &= -D(x,y) \left( \frac{\partial^2 w}{\partial y^2} + \nu \frac{\partial^2 w}{\partial x^2} \right) \\ M_{xy} &= -D(x,y) (1-\nu) \frac{\partial^2 w}{\partial x \partial y} \end{aligned}$$

### 3. 变厚度薄板受迫振动控制方程
取微元体进行受力分析。设横向分布载荷为 $q(x,y,t)$，惯性力为 $-\rho h(x,y) \frac{\partial^2 w}{\partial t^2}$。由微元体的动量平衡（牛顿第二定律）和角动量平衡，可得：
$$\frac{\partial^2 M_x}{\partial x^2} + 2\frac{\partial^2 M_{xy}}{\partial x \partial y} + \frac{\partial^2 M_y}{\partial y^2} + q(x,y,t) = \rho h(x,y) \frac{\partial^2 w}{\partial t^2}$$

将内力表达式 (3) 代入运动方程 (4) 中。需要注意的是，由于 $D$ 是 $x$ 和 $y$ 的函数，求导时必须将其作为变量处理：
$$\begin{aligned} & \frac{\partial^2}{\partial x^2} \left[ D \left( \frac{\partial^2 w}{\partial x^2} + \nu \frac{\partial^2 w}{\partial y^2} \right) \right] \\ + & 2 \frac{\partial^2}{\partial x \partial y} \left[ D (1-\nu) \frac{\partial^2 w}{\partial x \partial y} \right] \\ + & \frac{\partial^2}{\partial y^2} \left[ D \left( \frac{\partial^2 w}{\partial y^2} + \nu \frac{\partial^2 w}{\partial x^2} \right) \right] + \rho h \frac{\partial^2 w}{\partial t^2} = q(x,y,t) \end{aligned}$$

将其展开并合并同类项，得到变厚度薄板的振动控制方程的显式形式：
$$\nabla^2 \left( D \nabla^2 w \right) - (1-\nu) \left( \frac{\partial^2 D}{\partial x^2}\frac{\partial^2 w}{\partial y^2} - 2\frac{\partial^2 D}{\partial x \partial y}\frac{\partial^2 w}{\partial x \partial y} + \frac{\partial^2 D}{\partial y^2}\frac{\partial^2 w}{\partial x^2} \right) + \rho h \frac{\partial^2 w}{\partial t^2} = q(x,y,t)$$
其中，$\nabla^2 = \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}$ 为拉普拉斯算子。若板厚均匀（$D$ 为常数），中间的交叉项将消失，方程退化为经典的 $D\nabla^4 w + \rho h \ddot{w} = q$。

### 4. 连续问题的算子形式：$(\mathcal{K} - \omega^2 \mathcal{M})W = Q$

从薄板受迫振动的稳态控制方程出发，设横向载荷和位移场为简谐形式 $q(x,y,t) = Q(x,y)e^{i\omega t}$，挠度为 $w(x,y,t) = W(x,y)e^{i\omega t}$。

将上式代入动力学平衡方程 $\frac{\partial^2 M_x}{\partial x^2} + 2\frac{\partial^2 M_{xy}}{\partial x \partial y} + \frac{\partial^2 M_y}{\partial y^2} + Q(x,y) = -\rho h(x,y) \omega^2 W$，通过移项，我们可将其写为统一的线性算子形式：
$$\mathcal{A} W(x,y) = Q(x,y)$$
具体而言，算子 $\mathcal{A}$ 可以分解为刚度算子 $\mathcal{K}$ 和质量算子 $\mathcal{M}$ 的组合：
$$(\mathcal{K} - \omega^2 \mathcal{M}) W(x,y) = Q(x,y)$$

其中，**质量算子** $\mathcal{M}$ 定义为：
$$\mathcal{M}[W] = \rho h(x,y) W$$

**刚度算子** $\mathcal{K}$ 定义为：
$$\mathcal{K}[W] = -\left( \frac{\partial^2 M_x(W)}{\partial x^2} + 2\frac{\partial^2 M_{xy}(W)}{\partial x \partial y} + \frac{\partial^2 M_y(W)}{\partial y^2} \right)$$
将其完全展开即为：
$$\mathcal{K}[W] = \frac{\partial^2}{\partial x^2} \left[ D \left( \frac{\partial^2 W}{\partial x^2} + \nu \frac{\partial^2 W}{\partial y^2} \right) \right] + 2 \frac{\partial^2}{\partial x \partial y} \left[ D (1-\nu) \frac{\partial^2 W}{\partial x \partial y} \right] + \frac{\partial^2}{\partial y^2} \left[ D \left( \frac{\partial^2 W}{\partial y^2} + \nu \frac{\partial^2 W}{\partial x^2} \right) \right]$$

### 5. 数值离散与矩阵求解形式：$\mathbf{A}\mathbf{w} = \mathbf{q}$
该方法自由边界条件较难处理，实际使用能量泛函方法

---

# 能量泛函

## 1. 连续系统的稳态能量泛函
对于频率为 $\omega$、激振力幅值为 $F(x,y)$ 的稳态振动，设其振幅场为 $W(x,y)$。系统的总能量泛函 $\Pi(W)$ 可表示为最大应变能 $U$、最大动能 $T$ 与外力虚功 $W_{ext}$ 之组合：

$$\Pi(W) = U(W) - T(W) - W_{ext}(W)$$

代入变厚度 $h(x,y)$ 和变刚度 $D(x,y)$ 参数，泛函的具体积分为：

$$\begin{aligned} \Pi(W) =& \frac{1}{2} \iint_{\Omega} D(x,y) \left[ (\nabla^2 W)^2 - 2(1-\nu)\left( \frac{\partial^2 W}{\partial x^2}\frac{\partial^2 W}{\partial y^2} - \left(\frac{\partial^2 W}{\partial x \partial y}\right)^2 \right) \right] dxdy \\ &- \frac{1}{2} \omega^2 \iint_{\Omega} \rho h(x,y) W^2 dxdy - \iint_{\Omega} F(x,y) W dxdy \end{aligned}$$

物理系统的真实振幅场，必然使得该能量泛函取极值或驻点，即一阶变分 $\delta \Pi = 0$。在此能量变分框架下，完全自由边界条件（法向弯矩和等效剪力为零）作为自然边界条件。

## 2. 空间离散与二次型泛函
对空间离散。利用基函数矩阵 $\mathbf{N}$ 将连续场 $W(x,y)$ 映射为有限维节点位移向量 $\mathbf{w}$：

$$W(x,y) = \mathbf{N}(x,y) \mathbf{w}$$

代入式 (2) 后，连续泛函 $\Pi(W)$ 退化为一个关于离散向量 $\mathbf{w}$ 的多维代数二次型函数：

$$\Pi(\mathbf{w}) = \frac{1}{2} \mathbf{w}^T \mathbf{K} \mathbf{w} - \frac{1}{2} \omega^2 \mathbf{w}^T \mathbf{M} \mathbf{w} - \mathbf{w}^T \mathbf{f}$$

其中，$\mathbf{K}$ 为总刚度矩阵，$\mathbf{M}$ 为总质量矩阵，$\mathbf{f}$ 为等效节点载荷向量。令动力刚度矩阵 $\mathbf{A} = \mathbf{K} - \omega^2 \mathbf{M}$，上式为：

$$\Pi(\mathbf{w}) = \frac{1}{2} \mathbf{w}^T \mathbf{A} \mathbf{w} - \mathbf{w}^T \mathbf{f}$$

## 3. 梯度寻优与 Hessian 矩阵求解
从最优化的视角，求解振幅场 $\mathbf{w}$ 等价于寻找多元函数 $\Pi(\mathbf{w})$ 的驻点。
令泛函对变量 $\mathbf{w}$ 的梯度为零：

$$\nabla_{\mathbf{w}} \Pi(\mathbf{w}) = \mathbf{A} \mathbf{w} - \mathbf{f} = \mathbf{0}$$

移项后即得到算法的最终求解方程：

$$\mathbf{A} \mathbf{w} = \mathbf{f}$$

进一步分析泛函的曲率张量，可得：

$$\mathbf{H} = \nabla_{\mathbf{w}}^2 \Pi(\mathbf{w}) = \mathbf{A}$$

线性方程组中的算子矩阵 $\mathbf{A}$，正是系统能量泛函的 Hessian 矩阵。本算法求解 $\mathbf{A}\mathbf{w}=\mathbf{f}$ 的过程，即是在能量流形上，沿梯度方向，得到驻点（鞍点）的过程。

---


## 伴随方法数学原理

在逆向设计中，我们需要优化具有 $M$ 个参数的物理系统。

### 1. 问题定义

假设物理系统的控制方程为线性方程组：
$$\mathbf{A}(\mathbf{p}) \mathbf{u} = \mathbf{f}$$
其中：
- $\mathbf{p} \in \mathbb{R}^M$ 是设计参数。
- $\mathbf{u} \in \mathbb{R}^N$ 是物理状态。
- $\mathbf{A}(\mathbf{p})$ ，刚度矩阵。
- $\mathbf{f}$ 是外部激励载荷。

目标是最小化loss：
$$\min_{\mathbf{p}} \mathcal{L}(\mathbf{u}, \mathbf{p})$$

### 2. 梯度

loss对参数 $\mathbf{p}$ 的全导数为：
$$\frac{d\mathcal{L}}{d\mathbf{p}} = \frac{\partial \mathcal{L}}{\partial \mathbf{p}} + \frac{\partial \mathcal{L}}{\partial \mathbf{u}} \frac{d\mathbf{u}}{d\mathbf{p}}$$

其中 $\frac{d\mathbf{u}}{d\mathbf{p}}$ 表示参数改变对物理状态的影响。对控制方程两边求导可得：
$$\frac{d\mathbf{K}}{d\mathbf{p}} \mathbf{u} + \mathbf{K} \frac{d\mathbf{u}}{d\mathbf{p}} = 0 \implies \frac{d\mathbf{u}}{d\mathbf{p}} = -\mathbf{K}^{-1} \left( \frac{d\mathbf{K}}{d\mathbf{p}} \mathbf{u} \right)$$

直接计算 $\frac{d\mathbf{u}}{d\mathbf{p}}$ 需要对 $M$ 个参数分别求解线性方程组，对于大规模参数这在计算上是不可接受的。

### 3. 伴随法推导

我们将 $\frac{d\mathbf{u}}{d\mathbf{p}}$ 代入全导数公式：
$$\frac{d\mathcal{L}}{d\mathbf{p}} = \frac{\partial \mathcal{L}}{\partial \mathbf{p}} - \underbrace{ \frac{\partial \mathcal{L}}{\partial \mathbf{u}} \mathbf{K}^{-1} }_{\boldsymbol{\lambda}^T} \left( \frac{d\mathbf{K}}{d\mathbf{p}} \mathbf{u} \right)$$

为了避免显式计算 $\mathbf{K}^{-1}$，引入 **伴随变量** $\boldsymbol{\lambda}$，满足：
$$\mathbf{K}^T \boldsymbol{\lambda} = \left( \frac{\partial \mathcal{L}}{\partial \mathbf{u}} \right)^T$$

解出 $\boldsymbol{\lambda}$，最终的梯度计算公式简化为：
$$\frac{d\mathcal{L}}{d\mathbf{p}} = \frac{\partial \mathcal{L}}{\partial \mathbf{p}} - \boldsymbol{\lambda}^T \left( \frac{\partial \mathbf{K}}{\partial \mathbf{p}} \mathbf{u} \right)$$

### 4. 算法优势

1. **计算效率**：无论设计参数 $\mathbf{p}$ 的维度有多大，计算完整梯度只需要 **2 次** 线性方程组求解（一次正向求 $\mathbf{u}$，一次反向求 $\boldsymbol{\lambda}$）。
2. **复用矩阵分解**：由于伴随方程的左端项是 $\mathbf{K}^T$（对于结构矩阵通常 $\mathbf{K} = \mathbf{K}^T$），我们可以复用正向求解时的 LU 或 Cholesky 分解结果，使反向求解几乎是瞬时完成的。
3. **内存节省**：不需要自动微分记录中间迭代过程，只需存储最终的状态向量。


---