

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

### 2. 梯度的困境

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