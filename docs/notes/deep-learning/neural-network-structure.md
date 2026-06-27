# 神经网络的结构

神经网络可以先理解成一串函数组合：

```text
输入 x -> 线性变换 -> 非线性激活 -> 线性变换 -> 输出
```

一层网络负责把输入换到新的表示空间，多层网络把这种变换重复多次。参数就是网络要学习的权重和偏置；激活函数让这些层组合后不再只是一个大的线性函数。

## 从单个神经元开始

<svg class="dl-figure" viewBox="0 0 920 260" role="img" aria-labelledby="neuron-title">
  <title id="neuron-title">单个神经元的计算流程</title>
  <defs>
    <marker id="arrow-neuron" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#2f5f9f"></path>
    </marker>
  </defs>
  <circle cx="95" cy="85" r="32" class="dl-node"></circle>
  <text x="95" y="91" text-anchor="middle" class="dl-label">x1</text>
  <circle cx="95" cy="175" r="32" class="dl-node"></circle>
  <text x="95" y="181" text-anchor="middle" class="dl-label">x2</text>
  <line x1="130" y1="85" x2="315" y2="120" class="dl-arrow" marker-end="url(#arrow-neuron)"></line>
  <line x1="130" y1="175" x2="315" y2="140" class="dl-arrow" marker-end="url(#arrow-neuron)"></line>
  <text x="218" y="84" class="dl-small">乘以 w1</text>
  <text x="218" y="184" class="dl-small">乘以 w2</text>
  <rect x="330" y="82" width="220" height="96" rx="8" class="dl-box"></rect>
  <text x="440" y="122" text-anchor="middle" class="dl-label">w1 x1 + w2 x2 + b</text>
  <text x="440" y="150" text-anchor="middle" class="dl-small">线性组合</text>
  <line x1="560" y1="130" x2="665" y2="130" class="dl-arrow" marker-end="url(#arrow-neuron)"></line>
  <rect x="680" y="82" width="150" height="96" rx="8" class="dl-box-accent"></rect>
  <text x="755" y="122" text-anchor="middle" class="dl-label">activation</text>
  <text x="755" y="150" text-anchor="middle" class="dl-small">非线性</text>
</svg>

如果没有激活函数，多层线性层叠起来仍然等价于一层线性层。ReLU、Sigmoid 这类激活函数的作用，是让模型能表达弯曲边界和更复杂的模式。

## 一个最小 MLP

```python
import torch
from torch import nn


class TinyMLP(nn.Module):
    """一个最小两层神经网络，用来演示线性层和激活函数如何组合。"""

    def __init__(self):
        """初始化网络结构：二维输入、八个隐藏单元、一个二分类输出。"""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """接收二维点，输出每个点属于正类的概率。"""
        return self.net(x)


model = TinyMLP()
points = torch.tensor([[0.0, 0.0], [1.0, 1.0], [1.0, -1.0]])
probabilities = model(points)
print(probabilities)
```

这个模型的输入是二维点，输出是一个 0 到 1 之间的概率。此时它还没有训练，所以输出只是随机初始化参数下的结果。

## 层结构

<svg class="dl-figure" viewBox="0 0 920 360" role="img" aria-labelledby="mlp-title">
  <title id="mlp-title">两层 MLP 的结构</title>
  <defs>
    <marker id="arrow-mlp" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#2f5f9f"></path>
    </marker>
  </defs>
  <g class="dl-layer">
    <circle cx="90" cy="130" r="24" class="dl-node"></circle>
    <circle cx="90" cy="220" r="24" class="dl-node"></circle>
    <text x="90" y="70" text-anchor="middle" class="dl-label">输入层</text>
    <text x="90" y="135" text-anchor="middle" class="dl-small">x1</text>
    <text x="90" y="225" text-anchor="middle" class="dl-small">x2</text>
  </g>
  <g class="dl-layer">
    <circle cx="330" cy="80" r="24" class="dl-node-accent"></circle>
    <circle cx="330" cy="140" r="24" class="dl-node-accent"></circle>
    <circle cx="330" cy="200" r="24" class="dl-node-accent"></circle>
    <circle cx="330" cy="260" r="24" class="dl-node-accent"></circle>
    <text x="330" y="35" text-anchor="middle" class="dl-label">隐藏层</text>
  </g>
  <g class="dl-layer">
    <circle cx="610" cy="170" r="28" class="dl-node"></circle>
    <text x="610" y="70" text-anchor="middle" class="dl-label">输出层</text>
    <text x="610" y="176" text-anchor="middle" class="dl-small">p</text>
  </g>
  <path d="M115 130 C185 90, 245 80, 305 80" class="dl-thin"></path>
  <path d="M115 130 C185 125, 245 140, 305 140" class="dl-thin"></path>
  <path d="M115 130 C185 160, 245 200, 305 200" class="dl-thin"></path>
  <path d="M115 130 C185 205, 245 260, 305 260" class="dl-thin"></path>
  <path d="M115 220 C185 105, 245 80, 305 80" class="dl-thin"></path>
  <path d="M115 220 C185 155, 245 140, 305 140" class="dl-thin"></path>
  <path d="M115 220 C185 205, 245 200, 305 200" class="dl-thin"></path>
  <path d="M115 220 C185 245, 245 260, 305 260" class="dl-thin"></path>
  <path d="M355 80 C435 110, 515 150, 582 170" class="dl-thin"></path>
  <path d="M355 140 C435 150, 515 164, 582 170" class="dl-thin"></path>
  <path d="M355 200 C435 190, 515 176, 582 170" class="dl-thin"></path>
  <path d="M355 260 C435 235, 515 195, 582 170" class="dl-thin"></path>
  <line x1="675" y1="170" x2="815" y2="170" class="dl-arrow" marker-end="url(#arrow-mlp)"></line>
  <text x="840" y="176" class="dl-label">类别概率</text>
</svg>

## 张量形状

| 位置 | 张量形状 | 含义 |
| --- | --- | --- |
| 输入 `x` | `[batch, 2]` | 每个样本有两个特征。 |
| 第一层输出 | `[batch, 8]` | 每个样本被映射成 8 个隐藏特征。 |
| 第二层输出 | `[batch, 1]` | 每个样本得到一个分类分数。 |
| Sigmoid 后 | `[batch, 1]` | 分数被压到 0 到 1 之间。 |

## 小结

- 神经网络是函数组合，不是手写规则集合。
- `nn.Linear` 保存权重和偏置，`ReLU` 引入非线性。
- 训练前的网络只是随机函数；训练会通过损失和梯度调整参数。
