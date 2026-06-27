# 反向传播算法

反向传播不是另一个神秘训练算法，它是链式法则在计算图上的高效组织方式。

训练时发生两件事：

1. 前向传播：从输入一路算到损失。
2. 反向传播：从损失倒着走，计算每个参数对损失的影响。

梯度下降用这些梯度更新参数。也就是说，反向传播负责“算梯度”，梯度下降负责“用梯度”。

## 一个最小计算图

<svg class="dl-figure" viewBox="0 0 920 360" role="img" aria-labelledby="bp-title">
  <title id="bp-title">前向传播和反向传播计算图</title>
  <defs>
    <marker id="arrow-bp" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#2f5f9f"></path>
    </marker>
    <marker id="arrow-bp-back" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#b35c23"></path>
    </marker>
  </defs>
  <rect x="60" y="105" width="100" height="60" rx="8" class="dl-box"></rect>
  <text x="110" y="142" text-anchor="middle" class="dl-label">x</text>
  <rect x="60" y="205" width="100" height="60" rx="8" class="dl-box"></rect>
  <text x="110" y="242" text-anchor="middle" class="dl-label">w, b</text>
  <rect x="290" y="150" width="150" height="80" rx="8" class="dl-box-accent"></rect>
  <text x="365" y="184" text-anchor="middle" class="dl-label">pred</text>
  <text x="365" y="210" text-anchor="middle" class="dl-small">w x + b</text>
  <rect x="570" y="150" width="150" height="80" rx="8" class="dl-box"></rect>
  <text x="645" y="184" text-anchor="middle" class="dl-label">loss</text>
  <text x="645" y="210" text-anchor="middle" class="dl-small">(pred - y)^2</text>
  <line x1="162" y1="135" x2="280" y2="178" class="dl-arrow" marker-end="url(#arrow-bp)"></line>
  <line x1="162" y1="235" x2="280" y2="202" class="dl-arrow" marker-end="url(#arrow-bp)"></line>
  <line x1="448" y1="190" x2="560" y2="190" class="dl-arrow" marker-end="url(#arrow-bp)"></line>
  <path d="M570 245 C450 315, 260 305, 160 255" fill="none" class="dl-back-arrow" marker-end="url(#arrow-bp-back)"></path>
  <path d="M570 135 C450 65, 270 70, 165 118" fill="none" class="dl-back-arrow" marker-end="url(#arrow-bp-back)"></path>
  <text x="385" y="50" class="dl-small dl-blue">前向：计算值</text>
  <text x="385" y="330" class="dl-small dl-orange">反向：传回梯度</text>
</svg>

## 观察 PyTorch 自动梯度

```python
import torch


def inspect_backward():
    """演示 PyTorch 如何通过反向传播计算 w 和 b 的梯度。"""
    x = torch.tensor([2.0])
    y = torch.tensor([5.0])

    w = torch.tensor([1.0], requires_grad=True)
    b = torch.tensor([0.0], requires_grad=True)

    prediction = w * x + b
    loss = (prediction - y) ** 2

    loss.backward()

    print('prediction:', float(prediction))
    print('loss:', float(loss))
    print('w.grad:', float(w.grad))
    print('b.grad:', float(b.grad))


inspect_backward()
```

这段代码中，`prediction = 2`，目标 `y = 5`，损失是 `9`。反向传播会算出：如果微调 `w` 或 `b`，损失会怎样变化。

## 反向传播缓存了什么

| 节点 | 前向时的值 | 反向时需要的东西 |
| --- | --- | --- |
| `prediction = w * x + b` | 预测值 | `x`、`w`、上游梯度 |
| `loss = (prediction - y) ** 2` | 损失值 | `prediction - y` |
| 参数 `w` | 当前权重 | `loss` 对 `w` 的梯度 |
| 参数 `b` | 当前偏置 | `loss` 对 `b` 的梯度 |

PyTorch 的 autograd 会记录这些计算关系。调用 `backward()` 时，它沿着图反向执行局部梯度计算，并按链式法则把它们乘起来。

## 和梯度下降的关系

```text
forward      : 用当前参数得到预测和损失
backward     : 计算每个参数的梯度
optimizer.step: 用梯度更新参数
zero_grad    : 清掉旧梯度，准备下一轮
```

## 小结

- 反向传播是链式法则在计算图上的实现。
- 梯度是“损失对参数的敏感度”。
- `backward()` 后，梯度保存在参数的 `.grad` 里。
- 梯度默认会累加，所以训练循环里要清空梯度。
