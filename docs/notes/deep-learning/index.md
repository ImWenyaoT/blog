# Deep Learning 专题导读

这个专题从最小的神经网络开始，一路走到 Transformer 和注意力机制。目标不是堆公式，而是让每个概念都能对应到一个简单 PyTorch 代码块、一张图和一个可复述的直觉。

阅读顺序固定为五章：

1. [神经网络的结构](neural-network-structure.md)：神经网络是一串带参数的函数组合。
2. [梯度下降法](gradient-descent.md)：训练是在损失曲面上调整参数。
3. [反向传播算法](backpropagation.md)：反向传播负责高效计算每个参数的梯度。
4. [GPT 是什么？直观讲解 Transformer](gpt-transformer.md)：GPT 用 Transformer 做下一个 token 预测。
5. [直观解释注意力机制，Transformer 的核心](attention.md)：注意力是带权重的信息汇总。

## 读法

每章都按同一条路径组织：

| 部分 | 作用 |
| --- | --- |
| 直觉 | 先解释这个机制解决什么问题。 |
| 图解 | 用朴素图形把数据流、参数流或梯度流画出来。 |
| PyTorch 代码 | 保留一个最小可运行骨架，代码尽量短。 |
| 形状表 | 把张量维度写清楚，减少“能跑但不理解”。 |
| 边界 | 说明这个简化版本没有覆盖什么。 |

## 一条主线

深度学习可以先看成三件事：

<svg class="dl-figure" viewBox="0 0 920 210" role="img" aria-labelledby="dl-overview-title">
  <title id="dl-overview-title">深度学习训练主线</title>
  <defs>
    <marker id="arrow-overview" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#2f5f9f"></path>
    </marker>
  </defs>
  <rect x="30" y="55" width="190" height="90" rx="8" class="dl-box"></rect>
  <text x="125" y="92" text-anchor="middle" class="dl-label">模型</text>
  <text x="125" y="120" text-anchor="middle" class="dl-small">带参数的函数</text>
  <line x1="230" y1="100" x2="335" y2="100" class="dl-arrow" marker-end="url(#arrow-overview)"></line>
  <rect x="350" y="55" width="190" height="90" rx="8" class="dl-box"></rect>
  <text x="445" y="92" text-anchor="middle" class="dl-label">损失</text>
  <text x="445" y="120" text-anchor="middle" class="dl-small">当前错得多离谱</text>
  <line x1="550" y1="100" x2="655" y2="100" class="dl-arrow" marker-end="url(#arrow-overview)"></line>
  <rect x="670" y="55" width="190" height="90" rx="8" class="dl-box"></rect>
  <text x="765" y="92" text-anchor="middle" class="dl-label">更新</text>
  <text x="765" y="120" text-anchor="middle" class="dl-small">沿梯度反方向调参数</text>
  <path d="M760 154 C730 190, 180 190, 130 154" fill="none" class="dl-muted-arrow" marker-end="url(#arrow-overview)"></path>
  <text x="445" y="190" text-anchor="middle" class="dl-small">重复很多次，模型逐渐变好</text>
</svg>

Transformer 只是这条主线上的一种强大模型结构。先把“参数如何学习”讲清楚，再看注意力机制，理解成本会低很多。

## 参考资料

这个专题参考了 [D2L 动手学深度学习](https://zh.d2l.ai/index.html) 的学习顺序，也参考了 Wikipedia 对 neural network、gradient descent、backpropagation、Transformer 和 attention 的定义性描述。正文是面向这个博客重新组织的学习笔记，不是原文摘录。
