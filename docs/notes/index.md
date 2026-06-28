# Notes

这里是我整理 AI 学习笔记的地方。它不是访谈记录，也不是论文摘要库，而是把读到的机制、边界和判断沉淀成以后能复用的 study notes。

当前有两个 topic：

| Topic | 关注点 |
| --- | --- |
| [AI Agent](ai-agent/index.md) | 语言模型如何通过工具、环境反馈和 runtime loop 行动。 |
| [Deep Learning](deep-learning/index.md) | 从神经网络、梯度下降、反向传播一路读到 Transformer 和注意力机制。 |

## 为什么这样写

每个 topic 先给概念地图，再把文章和论文笔记放进去。这样以后扩展新内容时，不会变成一串没有结构的页面。

写作时遵守四个约束：

1. 先写自己的理解，再补原文证据。
2. 不把对话逐字搬进笔记，只保留稳定结论。
3. 每篇文章只解决一个问题，避免把所有内容堆到首页。
4. 能用图解释的地方，优先用图、表、伪代码或 Mermaid 替代长段文字。

## 基本组件

| 组件 | 用途 |
| --- | --- |
| 读法 | 统一每篇论文的阅读输出形状，避免越读越散。 |
| Topic | 一条长期主线，例如 AI Agent 或 Deep Learning。 |
| Article | 解释一个稳定概念，例如反向传播、注意力机制、provider boundary。 |
| Paper | 记录一篇论文的核心问题、机制和边界。 |
| 原文材料 | PDF 保存在 `docs/paper/`，页面中只摘关键原文抓手和自己的理解。 |

## 入门起点

如果第一次打开这个站点，按这个顺序读：

1. [快速开始](reading-route.md)：了解笔记怎样从论文里抽取问题、机制和边界。
2. [AI Agent Topic](ai-agent/index.md)：理解 agent loop、工具和环境反馈。
3. [Deep Learning Topic](deep-learning/index.md)：从第一性原理读到 Transformer。

## 路径选择

| 如果你想要... | 接下来读 |
| --- | --- |
| 快速知道站点怎么读 | [快速开始](reading-route.md) |
| 进入 AI Agent 主线 | [AI Agent Topic](ai-agent/index.md) |
| 理解 ReAct、CoT、Act-only 的关系 | [Agent 基本概念](topics/agent-basics.md) |
| 读 ReAct 论文的核心贡献 | [ReAct：推理和行动交替](papers/react.md) |
| 读 SWE-agent 论文的接口贡献 | [SWE-agent：给语言模型设计电脑界面](papers/swe-agent.md) |
| 从神经网络一路读到 Transformer | [Deep Learning Topic](deep-learning/index.md) |
| 判断换模型时哪些代码该动 | [实现与 provider 边界](topics/implementation-boundaries.md) |
| 理解 memory、session、compact 的区别 | [长期记忆](topics/long-term-memory.md) |

这个站点以后会继续扩展。新的页面会优先挂到某个 topic 下，而不是直接堆在全局导航里。
