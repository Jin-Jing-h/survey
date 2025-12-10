# 📚 Survey

| 名称 | 链接 | 年份 | 涉及的<br>领域 | 代码 | 创新点 | 不足点 |
|:----|:----|:----:|:--------------:|:----:|:------|:------|
| <small>FovEx: Human-Inspired Explanations for Vision Transformers and CNNs</small> | <small>[IJCV](https://arxiv.org/abs/2408.02123)</small> | <small>2025</small> | <small>视觉可解释性</small> | <small>[GitHub](https://github.com/mahadev1995/FovEx)</small> | <small>[概述](#fovex-ijcv-2025)</small> | <small>[概述](#fovex-ijcv-2025)</small> |
| <small>Rich Human Feedback for Text-to-Image Generation</small> | <small>[CVPR](https://arxiv.org/abs/2312.10240)</small> | <small>2024</small> | <small>文本生成图像</small> | <small>[GitHub](https://github.com/youweiliang/RichHF)</small> | <small>[概述](#rich-human-feedback-for-text-to-image-generation-cvpr-2024)</small> | <small>[概述](#rich-human-feedback-for-text-to-image-generation-cvpr-2024)</small> |
| <small>Mip-Splatting: Alias-free 3D Gaussian Splatting</small> | <small>[CVPR](https://arxiv.org/abs/2311.16493)</small> | <small>2024</small> | <small>3D 高斯/新视角</small> | <small>[GitHub](https://github.com/autonomousvision/mip-splatting)</small> | <small>[概述](#mip-splatting-cvpr-2024)</small> | <small>[概述](#mip-splatting-cvpr-2024)</small> |
| <small>BioCLIP: A Vision Foundation Model for the Tree of Life</small> | <small>[CVPR](https://arxiv.org/abs/2311.18803)</small> | <small>2024</small> | <small>生物视觉基础模型</small> | <small>[GitHub](https://github.com/Imageomics/bioclip)</small> | <small>[概述](#bioclip-cvpr-2024)</small> | <small>[概述](#bioclip-cvpr-2024)</small> |
| <small>SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow</small> | <small>[ECCV](https://arxiv.org/abs/2405.14793)</small> | <small>2024</small> | <small>光流估计</small> | <small>[GitHub](https://github.com/princeton-vl/SEA-RAFT)</small> | <small>[概述](#sea-raft-eccv-2024)</small> | <small>[概述](#sea-raft-eccv-2024)</small> |
| <small>PointLLM: Empowering LLMs to Understand Point Clouds</small> | <small>[ECCV](https://arxiv.org/abs/2308.16911)</small> | <small>2024</small> | <small>3D 多模态 + LLM</small> | <small>[GitHub](https://github.com/InternRobotics/PointLLM)</small> | <small>[概述](#pointllm-eccv-2024)</small> | <small>[概述](#pointllm-eccv-2024)</small> |
| <small>Generating Physically Stable and Buildable Brick Structures from Text</small> | <small>[ICCV](https://arxiv.org/abs/2505.05469)</small> | <small>2025</small> | <small>Text-to-3D + 物理</small> | <small>[GitHub](https://github.com/AvaLovelace1/BrickGPT)</small> | <small>[概述](#generating-physically-stable-and-buildable-brick-structures-from-text-iccv-2025)</small> | <small>[概述](#generating-physically-stable-and-buildable-brick-structures-from-text-iccv-2025)</small> |
| <small>Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction</small> | <small>[NeurIPS](https://arxiv.org/abs/2404.02905)</small> | <small>2024</small> | <small>视觉生成大模型</small> | <small>[GitHub](https://github.com/FoundationVision/VAR)</small> | <small>[概述](#visual-autoregressive-modeling-scalable-image-generation-via-next-scale-prediction-neurips-2024)</small> | <small>[概述](#visual-autoregressive-modeling-scalable-image-generation-via-next-scale-prediction-neurips-2024)</small> |
| <small>A Survey of Visual Transformers</small> | <small>[TNNLS](https://arxiv.org/abs/2111.06091)</small> | <small>2024*</small> | <small>视觉 Transformer 总览</small> | <small>[Awesome](https://github.com/liuyang-ict/awesome-visual-transformers)</small> | <small>[概述](#a-survey-of-visual-transformers-tnnls-2024)</small> | <small>[概述](#a-survey-of-visual-transformers-tnnls-2024)</small> |

---

## 📖 论文详细笔记

### FovEx （IJCV 2025）

**创新点：**
本文提出了 FovEx，这一结合类人凹视机制与梯度驱动扫视、可同时适用于 CNN 与 ViT 的统一 XAI 方法，在多项信赖度指标与人眼凝视一致性上优于现有方法

**不足点：**
优化目标偏向“保留”关键信息导致在 DELETE 指标上表现欠佳且仅在有限数据集与任务上验证，存在泛化性和人群偏置方面的潜在局限。

