# 📚 Survey

| 名称 | 链接 | 年份 | 涉及的<br>领域 | 代码 | 创新点 | 不足点 |
|:----|:----|:----:|:--------------:|:----:|:------|:------|
| <small>FovEx: Human-Inspired Explanations for Vision Transformers and CNNs</small> | <small>[IJCV](https://arxiv.org/abs/2408.02123)</small> | <small>2025</small> | <small>视觉可解释性</small> | <small>[GitHub](https://github.com/mahadev1995/FovEx)</small> | <small>[概述](#fovex-ijcv-2025)</small> | <small>[概述](#fovex-ijcv-2025)</small> |
| <small>MaIR: A Locality- and Continuity-Preserving Mamba for Image Restoration</small> | <small>[CVPR](https://arxiv.org/abs/2412.20066)</small> | <small>2025</small> | <small>通用图像恢复 </small> | <small>[GitHub](https://github.com/XLearning-SCU/2025-CVPR-MaIR)</small> | <small>[概述](#mair-cvpr-2025)</small> | <small>[概述](#mair-cvpr-2025)</small> |
| <small>Visual-Instructed Degradation Diffusion for All-in-One Image Restoration</small> | <small>[CVPR](https://openaccess.thecvf.com/content/CVPR2025/papers/Luo_Visual-Instructed_Degradation_Diffusion_for_All-in-One_Image_Restoration_CVPR_2025_paper.pdf)</small> | <small>2025</small> | <small>一体化图像恢复</small> | <small>[GitHub](https://github.com/luowyang/Defusion)</small> | <small>[概述](#defusion-cvpr-2025)</small> | <small>[概述](#defusion-cvpr-2025)</small> |
| <small>DarkIR: Robust Low-Light Image Restoration</small> | <small>[arXiv](https://arxiv.org/abs/2412.13443)</small> | <small>2025</small> | <small>低照度图像恢复</small> | <small>[GitHub](https://github.com/cidautai/DarkIR)</small> | <small>[概述](#darkir-cvpr-2025)</small> | <small>[概述](#darkir-cvpr-2025)</small> |
| <small>FaithDiff: Unleashing Diffusion Priors for Faithful Image Super-resolution</small> | <small>[CVPR](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_FaithDiff_Unleashing_Diffusion_Priors_for_Faithful_Image_Super-resolution_CVPR_2025_paper.pdf)</small> | <small>2025</small> | <small>图像超分辨率</small> | <small>[GitHub](https://github.com/JyChen9811/FaithDiff)</small> | <small>[概述](#faithdiff-cvpr-2025)</small> | <small>[概述](#faithdiff-cvpr-2025)</small> |
| <small>Multimodal Autoregressive Pre-training of Large Vision Encoders (AIMv2)</small> | <small>[CVPR](https://openaccess.thecvf.com/content/CVPR2025/papers/Fini_Multimodal_Autoregressive_Pre-training_of_Large_Vision_Encoders_CVPR_2025_paper.pdf)</small> | <small>2025</small> | <small>多模态视觉预训练</small> | <small>[GitHub](https://github.com/apple/ml-aim)</small> | <small>[概述](#aimv2-cvpr-2025)</small> | <small>[概述](#aimv2-cvpr-2025)</small> |
| <small>Task Preference Optimization: Improving Multimodal Large Language Models with Vision Task Alignment</small> | <small>[CVPR](https://openaccess.thecvf.com/content/CVPR2025/html/Yan_Task_Preference_Optimization_Improving_Multimodal_Large_Language_Models_with_Vision_CVPR_2025_paper.html)</small> | <small>2025</small> | <small>MLLM 视觉任务对齐</small> | <small>[GitHub](https://github.com/OpenGVLab/TPO)</small> | <small>[概述](#tpo-cvpr-2025)</small> | <small>[概述](#tpo-cvpr-2025)</small> |
| <small>Multi-Layer Visual Feature Fusion in Multimodal LLMs: Methods, Analysis, and Best Practices</small> | <small>[论文页](https://www.researchgate.net/publication/394621062_Multi-Layer_Visual_Feature_Fusion_in_Multimodal_LLMs_Methods_Analysis_and_Best_Practices)</small> | <small>2025</small> | <small>多层视觉特征融合</small> | <small>[GitHub](https://github.com/EIT-NLP/Layer_Select_Fuse_for_MLLM)</small> | <small>[概述](#multi-layer-fusion-cvpr-2025)</small> | <small>[概述](#multi-layer-fusion-cvpr-2025)</small> |
---

## 📖 论文详细笔记
<a id="fovex-ijcv-2025"></a>
### FovEx （IJCV 2025）

**创新点：**
本文提出了 FovEx，这一结合类人凹视机制与梯度驱动扫视、可同时适用于 CNN 与 ViT 的统一 XAI 方法，在多项信赖度指标与人眼凝视一致性上优于现有方法

**不足点：**
优化目标偏向“保留”关键信息导致在 DELETE 指标上表现欠佳且仅在有限数据集与任务上验证，存在泛化性和人群偏置方面的潜在局限。
<a id="mair-cvpr-2025"></a>
### MaIR: A Locality- and Continuity-Preserving Mamba for Image Restoration（CVPR 2025）

**创新点：**
MaIR 提出在 Mamba 状态空间模型里加入 Nested S-shaped Scanning（NSS）+ Sequence Shuffle Attention（SSA），同时保持图像的局部性和空间连续性，相比以往简单按行/列展平成 1D 序列的 Mamba 方案，在超分、去噪、去模糊、去雾等 4 大任务、14 个数据集上全面刷了 40 个基线。

**不足点：**
主要还是在配对的标准低层视觉基准上做实验，对真实未知退化（JPEG 压缩、相机 ISP、混合退化等）的验证有限，而且 Mamba 结构本身比较复杂，训练成本和工程落地门槛都不算低。
