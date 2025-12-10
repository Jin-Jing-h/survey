# 📚 Survey

| 名称 | 链接 | 年份 | 涉及的<br>领域 | 代码 | 创新点 | 不足点 |
|:----|:----|:----:|:--------------:|:----:|:------|:------|
| <small>FovEx: Human-Inspired Explanations for Vision Transformers and CNNs</small> | <small>[IJCV](https://arxiv.org/abs/2408.02123)</small> | <small>2025</small> | <small>视觉可解释性</small> | <small>[GitHub](https://github.com/mahadev1995/FovEx)</small> | <small>[概述](#fovex-ijcv-2025)</small> | <small>[概述](#fovex-ijcv-2025)</small> |
| <small>BioCLIP: A Vision Foundation Model for the Tree of Life</small> | <small>[CVPR](https://arxiv.org/abs/2311.18803)</small> | <small>2024</small> | <small>生物视觉基础模型</small> | <small>[GitHub](https://github.com/Imageomics/bioclip)</small> | <small>[概述](#bioclip-cvpr-2024)</small> | <small>[概述](#bioclip-cvpr-2024)</small> |
| <small>MaIR: A Locality- and Continuity-Preserving Mamba for Image Restoration</small> | <small>[arXiv](https://arxiv.org/abs/2412.20066)</small> | <small>2025</small> | <small>通用图像恢复 </small> | <small>[GitHub](https://github.com/XLearning-SCU/2025-CVPR-MaIR)</small> | <small>[概述](#mair-cvpr-2025)</small> | <small>[概述](#mair-cvpr-2025)</small> |
| <small>Visual-Instructed Degradation Diffusion for All-in-One Image Restoration (Defusion)</small> | <small>[CVPR](https://openaccess.thecvf.com/content/CVPR2025/papers/Luo_Visual-Instructed_Degradation_Diffusion_for_All-in-One_Image_Restoration_CVPR_2025_paper.pdf)</small> | <small>2025</small> | <small>一体化图像恢复</small> | <small>[GitHub](https://github.com/luowyang/Defusion)</small> | <small>[概述](#defusion-cvpr-2025)</small> | <small>[概述](#defusion-cvpr-2025)</small> |
| <small>DarkIR: Robust Low-Light Image Restoration</small> | <small>[arXiv](https://arxiv.org/abs/2412.13443)</small> | <small>2025</small> | <small>低照度图像恢复</small> | <small>[GitHub](https://github.com/cidautai/DarkIR)</small> | <small>[概述](#darkir-cvpr-2025)</small> | <small>[概述](#darkir-cvpr-2025)</small> |
| <small>FaithDiff: Unleashing Diffusion Priors for Faithful Image Super-resolution</small> | <small>[CVPR](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_FaithDiff_Unleashing_Diffusion_Priors_for_Faithful_Image_Super-resolution_CVPR_2025_paper.pdf)</small> | <small>2025</small> | <small>图像超分辨率</small> | <small>[GitHub](https://github.com/JyChen9811/FaithDiff)</small> | <small>[概述](#faithdiff-cvpr-2025)</small> | <small>[概述](#faithdiff-cvpr-2025)</small> |

---

## 📖 论文详细笔记
<a id="fovex-ijcv-2025"></a>
### FovEx （IJCV 2025）

**创新点：**
本文提出了 FovEx，这一结合类人凹视机制与梯度驱动扫视、可同时适用于 CNN 与 ViT 的统一 XAI 方法，在多项信赖度指标与人眼凝视一致性上优于现有方法

**不足点：**
优化目标偏向“保留”关键信息导致在 DELETE 指标上表现欠佳且仅在有限数据集与任务上验证，存在泛化性和人群偏置方面的潜在局限。
<a id="bioclip-cvpr-2024"></a>
### BioCLIP: A Vision Foundation Model for the Tree of Life（CVPR 2024）

**创新点：**
构建 TreeOfLife-10M，目前规模最大、按生物分类树组织的图像数据集之一。

训练 BioCLIP，利用图像 + 结构化生物分类信息，使模型自带“层级表示”，在各类细粒度生物识别任务上大幅超越通用 CLIP。

说明了“领域专用 foundation model”路线在科学场景（生物多样性、生态监测）中的巨大价值。

**不足点：**
目前只处理静态图像 + 物种标签，对行为、时序、生态位等更复杂生物信息仍然无能为力。

模型和数据规模很大，生物学家团队想自己微调或重训，算力门槛较高。

过于依赖现有分类体系，对分类体系不完善或争议的物种，表现和可解释性都有不确定性
