# 📚 Survey

| 名称 | 链接 | 年份 | 涉及的领域 | 代码 | 创新点 | 不足点 |
|:----|:----|:----:|:--------------:|:----:|:------|:------|
| <small>FovEx: Human-Inspired Explanations for Vision Transformers and CNNs</small> | <small>[IJCV](https://arxiv.org/abs/2408.02123)</small> | <small>2025</small> | <small>视觉可解释性</small> | <small>[GitHub](https://github.com/mahadev1995/FovEx)</small> | <small>[概述](#fovex-ijcv-2025)</small> | <small>[概述](#fovex-ijcv-2025)</small> |
| <small>MaIR: A Locality- and Continuity-Preserving Mamba for Image Restoration</small> | <small>[CVPR](https://arxiv.org/abs/2412.20066)</small> | <small>2025</small> | <small>通用图像恢复 </small> | <small>[GitHub](https://github.com/XLearning-SCU/2025-CVPR-MaIR)</small> | <small>[概述](#mair-cvpr-2025)</small> | <small>[概述](#mair-cvpr-2025)</small> |
| <small>Visual-Instructed Degradation Diffusion for All-in-One Image Restoration</small> | <small>[CVPR](https://openaccess.thecvf.com/content/CVPR2025/papers/Luo_Visual-Instructed_Degradation_Diffusion_for_All-in-One_Image_Restoration_CVPR_2025_paper.pdf)</small> | <small>2025</small> | <small>一体化图像恢复</small> | <small>[GitHub](https://github.com/luowyang/Defusion)</small> | <small>[概述](#defusion-cvpr-2025)</small> | <small>[概述](#defusion-cvpr-2025)</small> |
| <small>DarkIR: Robust Low-Light Image Restoration</small> | <small>[CVPR](https://arxiv.org/abs/2412.13443)</small> | <small>2025</small> | <small>低照度图像恢复</small> | <small>[GitHub](https://github.com/cidautai/DarkIR)</small> | <small>[概述](#darkir-cvpr-2025)</small> | <small>[概述](#darkir-cvpr-2025)</small> |
| <small>FaithDiff: Unleashing Diffusion Priors for Faithful Image Super-resolution</small> | <small>[CVPR](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_FaithDiff_Unleashing_Diffusion_Priors_for_Faithful_Image_Super-resolution_CVPR_2025_paper.pdf)</small> | <small>2025</small> | <small>图像超分辨率</small> | <small>[GitHub](https://github.com/JyChen9811/FaithDiff)</small> | <small>[概述](#faithdiff-cvpr-2025)</small> | <small>[概述](#faithdiff-cvpr-2025)</small> |
| <small>GEM: A Generalizable Ego-Vision Multimodal World Model for Fine-Grained Ego-Motion, Object Dynamics, and Scene Composition Control</small> | <small>[CVPR](https://arxiv.org/abs/2412.11198)</small> | <small>2024</small> | <small>多模态世界模型<br>RGB+Depth+Pose</small> | <small>[GitHub](https://github.com/vita-epfl/GEM)</small> | <small>[概述](#gem-cvpr-2025)</small> | <small>[概述](#gem-cvpr-2025)</small> |
| <small>ProtoDepth: Unsupervised Continual Depth Completion with Prototypes</small> | <small>[CVPR](https://arxiv.org/abs/2503.12745)</small> | <small>2025</small> | <small>RGB+点云<br>深度补全</small> | <small>[GitHub](https://github.com/patrickqrim/ProtoDepth)</small> | <small>[概述](#protodepth-cvpr-2025)</small> | <small>[概述](#protodepth-cvpr-2025)</small> |
| <small>Dynamic Derivation and Elimination: Audio Visual Segmentation with Enhanced Audio Semantics (DDESeg)</small> | <small>[CVPR](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Dynamic_Derivation_and_Elimination_Audio_Visual_Segmentation_with_Enhanced_Audio_CVPR_2025_paper.html)</small> | <small>2025</small> | <small>音视频<br>目标分割</small> | <small>[GitHub](https://github.com/YenanLiu/DDESeg)</small> | <small>[概述](#ddeseg-cvpr-2025)</small> | <small>[概述](#ddeseg-cvpr-2025)</small> |
| <small>MMAudio: Taming Multimodal Joint Training for High-Quality Video-to-Audio Synthesis</small> | <small>[CVPR](https://arxiv.org/abs/2412.15322)</small> | <small>2025</small> | <small>视频→音频<br>视听生成</small> | <small>[GitHub](https://github.com/hkchengrex/MMAudio)</small> | <small>[概述](#mmaudio-cvpr-2025)</small> | <small>[概述](#mmaudio-cvpr-2025)</small> |
| <small>MulFS-CAP: Multimodal Fusion-Supervised Cross-Modality Alignment Perception for Unregistered Infrared-Visible Image Fusion</small> | <small>[TPAMI](https://doi.org/10.1109/TPAMI.2025.3535617)</small> | <small>2025</small> | <small>红外+可见光<br>图像融合</small> | <small>[GitHub](https://github.com/YR0211/MulFS-CAP)</small> | <small>[概述](#mulfs-cap-tpami-2025)</small> | <small>[概述](#mulfs-cap-tpami-2025)</small> |


---

## 📖 论文详细笔记
<a id="fovex-ijcv-2025"></a>
### FovEx: Human-Inspired Explanations for Vision Transformers and CNNs （IJCV 2025）

**数据集：**  
[ImageNet-1K](https://www.image-net.org/challenges/LSVRC/)  
[MIT1003](https://people.csail.mit.edu/tjudd/WherePeopleLook/)


**创新点：**
本文提出了 FovEx，这一结合类人凹视机制与梯度驱动扫视、可同时适用于 CNN 与 ViT 的统一 XAI 方法，在多项信赖度指标与人眼凝视一致性上优于现有方法

**不足点：**
优化目标偏向“保留”关键信息导致在 DELETE 指标上表现欠佳且仅在有限数据集与任务上验证，存在泛化性和人群偏置方面的潜在局限。

<a id="mair-cvpr-2025"></a>
### MaIR: A Locality- and Continuity-Preserving Mamba for Image Restoration（CVPR 2025）

**数据集：**  
[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)  
[Flickr2K](https://www.kaggle.com/datasets/daehoyang/flickr2k)  
[WED](https://ivc.uwaterloo.ca/database/WaterlooExploration/)  
[BSD400 / BSD68](https://github.com/clausmichele/CBSD68-dataset)  
[Kodak24](https://r0k.us/graphics/kodak/)  
[McMaster](https://www4.comp.polyu.edu.hk/~cslzhang/CDM_Dataset.htm)  
[Urban100](https://huggingface.co/datasets/eugenesiow/Urban100)  
[SIDD-Medium](https://abdokamel.github.io/sidd/)  
[GoPro](https://seungjunnah.github.io/Datasets/gopro.html)  
[HIDE](https://github.com/joanshen0508/HA_deblur)  
[RESIDE (ITS/OTS/SOTS)](https://sites.google.com/view/reside-dehaze-datasets/reside-standard)  
[RESIDE-6K](https://gts.ai/dataset-download/reside-6k/)

**创新点：**
MaIR 提出在 Mamba 状态空间模型里加入 Nested S-shaped Scanning（NSS）+ Sequence Shuffle Attention（SSA），同时保持图像的局部性和空间连续性，相比以往简单按行或者列展平成 1D 序列的 Mamba 方案，在超分、去噪、去模糊、去雾等 4 大任务、14 个数据集上全面刷了 40 个基线。

**不足点：**
文中没有提到

<a id="defusion-cvpr-2025"></a>
### Visual-Instructed Degradation Diffusion for All-in-One Image Restoration（CVPR 2025）

**创新点：**
Defusion把all-in-one 图像恢复做成一个视觉指令驱动的退化扩散，不是用模糊的文本prompt，而是构造与不同退化（去噪、去模糊、去雾、低照度等）显式对齐的视觉指令图，作为条件去引导扩散模型，对未知退化场景也能统一建模，在多种一体化恢复基准上达到了新的SOTA。

**不足点：**
文中没有提到

<a id="darkir-cvpr-2025"></a>
### DarkIR: Robust Low-Light Image Restoration（CVPR 2024）

**创新点：**
DarkIR 针对夜景/低照环境下同时存在的 噪声 + 低照度 + 运动模糊，在高效 CNN 上设计新的注意力机制扩展感受野，构建了一个统一的多任务低照度恢复网络，在 LOLBlur、LOLv2、Real-LOLBlur 等数据集上刷新 SOTA，并在 NTIRE 2025 低照度挑战中获得最佳方法，同时保持参数量和 MAC 数显著低于大多数 Transformer 模型。

**不足点：**
DarkIR 的主要局限在于：虽然通过大量使用 depth-wise 卷积显著降低了参数量和 MACs，但作者在 Limitations 中明确指出，这类算子在实际 GPU 上算术强度较低、对硬件不够友好，因此推理时间并不会随着计算量成比例下降；此外，多任务 all-in-one 版本在获得更强泛化能力的同时，在 LOLBlur 等数据集上仍存在约 0.5 dB 的轻微性能损失，说明在统一建模多种低照退化时仍面临精度与泛化、效率之间的折中。

<a id="faithdiff-cvpr-2025"></a>
### FaithDiff: Unleashing Diffusion Priors for Faithful Image Super-resolution（CVPR 2025）

**创新点：**
FaithDiff 针对“既要好看又要保真”的真实场景超分问题，提出在 latent diffusion 上加入 特征对齐模块 + 编码器与扩散模型的联合微调，显式对齐退化输入特征与扩散噪声空间，让大模型的先验既能生成细节又不过度幻觉，在多种 SR 基准上对结构保持和视觉质量都明显优于以往扩散式 SR 方法

**不足点：**
文中没有提到

<a id="gem-cvpr-2025"></a>
### GEM: A Generalizable Ego-Vision Multimodal World Model for Fine-Grained Ego-Motion, Object Dynamics, and Scene Composition Control（CVPR 2025）

**创新点：**
提出 GEM 这一统一的自监督多模态世界模型，用单个生成骨干在 4000+ 小时的 RGB 图像、深度、人体姿态和自车轨迹数据上联合建模，利用参考帧 + 稀疏特征 + 控制信号生成未来的 RGB 与深度序列，并通过新的 COM 指标系统定量评估对自车运动、物体动态和场景组合的可控性与跨场景泛化。

**不足点：**
当前模型在超长时序视频上的生成质量和时空一致性仍然有限，而且用于训练的自动伪标注精度受限，从而对控制和泛化能力带来一定约束


<a id="#protodepth-cvpr-2025"></a>
### ProtoDepth: Unsupervised Continual Depth Completion with Prototypes（CVPR 2025）

**创新点：**
将 RGB+稀疏点云深度补全视作原型驱动的持续学习问题，通过跨域共享的深度原型和域描述符，在无监督光度重投影框架下实现不同分布间的连续适配，在学习新场景的同时显著缓解传统深度补全模型的遗忘问题。

**不足点：**
它依赖先验的“数据集边界”来给新域分配 prototype 集，尚不能在无明确边界的真实在线场景中自动检测域变化并创建新 prototype；同时，在 domain-agnostic 场景下的 ProtoDepth-A 无法完全消除遗忘，在户外这类 domain gap 较大的序列中还会因为原型选择错误导致性能下降。


<a id="#ddeseg-cvpr-2025"></a>
### Dynamic Derivation and Elimination: Audio Visual Segmentation with Enhanced Audio Semantics（CVPR 2025）

**创新点：**
从音频本质出发提出 Dynamic Derivation and Elimination 框架：先通过语义重构从混合音频中派生出具有区分性的音源级表示，再用判别特征学习与动态消除模块过滤与画面无关的音源，使真正与视觉目标相关的声音区域得到更精准的匹配，从而在多种 AVS 数据集上显著提升声音引导分割性能。

**不足点：**
文中没有提到


<a id="#mmaudio-cvpr-2025"></a>
### MMAudio: Taming Multimodal Joint Training for High-Quality Video-to-Audio Synthesis（CVPR 2025）

**创新点：**
提出 MMAudio 多模态联合训练框架，将少量视频-音频配对数据与大规模文本-音频数据一起训练，并引入条件同步模块在音频潜变量上做帧级对齐，在流匹配目标下实现 157M 参数规模的高效视频→音频生成，在音质、语义对齐和视听同步上都优于现有公开方法，同时还在纯文本→音频任务上保持竞争力。

**不足点：**
MMAudio 主要面向 Foley 类通用音效，对人类语音这类复杂信号支持较弱——在生成说话声时，经常只产生听不清的含糊声音。
