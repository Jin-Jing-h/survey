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

##  论文详细笔记
<a id="fovex-ijcv-2025"></a>
### 📖FovEx: Human-Inspired Explanations for Vision Transformers and CNNs （IJCV 2025）

**数据集：**  
[ImageNet-1K](https://www.image-net.org/challenges/LSVRC/)  
> 用于：在 ImageNet-1K 验证集中随机选取 5000 张**被模型正确分类**的图像，作为 FovEx 评估解释“可信度/定位能力”（如保持度、删除度、局部化等指标）的主要测试集，为 CNN 和 ViT 提供统一的分类场景。  

[MIT1003](https://people.csail.mit.edu/tjudd/WherePeopleLook/)  
> 用于：提供 1003 张图像及对应的人眼注视/显著性图，FovEx 用其对比“方法生成的解释热力图”和**真实人眼凝视模式**的一致性，用来评估解释的“类人程度（human alignment）”。  



**创新点：**
本文提出了 FovEx，这一结合类人凹视机制与梯度驱动扫视、可同时适用于 CNN 与 ViT 的统一 XAI 方法，在多项信赖度指标与人眼凝视一致性上优于现有方法

**不足点：**
优化目标偏向“保留”关键信息导致在 DELETE 指标上表现欠佳且仅在有限数据集与任务上验证，存在泛化性和人群偏置方面的潜在局限。

<a id="mair-cvpr-2025"></a>
### 📖MaIR: A Locality- and Continuity-Preserving Mamba for Image Restoration（CVPR 2025）
**数据集：**  
[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)  
> 用于：超分辨率与去噪任务的基础高分辨率训练集之一，为 MaIR 提供多样自然场景图像，并在验证集上评估标准合成退化下的恢复效果。  

[Flickr2K](https://www.kaggle.com/datasets/daehoyang/flickr2k)  
> 用于：与 DIV2K 组成 DF2K 扩充训练集，进一步增加自然场景与纹理多样性，用于超分与去噪训练。  

[WED](https://ivc.uwaterloo.ca/database/WaterlooExploration/)  
> 用于：合成高斯噪声去噪任务的训练数据之一（DFWB 训练集合部分），提升 MaIR 在多种内容和噪声强度下的鲁棒性。  

[BSD400 / BSD68](https://github.com/clausmichele/CBSD68-dataset)  
> 用于：BSD400 参与合成去噪训练；BSD68 作为经典测试集，评估 MaIR 在不同噪声等级下的合成高斯去噪性能。  

[Kodak24](https://r0k.us/graphics/kodak/)  
> 用于：合成噪声去噪的测试集之一，包含 24 张高质量自然图像，用于检验 MaIR 在真实纹理和色彩保持方面的表现。  

[McMaster](https://www4.comp.polyu.edu.hk/~cslzhang/CDM_Dataset.htm)  
> 用于：彩色图像去噪测试集，强调纹理与色彩信息，评估 MaIR 在高频纹理和强颜色区域的恢复能力。  

[Urban100](https://huggingface.co/datasets/eugenesiow/Urban100)  
> 用于：既作为超分辨率任务的测试集（考察城市场景结构和细节恢复），也用于合成噪声去噪评测 MaIR 在复杂几何结构上的表现。  

[SIDD-Medium](https://abdokamel.github.io/sidd/)  
> 用于：真实噪声去噪任务的训练与测试数据集，来自手机拍摄 RAW/ sRGB 图像，用于评估 MaIR 对真实成像噪声的建模与泛化能力。  

[GoPro](https://seungjunnah.github.io/Datasets/gopro.html)  
> 用于：动态场景运动模糊去除任务的主要训练与测试集，评估 MaIR 在复杂相机与物体运动环境下的去模糊能力。  

[HIDE](https://github.com/joanshen0508/HA_deblur)  
> 用于：以人脸和人物为主的去模糊测试集，检验 MaIR 在含人物/人脸场景中的细节恢复和感知质量。  

[RESIDE (ITS/OTS/SOTS)](https://sites.google.com/view/reside-dehaze-datasets/reside-standard)  
> 用于：合成图像去雾任务的主要训练与测试数据集，其中 ITS/OTS 用于室内/室外合成雾图训练，SOTS-Indoor/Outdoor 用于标准去雾测试，评估 MaIR 在不同雾密度与场景下的表现。  

[RESIDE-6K](https://gts.ai/dataset-download/reside-6k/)  
> 用于：更大规模、多场景的去雾训练/验证数据集，为 MaIR 提供更丰富的雾天场景和退化分布，进一步提升去雾任务的泛化能力。  


**创新点：**
MaIR 提出在 Mamba 状态空间模型里加入 Nested S-shaped Scanning（NSS）+ Sequence Shuffle Attention（SSA），同时保持图像的局部性和空间连续性，相比以往简单按行或者列展平成 1D 序列的 Mamba 方案，在超分、去噪、去模糊、去雾等 4 大任务、14 个数据集上全面刷了 40 个基线。

**不足点：**
文中没有提到

<a id="defusion-cvpr-2025"></a>
### 📖Visual-Instructed Degradation Diffusion for All-in-One Image Restoration（CVPR 2025）

**数据集：**

[Rain1400](https://xueyangfu.github.io/projects/cvpr2017.html)

> 用于：合成雨条/雨丝去雨（single-image deraining）的经典基准数据集。

[Outdoor-Rain](https://github.com/liruoteng/HeavyRainRemoval)

> 用于：复杂户外重雨场景去雨，评估模型在多种雨型和背景条件下的鲁棒性。

[RESIDE (ITS/OTS/SOTS)](https://sites.google.com/view/reside-dehaze-datasets/reside-standard)

> 用于：单幅图像去雾（dehazing），涵盖室内/室外、合成/真实等多种雾场景，是最常用的去雾基准之一。

[Dense-Haze](https://data.vision.ee.ethz.ch/cvl/ntire19/dense-haze/)

> 用于：高浓度、近似白雾场景的去雾任务，考察在极端雾霾条件下的复原能力。

[Snow100K](https://sites.google.com/view/yunfuliu/desnownet)

> 用于：合成雪花/雪点去雪（desnowing），支持评估不同雪密度与形态下的性能。

[RealSnow](https://github.com/zhuyr97/WGWS-Net)

> 用于：真实采集雪景的去雪任务，检验模型从合成数据泛化到真实场景的能力。

[RainDrop](https://github.com/rui1996/DeRaindrop)

> 用于：去除镜头/玻璃上的雨滴遮挡（raindrop removal），同时恢复被遮挡的背景细节。

[RainDS](https://github.com/Songforrr/RainDS_CCN)

> 用于：同时包含“雨丝 + 雨滴”的混合退化场景，适合评估 all-in-one 去雨模型。

[SIDD](https://abdokamel.github.io/sidd/)

> 用于：真实手机拍摄噪声去噪（denoising），提供成对 noisy/clean 图像。

[GoPro](https://seungjunnah.github.io/Datasets/gopro.html)

> 用于：由相机或物体运动产生的运动模糊去模糊（motion deblurring），经典配对数据集。

[RealBlur](https://cg.postech.ac.kr/research/realblur/)

> 用于：真实拍摄模糊图像的去模糊，相比 GoPro 更贴近真实模糊分布。

[DPDD](https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel)

> 用于：离焦/景深模糊去模糊（defocus deblurring），基于双像素（dual-pixel）成像。

[LIVE1](https://live.ece.utexas.edu/research/Quality/)

> 用于：图像质量评估及压缩伪影/去噪等任务的小型测试集，常用于客观/主观质量对比。

[NH-HAZE](https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/)

> 用于：非均匀雾（non-homogeneous haze）场景去雾，更贴近真实复杂雾分布。

[LHP-Rain](https://github.com/yunguo224/LHP-Rain)

> 用于：大规模真实雨场景去雨（real rain removal），用于评估模型在真实雨环境下的泛化能力。

[WED](https://ivc.uwaterloo.ca/database/WaterlooExploration/)

> 用于：通用自然图像质量评估，以及图像增强/复原算法的测试与对比。

[EUVP](https://irvlab.cs.umn.edu/resources/euvp-dataset)

> 用于：水下图像增强与色彩/对比度恢复（underwater image enhancement）。



**创新点：**
Defusion把all-in-one 图像恢复做成一个视觉指令驱动的退化扩散，不是用模糊的文本prompt，而是构造与不同退化（去噪、去模糊、去雾、低照度等）显式对齐的视觉指令图，作为条件去引导扩散模型，对未知退化场景也能统一建模，在多种一体化恢复基准上达到了新的SOTA。

**不足点：**
文中没有提到

<a id="darkir-cvpr-2025"></a>
### 📖DarkIR: Robust Low-Light Image Restoration（CVPR 2024）

**数据集：**  
[LOL-Blur](https://github.com/sczhou/LEDNet#lol-blur-dataset)  
> 用于：DarkIR 的**主训练集**和合成低照+运动模糊场景下的**核心测试基准**，评估同时去噪、去模糊与提亮能力  

[Real-LOLBlur](https://github.com/sczhou/LEDNet#lol-blur-dataset)  
> 用于：从 RealBlur-J 等真实视频中截取的夜景模糊图像，无 GT，作为**真实世界低照+模糊场景的无参考测试集**，检验泛化与视觉效果  

[LOL（LOw-Light paired dataset）](https://daooshee.github.io/BMVC2018website/)  
> 用于：经典**低照度增强配对数据集**（无模糊），在 DarkIR 中作为额外基准，验证网络在纯 LLIE 任务上的恢复质量  

[LOL-v2-Real](https://huggingface.co/datasets/okhater/lolv2-real)  
> 用于：包含 689/100 对真实低照–正常光图像的**配对数据集**，在更复杂真实场景下训练/评估 DarkIR 的低照增强与去噪性能  

[LOL-v2-Synthetic](https://huggingface.co/datasets/okhater/lolv2-synthetic)  
> 用于：900/100 对合成低照–正常光图像，作为补充数据提升 DarkIR 在**合成低照场景**下的泛化与稳定性  

[LSRW](https://github.com/JianghaiSCU/R2RNet#dataset)  
> 用于：由 Nikon 相机与华为手机采集的**真实低照配对数据集**，在不同设备与曝光条件下评估 DarkIR 的跨设备泛化能力

**创新点：**
DarkIR 针对夜景/低照环境下同时存在的 噪声 + 低照度 + 运动模糊，在高效 CNN 上设计新的注意力机制扩展感受野，构建了一个统一的多任务低照度恢复网络，在 LOLBlur、LOLv2、Real-LOLBlur 等数据集上刷新 SOTA，并在 NTIRE 2025 低照度挑战中获得最佳方法，同时保持参数量和 MAC 数显著低于大多数 Transformer 模型。

**不足点：**
DarkIR 的主要局限在于：虽然通过大量使用 depth-wise 卷积显著降低了参数量和 MACs，但作者在 Limitations 中明确指出，这类算子在实际 GPU 上算术强度较低、对硬件不够友好，因此推理时间并不会随着计算量成比例下降；此外，多任务 all-in-one 版本在获得更强泛化能力的同时，在 LOLBlur 等数据集上仍存在约 0.5 dB 的轻微性能损失，说明在统一建模多种低照退化时仍面临精度与泛化、效率之间的折中。

<a id="faithdiff-cvpr-2025"></a>
### 📖FaithDiff: Unleashing Diffusion Priors for Faithful Image Super-resolution（CVPR 2025）

**数据集：**  
[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)  
> 用于：合成退化超分的基础训练与验证数据之一，在 DIV2K-Valid 上构造不同退化等级 (D-level) 用于评测模型在标准自然场景上的保真重建能力。  

[LSDIR](https://github.com/ofsoundof/LSDIR)  
> 用于：大规模真实/合成混合恢复数据集，为 FaithDiff 提供多样化高分辨率图像，在 LSDIR-Valid 上合成多种真实感退化以训练和评估对复杂退化的鲁棒性。  

[Flickr2K](https://www.kaggle.com/datasets/daehoyang/flickr2k)  
> 用于：补充自然场景高分辨率图像，与 DIV2K 等一起扩充训练集，提高模型在多类型真实照片上的泛化能力。  

[DIV8K](https://huggingface.co/datasets/yangtao9009/DIV8K)  
> 用于：提供高达 8K 分辨率的图像，帮助模型学习更丰富的高频细节和复杂场景纹理，用于高分辨率超分的训练/验证。  

[FFHQ](https://github.com/NVlabs/ffhq-dataset)  
> 用于：抽取人脸高分辨率图像，专门增强 FaithDiff 在面部细节（皮肤纹理、五官结构等）上的重建能力。  

[RealPhoto60](https://drive.google.com/drive/folders/1yELzm5SvAi9e7kPcO_jPp2XkTs4vK6aR)  
> 用于：由 SUPIR 提供的 60 张真实退化图像集合，在无 GT 的真实场景下评估 FaithDiff 的视觉质量和结构一致性。  

[RealDeg](https://drive.google.com/file/d/1B8BaaMjXJ-1TfcTgE9MrAg8ufvaGkndP/view)  
> 用于：作者新收集的 238 张真实退化图像（老照片、电影剧照、社交媒体图片等），专门用于检验模型在多种未知真实退化类型下的稳健性与泛化表现。  

**创新点：**
FaithDiff 针对“既要好看又要保真”的真实场景超分问题，提出在 latent diffusion 上加入 特征对齐模块 + 编码器与扩散模型的联合微调，显式对齐退化输入特征与扩散噪声空间，让大模型的先验既能生成细节又不过度幻觉，在多种 SR 基准上对结构保持和视觉质量都明显优于以往扩散式 SR 方法

**不足点：**
文中没有提到

<a id="gem-cvpr-2025"></a>
### 📖GEM: A Generalizable Ego-Vision Multimodal World Model for Fine-Grained Ego-Motion, Object Dynamics, and Scene Composition Control（CVPR 2025）

**数据集：**  
[OpenDV](https://github.com/OpenDriveLab/DriveAGI)  
> 用于：GEM 的核心大规模驾驶视频语料（1700+ 小时多城市、多天气前视视频），作为主要训练数据之一，并在其验证集子集上评估长时序视频生成质量与可控性。  

[nuScenes](https://www.nuscenes.org/)  
> 用于：多传感器自动驾驶数据集（包含 3D 标注与精确轨迹），在 GEM 中既参与训练，又作为带 GT 轨迹的关键评测基准，用于计算 ADE 等控制误差与视频质量指标。  

[DrivingDojo](https://huggingface.co/datasets/Yuqi1997/DrivingDojo)  
> 用于：强调多智能体交互与复杂交通行为的大规模驾驶数据集，为 GEM 提供包含变道、跟车、拥堵等复杂动态的场景，提升模型在高交互场景下的可控生成能力。  

[Honda HDD](https://usa.honda-ri.com/hdd)  
> 用于：包含 100+ 小时真实自然驾驶的视频与车辆 CAN 信号，在 GEM 中作为额外驾驶行为语料，帮助模型学习更贴近日常驾驶风格的世界动态。  

[Honda HAD](https://usa.honda-ri.com/had)  
> 用于：在 HDD 基础上加入人类建议/干预信息的人类驾驶数据集，用于补充带语义引导的驾驶场景，使 GEM 接触到更多样的驾驶意图与操作模式。  

[DoTA](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly)  
> 用于：交通异常检测数据集，包含大量事故与近事故片段，作为 GEM 训练中的稀有/极端场景补充，使世界模型对异常事件和危险情形的建模更为合理。  

[CarCrashDataset](https://github.com/Cogito2012/CarCrashDataset)  
> 用于：专注真实车祸与正常行驶对比的 dashcam 视频数据集，在 GEM 中进一步丰富安全关键事件样本，提升对碰撞、急刹等极端动态的表达能力。  

[EgoExo4D](https://docs.ego-exo4d-data.org/)  
> 用于：大规模第一/第三人称多视角人类活动数据集，在 GEM 中作为人类活动域的核心数据，用于训练和评估 human-pose 控制与复杂人–物体–场景交互的世界建模能力。  

[self-collected](https://vita-epfl.github.io/GEM.github.io/)  
> 用于：作者从 YouTube 自采集的约 27.4 小时无人机第一视角视频，用于扩展 GEM 到无人机导航域，检验模型在不同高度和视角下的泛化与可控生成能力。  

**创新点：**
提出 GEM 这一统一的自监督多模态世界模型，用单个生成骨干在 4000+ 小时的 RGB 图像、深度、人体姿态和自车轨迹数据上联合建模，利用参考帧 + 稀疏特征 + 控制信号生成未来的 RGB 与深度序列，并通过新的 COM 指标系统定量评估对自车运动、物体动态和场景组合的可控性与跨场景泛化。

**不足点：**
当前模型在超长时序视频上的生成质量和时空一致性仍然有限，而且用于训练的自动伪标注精度受限，从而对控制和泛化能力带来一定约束


<a id="protodepth-cvpr-2025"></a>
### 📖ProtoDepth: Unsupervised Continual Depth Completion with Prototypes（CVPR 2025）
**数据集：**  
[NYU Depth V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)  
> 用于：室内序列的起始数据集 D1，用来预训练室内深度补全模型，以及评估在典型室内场景下的深度补全性能与遗忘程度。  

[VOID](https://github.com/alexklwong/void-dataset)  
> 用于：室内序列中的目标域之一，具有极稀疏深度与强相机运动，作为 室内持续学习序列的后续域，评估 ProtoDepth 在低纹理和大位姿变化场景下的鲁棒性与遗忘情况。  

[ScanNet](https://www.scan-net.org/)  
> 用于：大规模 RGB-D 室内视频数据集，检验方法在跨多种室内场景和传感器配置时的持续适应能力。  

[KITTI](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion)  
> 用于：室外序列的起始数据集 D1，作为道路场景深度补全预训练基准，并在 KITTI、Waymo、VKITTI 持续学习序列中提供真实自动驾驶场景下的稀疏 LiDAR + RGB 训练与评测。  

[Waymo Open Dataset](https://waymo.com/open/)  
> 用于：室外持续学习序列中的第二个真实自动驾驶数据集，具有更高分辨率和更丰富路况，评估 ProtoDepth 在真实跨域驾驶场景中的适应与遗忘。  

[Virtual KITTI](https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/)  
> 用于：合成的 KITTI 场景克隆数据集，在 ProtoDepth 中作为室外序列中的合成目标域，通过添加不同天气和视角变换来模拟 domain shift，用于研究在合成域上的持续学习和遗忘抑制能力。  

[nuScenes](https://www.nuscenes.org/)  
> 用于：未见室外数据集的零样本泛化测试，在训练顺序 KITTI → Waymo → VKITTI 之后，对 nuScenes 进行推理，以评估 ProtoDepth-A 在新城市、新传感器配置下的零样本深度补全泛化性能。  


**创新点：**
将 RGB+稀疏点云深度补全视作原型驱动的持续学习问题，通过跨域共享的深度原型和域描述符，在无监督光度重投影框架下实现不同分布间的连续适配，在学习新场景的同时显著缓解传统深度补全模型的遗忘问题。

**不足点：**
它依赖先验的“数据集边界”来给新域分配 prototype 集，尚不能在无明确边界的真实在线场景中自动检测域变化并创建新 prototype；同时，在 domain-agnostic 场景下的 ProtoDepth-A 无法完全消除遗忘，在户外这类 domain gap 较大的序列中还会因为原型选择错误导致性能下降。


<a id="ddeseg-cvpr-2025"></a>
### 📖Dynamic Derivation and Elimination: Audio Visual Segmentation with Enhanced Audio Semantics（CVPR 2025）
**数据集：**  
[AVS-Object](https://github.com/OpenNLPLab/AVSBench)  
> 用于：DDESeg 的核心基准之一，包含 S4（Single Source）和 MS3（Multi Source）两部分，每段 5 秒视频并在每秒最后一帧提供像素级二值掩码，用于评估“单声源与多声源下是否正确分割出发声区域”。  

[AVS-Semantic](https://github.com/OpenNLPLab/AVSBench)  
> 用于：语义级音视频分割基准，含 12,356 段、71 类音视频片段，每个样本提供语义掩码与音频事件类别标签，用于评估 DDESeg 在“既要找对位置又要分类对类别”的语义 AVS 能力。  

[VPO](https://drive.google.com/file/d/12jq7-Ke09ZPoUI1od44q97DNLrThoHc3/view)  
> 用于：由 COCO 单帧图像与 VGGSound 3 秒音频按类别重配得到的合成 AVS 数据集，包含 VPO-SS / VPO-MS / VPO-MSMI 三个子集，用来检验 DDESeg 在“单/多声源、同类多目标”等更复杂组合场景下的泛化与鲁棒性。  

**创新点：**
从音频本质出发提出 Dynamic Derivation and Elimination 框架：先通过语义重构从混合音频中派生出具有区分性的音源级表示，再用判别特征学习与动态消除模块过滤与画面无关的音源，使真正与视觉目标相关的声音区域得到更精准的匹配，从而在多种 AVS 数据集上显著提升声音引导分割性能。

**不足点：**
文中没有提到


<a id="mmaudio-cvpr-2025"></a>
### 📖MMAudio: Taming Multimodal Joint Training for High-Quality Video-to-Audio Synthesis（CVPR 2025）
**数据集：**  
[VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/)  
> 用于：唯一同时包含视频–音频–文本三模态的核心训练与评测集，MMAudio 在其中进行主的视频→音频训练，并在测试集上评估音频质量、语义对齐度与视听同步性。  

[AudioCaps](https://audiocaps.github.io/)  
> 用于：高质量音频–文本配对数据集，为多模态联合训练提供人工标注描述，并在其测试集上评估文本→音频生成能力以及语义一致性。  

[Clotho](https://zenodo.org/record/3490684)  
> 用于：补充 AudioCaps 的另一套音频字幕数据，覆盖更丰富的环境声音和时长分布，用来提升 MMAudio 的音频–语言对齐能力与泛化表现。  

[WavCaps](https://github.com/XinhaoMei/WavCaps)  
> 用于：约 7600 小时的大规模弱标注音频字幕语料，作为主要的大规模音频–文本训练数据源，用于数据扩展和学习更通用的自然声音分布。  

[Greatest Hits](https://andrewowens.com/vis/)  
> 用于：鼓棒敲击物体的视频集合，作为额外的视听同步性评测基准，通过模型无关指标（如 onset/AVC 等）检验 MMAudio 的时间对齐能力。  

[Movie Gen Audio Bench](https://github.com/facebookresearch/MovieGenBench)  
> 用于：合成视频上的出分布评测基准，在无 GT 条件下通过主观评价和 IS、IB-score、CLAP、DeSync 等指标，对比 MMAudio 与 Movie Gen Audio 在复杂生成视频上的音频质量和语义一致性。  

**创新点：**
提出 MMAudio 多模态联合训练框架，将少量视频-音频配对数据与大规模文本-音频数据一起训练，并引入条件同步模块在音频潜变量上做帧级对齐，在流匹配目标下实现 157M 参数规模的高效视频→音频生成，在音质、语义对齐和视听同步上都优于现有公开方法，同时还在纯文本→音频任务上保持竞争力。

**不足点：**
MMAudio 主要面向 Foley 类通用音效，对人类语音这类复杂信号支持较弱——在生成说话声时，经常只产生听不清的含糊声音。
