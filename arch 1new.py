import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
from collections import OrderedDict

#########################################################################

Conv2d = nn.Conv2d
##########################################################################
## Layer Norm
def to_2d(x):
    return rearrange(x, 'b c h w -> b (h w c)')

def to_3d(x):
#    return rearrange(x, 'b c h w -> b c (h w)')
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
#    return rearrange(x, 'b c (h w) -> b c h w',h=h,w=w)
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

#        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) #* self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

#        self.weight = nn.Parameter(torch.ones(normalized_shape))
#        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) #* self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type="WithBias"):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FrozenCLIPBackbone(nn.Module):
    def __init__(self, clip_img_size=224):
        super().__init__()
        import clip
        clip_model, _ = clip.load("ViT-B/32", device="cpu", jit=False)
        self.clip_visual = clip_model.visual
        for p in self.clip_visual.parameters():
            p.requires_grad = False
        self.clip_visual.eval()

        self.clip_img_size = clip_img_size
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1,3,1,1)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1,3,1,1)
        self.register_buffer("clip_mean", mean, persistent=False)
        self.register_buffer("clip_std", std, persistent=False)

    @torch.no_grad()
    def forward(self, x):
        x = F.interpolate(x, size=(self.clip_img_size, self.clip_img_size),
                          mode="bilinear", align_corners=False)
        x = (x - self.clip_mean) / self.clip_std
        return self.clip_visual(x)  # [B, D]

class CLIPConditionerExact(nn.Module):
    def __init__(self, clip_out_dim, num_levels=4, feat_dim=512, prompt_dim=32):
        super().__init__()
        self.num_levels = num_levels
        self.feat_dim = feat_dim
        self.prompt_dim = prompt_dim
        # --- MLP  ---
        hidden = clip_out_dim // 16  # CoCoOp: hidden layer reduces input dim by 16×

        self.mlp = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(clip_out_dim, hidden)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(hidden, feat_dim)),  # 输出到你的 DeLevel feat 维度
        ]))

        #self.mlp_res_scale = nn.Parameter(torch.zeros(1))

        # --- learnable prompt  ---
        ctx_vectors = torch.empty(prompt_dim)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)  # CoCoOp初始化 std=0.02
        # meta_net：CoCoOp 的结构 Linear -> ReLU -> Linear
        self.meta_net = nn.Sequential(OrderedDict([
                    ("linear1", nn.Linear(feat_dim, feat_dim // 16)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("linear2", nn.Linear(feat_dim // 16, prompt_dim)),
        ]))
    def forward(self, clip_feat):
        #feat0 = self.mlp(clip_feat)
        delevel_feat = self.mlp(clip_feat)   # [B, feat_dim] 
        bias = self.meta_net(delevel_feat)                  # [B, prompt_dim]
        p_dynamic = self.ctx.unsqueeze(0) + bias            # [B, prompt_dim]  (CoCoOp 核心)
        concat_feat = torch.cat([delevel_feat, p_dynamic], dim=1)  # 直接拼接
        # prompt_vec = concat_feat.view(-1, self.feat_dim+self.prompt_dim, 1, 1)

        return concat_feat

##########################################################################
## Dual-scale Gated Feed-Forward Network (DGFF)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv_5 = Conv2d(hidden_features//4, hidden_features//4, kernel_size=5, stride=1, padding=2, groups=hidden_features//4, bias=bias)
        self.dwconv_dilated2_1 = Conv2d(hidden_features//4, hidden_features//4, kernel_size=3, stride=1, padding=2, groups=hidden_features//4, bias=bias, dilation=2)
        self.p_unshuffle = nn.PixelUnshuffle(2)
        self.p_shuffle = nn.PixelShuffle(2)

        self.project_out = Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        # assert x.shape[1] % 4 == 0, f"PixelShuffle expects channel%4==0, got {x.shape[1]}"
        x = self.p_shuffle(x)

        x1, x2 = x.chunk(2, dim=1)
#        x2_1, x2_2 = x2.chunk(2, dim=1)
        x1 = self.dwconv_5(x1)
        x2 = self.dwconv_dilated2_1( x2 )
#        x2_2 = self.dwconv_dilated3_1( x2_2 )
#        x2 = torch.cat([x2_1, x2_2], dim=1)
        x = (x2 * torch.tanh(F.softplus(x2))) * x1
        x = self.p_unshuffle(x)
        x = self.project_out(x)

        
#        x1 = self.dwconv_5(x)
#        x2 = self.dwconv_dilated_2(x)
#        x = F.mish(x2) * x1 + x 

        return x



##########################################################################
## Dynamic-range Histogram Self-Attention (DHSA)

class Attention_histogram(nn.Module):
    def __init__(self, dim, num_heads, bias, ifBox=True):
        super(Attention_histogram, self).__init__()
        self.factor = num_heads
        self.ifBox = ifBox
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = Conv2d(dim, dim*5, kernel_size=1, bias=bias)
        self.qkv_dwconv = Conv2d(dim*5, dim*5, kernel_size=3, stride=1, padding=1, groups=dim*5, bias=bias)
        self.project_out = Conv2d(dim, dim, kernel_size=1, bias=bias)


    def pad(self, x, factor):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw//factor+1)*factor-hw]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad
    def unpad(self, x, t_pad):
        _, _, hw = x.shape
        return x[:,:,t_pad[0]:hw-t_pad[1]]

    def softmax_1(self, x, dim=-1):
        logit = x.exp()
        logit  = logit / (logit.sum(dim, keepdim=True) + 1)
        return logit

    def normalize(self, x):
        mu = x.mean(-2, keepdim=True)
        sigma = x.var(-2, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) #* self.weight + self.bias
    

    def reshape_attn(self, q, k, v, ifBox):
        b, c = q.shape[:2]
        q, t_pad = self.pad(q, self.factor)
        k, t_pad = self.pad(k, self.factor)
        v, t_pad = self.pad(v, self.factor)
        hw = q.shape[-1] // self.factor
        shape_ori = "b (head c) (factor hw)" if ifBox else "b (head c) (hw factor)"
        shape_tar = "b head (c factor) hw"
        q = rearrange(q, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        k = rearrange(k, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        v = rearrange(v, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.softmax_1(attn, dim=-1)
        out = (attn @ v)
        out = rearrange(out, '{} -> {}'.format(shape_tar, shape_ori), factor=self.factor, hw=hw, b=b, head=self.num_heads)
        out = self.unpad(out, t_pad)
        return out

    def forward(self, x):
        b,c,h,w = x.shape
        x_sort, idx_h = x[:,:c//2].sort(-2)
        x_sort, idx_w = x_sort.sort(-1)
        x[:,:c//2] = x_sort
        qkv = self.qkv_dwconv(self.qkv(x))
        q1,k1,q2,k2,v = qkv.chunk(5, dim=1) # b,c,x,x

        v, idx = v.view(b,c,-1).sort(dim=-1)
        q1 = torch.gather(q1.view(b,c,-1), dim=2, index=idx)
        k1 = torch.gather(k1.view(b,c,-1), dim=2, index=idx)
        q2 = torch.gather(q2.view(b,c,-1), dim=2, index=idx)
        k2 = torch.gather(k2.view(b,c,-1), dim=2, index=idx)

        out1 = self.reshape_attn(q1, k1, v, True)
        out2 = self.reshape_attn(q2, k2, v, False)
        
        out1 = torch.scatter(out1, 2, idx, out1).view(b,c,h,w)
        out2 = torch.scatter(out2, 2, idx, out2).view(b,c,h,w)
        out = out1 * out2
        out = self.project_out(out)
        out_replace = out[:,:c//2]
        out_replace = torch.scatter(out_replace, -1, idx_w, out_replace)
        out_replace = torch.scatter(out_replace, -2, idx_h, out_replace)
        out[:,:c//2] = out_replace
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.attn_g = Attention_histogram(dim, num_heads, bias, True)
        self.norm_g = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

        self.norm_ff1 = LayerNorm(dim, LayerNorm_type)



    def forward(self, x):
        x = x + self.attn_g(self.norm_g(x))
        x_out = x + self.ffn(self.norm_ff1(x))

        return x_out



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

class SkipPatchEmbed(nn.Module):
    def __init__(self, in_c=3, dim=48, bias=False):
        super(SkipPatchEmbed, self).__init__()

        self.proj = nn.Sequential(
            nn.AvgPool2d( 2, stride=2, padding=0 , ceil_mode=False , count_include_pad=True , divisor_override=None ),
            Conv2d(in_c, dim, kernel_size=1, bias=bias),
            Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        )

    def forward(self, x, ):
        x = self.proj(x)

        return x

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
class Histoformer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], #四个尺度（level1~level4/latent）各自 TransformerBlock 的数量
        num_refinement_blocks = 4,
        heads = [1,2,4,8],#四个尺度各自注意力头数
        ffn_expansion_factor = 2.66,#FFN（前馈网络）扩展倍率，控制中间隐藏维度大小
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False,        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6

        # ===== prompt learning 相关新增 =====
        use_prompt=True,              # 是否启用 prompt
        prompt_mode="add",            # "add" or "concat"
        prompt_dim=16,                # prompt 向量通道数 Cp
        freeze_backbone=True,         # 是否冻结除 prompt 之外所有参数
        prompt_init="zero",           # "zero"prompt 初始化方式全 0 开始（默认对原模型输出零扰动，更稳） or "normal",prompt 用正态分布初始化

        # ===== DeLevel conditioning 新增 =====
        use_delevel=True,
        num_levels=4,            # 你分了几层就写几
        delevel_feat_dim=256,    # DeLevel feat 维度
        use_gt_level=True,       # 训练时用GT level选prompt；推理时自动用预测level
    ):

        super(Histoformer, self).__init__()
        self.inj_dim = delevel_feat_dim + prompt_dim
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.skip_patch_embed1 = SkipPatchEmbed(3, 3)
        self.skip_patch_embed2 = SkipPatchEmbed(3, 3)
        self.skip_patch_embed3 = SkipPatchEmbed(3, 3)
        self.reduce_chan_level_1 = Conv2d(int(dim*2**1)+3, int(dim*2**1), kernel_size=1, bias=bias)
        self.reduce_chan_level_2 = Conv2d(int(dim*2**2)+3, int(dim*2**2), kernel_size=1, bias=bias)
        self.reduce_chan_level_3 = Conv2d(int(dim*2**3)+3, int(dim*2**3), kernel_size=1, bias=bias)

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        # ---------------- prompt 注入模块（新增） ----------------
        self.use_prompt = use_prompt
        self.prompt_mode = prompt_mode
        self.prompt_dim = prompt_dim
        self.latent_dim = int(dim * 2**3)#原设计dim*2**32**3
        self.num_levels = num_levels
        if self.use_prompt:
            if prompt_mode == "add":
                self.prompt_proj = nn.Conv2d(self.inj_dim, self.latent_dim, kernel_size=1, bias=False)
            elif prompt_mode == "concat":
                
                fuse_in_dim = self.latent_dim + self.inj_dim
                hidden = int(fuse_in_dim * ffn_expansion_factor)

                hidden4 = hidden + ((4 - hidden % 4) % 4)
                ffn_prompt = hidden4 / fuse_in_dim
                # if hidden % 2 == 1:
                #     ffn_prompt = (hidden + 1) / fuse_in_dim
                
                self.prompt_tf = TransformerBlock(dim=fuse_in_dim,num_heads=heads[3],ffn_expansion_factor=ffn_prompt,bias=bias,LayerNorm_type=LayerNorm_type)
                print("PROMPT_TF fuse_in_dim:", fuse_in_dim,
                "project_in_out:", self.prompt_tf.ffn.project_in.out_channels,
                "out%4:", self.prompt_tf.ffn.project_in.out_channels % 4)

                #再 1×1 压回latent_dim
                self.prompt_fuse = nn.Conv2d(fuse_in_dim, self.latent_dim, kernel_size=1, bias=False)
            else:
                raise ValueError("prompt_mode must be 'add' or 'concat'")
        # ---------------- CLIP DeLevel conditioner（严格按图） ----------------
        self.use_gt_level = use_gt_level
        self.clip_backbone = FrozenCLIPBackbone(clip_img_size=224)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            clip_out_dim = self.clip_backbone(dummy).shape[1]

        self.conditioner = CLIPConditionerExact(
            clip_out_dim=clip_out_dim,
            num_levels=num_levels,
            feat_dim=delevel_feat_dim,
            prompt_dim=prompt_dim
        )
        #
        self.use_delevel = use_delevel
        if self.use_delevel:
            self.cls_head = nn.Linear(delevel_feat_dim + prompt_dim, num_levels - 1)
            #self.ce_loss = nn.CrossEntropyLoss()


        # ---------------- 冻结策略（严格按图） ----------------
        if freeze_backbone:
            for p in self.parameters():
                p.requires_grad = False

            # 1) bottleneck injection 可训练（火焰：Bottleneck Prompt Injection）
            if self.use_prompt:
                if self.prompt_mode == "add":
                    for p in self.prompt_proj.parameters():
                        p.requires_grad = True
                else:
                    for p in self.prompt_tf.parameters():
                        p.requires_grad = True
                    for p in self.prompt_fuse.parameters():
                        p.requires_grad = True
            for p in self.cls_head.parameters():
                p.requires_grad = True
            # 2) conditioner 可训练（火焰：MLP、learnable prompt、cls head、feat_to_prompt）
            for p in self.conditioner.parameters():
                p.requires_grad = True

            # clip_backbone 内部已冻结（保持冻结即可）


    # clip_backbone 保持冻结即可（它内部本来 requires_grad=False）
    def _inject_prompt(self, latent, prompt_vec):
        """
        latent: [B, C_latent, H, W]
        prompt_vec: [B, Cp, 1, 1]
        """
        b, c, h, w = latent.shape
        p = prompt_vec.expand(b, -1, h, w)

        if self.prompt_mode == "add":
            return latent + self.prompt_proj(p)
        else:
            x = torch.cat([latent, p], dim=1)
            x = self.prompt_tf(x)                                  # TransformerBlock
            x = self.prompt_fuse(x)                                # 1×1 压回 latent_dim
            return x
        
    def forward(self, inp_img, level_gt=None, return_cls=False):
        # ===== DeLevel branch: compute logits + DeLevel prompt =====
        clip_feat = self.clip_backbone(inp_img)  # [B, D]
        use_gt = (level_gt is not None) and self.training   # 训练用GT，推理不用
        concat_feat = self.conditioner(clip_feat)
        logits = self.cls_head(concat_feat)
        inj = concat_feat.view(concat_feat.size(0), self.inj_dim, 1, 1)
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1) # c,h,w

        inp_enc_level2 = self.down1_2(out_enc_level1) # 2c, h/2, w/2
        skip_enc_level1 = self.skip_patch_embed1(inp_img)
        inp_enc_level2 = self.reduce_chan_level_1(torch.cat([inp_enc_level2, skip_enc_level1], 1))

        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        skip_enc_level2 = self.skip_patch_embed2(skip_enc_level1)
        inp_enc_level3 = self.reduce_chan_level_2(torch.cat([inp_enc_level3, skip_enc_level2], 1))

        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        skip_enc_level3 = self.skip_patch_embed3(skip_enc_level2)
        inp_enc_level4 = self.reduce_chan_level_3(torch.cat([inp_enc_level4, skip_enc_level3], 1))

        latent = self.latent(inp_enc_level4) 
        # --- prompt 强度由难度标签控制（label越高越难）---
        if (level_gt is not None) and self.training:
            level_id = level_gt
        else:
            p = torch.sigmoid(logits)
            level_id = (p > 0.5).long().sum(dim=1)   # 0..3

        scale = (level_id.float() / (self.num_levels - 1)).view(-1,1,1,1)


        # scale = (level_id.float() / (self.num_levels - 1)).clamp(0, 1).view(-1, 1, 1, 1)

    # ===== bottleneck 注入：优先用 DeLevel prompt，否则退回你原来的 self.prompt =====
    # --- 3) bottleneck prompt injection（严格：只用 conditioner prompt）---
        if self.use_prompt:
            latent = self._inject_prompt(latent, inj)
            # latent = latent + scale * (latent_inj - latent)
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        ###########################

        out_dec_level1 = self.output(out_dec_level1)
        restored = out_dec_level1 + inp_img
        return (restored, logits) if return_cls else restored

        

