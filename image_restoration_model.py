import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

import os
import random
import numpy as np
import cv2
import torch.nn.functional as F
from functools import partial
#from audtorch.metrics.functional import pearsonr
import torch.autograd as autograd

class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1,1)).item()
    
        r_index = torch.randperm(target.size(0)).to(self.device)
    
        target = lam * target + (1-lam) * target[r_index, :]
        input_ = lam * input_ + (1-lam) * input_[r_index, :]
    
        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments)-1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_

class ImageCleanModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageCleanModel, self).__init__(opt)

        # define network

        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta       = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
            use_identity     = self.opt['train']['mixing_augs'].get('use_identity', False)
            self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)
        self.level_hist = torch.zeros(4, dtype=torch.long)
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        self.level = None
        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()
        self.psnr_best = -1
    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(self.opt['network_g']).to(
                self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            raise ValueError('pixel loss are None.')
        if train_opt.get('seq_opt'):
#            from audtorch.metrics.functional import pearsonr
#            self.cri_seq = pearsonr
            self.cri_seq = self.pearson_correlation_loss #
        # w = torch.tensor([1.0, 1.0, 1.2, 1.2], device=self.device)  # 先轻微拉一下 2/3
        self.cri_ord = torch.nn.BCEWithLogitsLoss()
        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def pearson_correlation_loss(self, x1, x2):
        assert x1.shape == x2.shape
        b, c = x1.shape[:2]
        dim = -1
        x1, x2 = x1.reshape(b, -1), x2.reshape(b, -1)
        x1_mean, x2_mean = x1.mean(dim=dim, keepdims=True), x2.mean(dim=dim, keepdims=True)
        numerator = ((x1 - x1_mean) * (x2 - x2_mean)).sum( dim=dim, keepdims=True )
        
        std1 = (x1 - x1_mean).pow(2).sum(dim=dim, keepdims=True).sqrt() 
        std2 = (x2 - x2_mean).pow(2).sum(dim=dim, keepdims=True).sqrt()
        denominator = std1 * std2
        corr = numerator.div(denominator + 1e-6)
        return corr

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        logger = get_root_logger()

        # 统计 trainable / frozen
        trainable = [(k, v.numel(), tuple(v.shape)) for k, v in self.net_g.named_parameters() if v.requires_grad]
        frozen    = [(k, v.numel(), tuple(v.shape)) for k, v in self.net_g.named_parameters() if not v.requires_grad]

        logger.info("==== Trainable parameters (will go to optimizer) ====")
        for k, numel, shp in trainable:
            logger.info(f"[T] {k} | shape={shp} | numel={numel}")

        logger.info(f"Trainable tensors: {len(trainable)}")
        logger.info(f"Trainable numel  : {sum(x[1] for x in trainable):,}")
        logger.info(f"Frozen numel     : {sum(x[1] for x in frozen):,}")
        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)
        opt_ids = set(id(p) for g in self.optimizer_g.param_groups for p in g["params"])

        logger.info("==== Parameters actually IN optimizer ====")
        for k, v in self.net_g.named_parameters():
            if id(v) in opt_ids:
                logger.info(f"[OPT] {k} | shape={tuple(v.shape)} | requires_grad={v.requires_grad}")

        logger.info("Optimizer total numel: %d", sum(p.numel() for g in self.optimizer_g.param_groups for p in g["params"]))


    def feed_train_data(self, data):
        self.feed_data(data)
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'label' in data:
            self.label = data['label']
#            self.label = torch.nn.functional.one_hot(data['label'], num_classes=3)
        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)
        if 'level' in data:
            self.level = data['level'].to(self.device)  # shape [B] 或 [B,1]

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if self.level is not None:
            for v in self.level.detach().cpu().view(-1):
                self.level_hist[int(v)] += 1
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        # label（可选）
        if 'label' in data:
            self.label = data['label']

        # mixing（只在训练时做）
        if self.is_train and self.mixing_flag and ('gt' in data):
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

        # level：关键——每个 iteration 都刷新，没给就置 None
        lvl = data.get('level', None)
        if lvl is None:
            self.level = None
        else:
            self.level = lvl.to(self.device).long().view(-1)
            # 可选：范围检查
            if (self.level.min() < 0) or (self.level.max() >= 4):
                print("BAD LEVEL RANGE:", self.level.min().item(), self.level.max().item())
        
    def check_inf_nan(self, x):
        x[x.isnan()] = 0
        x[x.isinf()] = 1e7
        return x
    def compute_correlation_loss(self, x1, x2):
        b, c = x1.shape[0:2]
        x1 = x1.reshape(b, -1)
        x2 = x2.reshape(b, -1)
#        print(x1, x2)
        pearson = (1. - self.cri_seq(x1, x2)) / 2.
        return pearson[~pearson.isnan()*~pearson.isinf()].mean()
    def compute_losses(self, output, gt, logits, level_gt):
        """Return dict of tensor losses (on current device)."""
        loss_dict = OrderedDict()

        l_pix = self.cri_pix(output, gt)
        loss_dict['l_pix'] = l_pix

        l_pear = self.compute_correlation_loss(output, gt)
        loss_dict['l_pear'] = l_pear

        # cls (ordinal)
        if logits is None or level_gt is None:
            l_cls = output.new_tensor(0.0)
            w = output.new_tensor(0.0)
        else:
            gt_lv = level_gt.view(-1).long()
            t = torch.stack([(gt_lv >= 1), (gt_lv >= 2), (gt_lv >= 3)], dim=1).float()
            l_cls = self.cri_ord(logits, t)

            # auto-balance weight (same as train)
            r = self.opt['train'].get('cls_ratio', 0.1)
            eps = 1e-8
            w_min = self.opt['train'].get('cls_w_min', 0.0)
            w_max = self.opt['train'].get('cls_w_max', 0.05)
            # val：使用 train 的 EMA 权重（不更新，只读取），保证 train/val 的 l_cls_w 可比
            w = getattr(self, 'cls_w_ema', None)
            if w is None:
                # 如果 val 先于 train 被调用（极少见），fallback 到即时权重
                w = (r * l_pix.detach()) / (l_cls.detach() + eps)
                w = torch.clamp(w, w_min, w_max)


        loss_dict['l_cls'] = l_cls
        loss_dict['cls_w'] = w.detach()
        loss_dict['l_cls_w'] = w * l_cls

        loss_dict['loss_total'] = loss_dict['l_pix'] + loss_dict['l_pear'] + loss_dict['l_cls_w']
        return loss_dict

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        level_gt = self.level
        self.output, logits = self.net_g(self.lq, level_gt=self.level, return_cls=True)
        # l_cls = self.cri_celoss(logits, gt)
        loss_dict = OrderedDict()
        # pixel loss
        l_pix = self.cri_pix(self.output, self.gt)
        loss_dict['l_pix'] = l_pix
        '''
        l_mask = self.cri_pix(self.pred_mask, self.gt - self.output.detach())
        loss_dict['l_mask'] = l_mask
        '''
        l_pear = self.compute_correlation_loss(self.output, self.gt)
        loss_dict['l_pear'] = l_pear
    # cls loss：同时保证 logits 不是 None
        # cls loss
        # cls loss
        if logits is None or level_gt is None:
            l_cls = self.output.new_tensor(0.0)
        else:
            # gt: [B] in {0,1,2,3}
            gt = level_gt.view(-1).long()

            # 生成 ordinal targets: [B,3]
            # t0 = (gt>=1), t1=(gt>=2), t2=(gt>=3)
            t = torch.stack([(gt >= 1), (gt >= 2), (gt >= 3)], dim=1).float()

            # logits: [B,3]
            l_cls = self.cri_ord(logits, t)

        # if logits is None or level_gt is None:
        #     l_cls = self.output.new_tensor(0.0)
        # else:
        #     # gt: [B] in {0,1,2}
        #     gt = level_gt.view(-1).long()

        #     # ordinal targets: [B,2]
        #     # t0=(gt>=1), t1=(gt>=2)
        #     t = torch.stack([(gt >= 1), (gt >= 2)], dim=1).float()

        #     # logits should be [B,2]
        #     l_cls = self.cri_ord(logits, t)


        loss_dict['l_cls'] = l_cls
        # if current_iter == 10:   # 只打印一次/某次
        #     x = self.lq
        #     print("[DBG] lq range:", float(x.min()), float(x.max()),
        #         "mean:", float(x.mean()), "std:", float(x.std()))
        # if current_iter == 10:
        #     if level_gt is None:
        #         print("DEBUG level: None")
        #     else:
        #         print("DEBUG level:", level_gt[:8].detach().cpu().numpy(), "dtype:", level_gt.dtype)

        #     if logits is None:
        #         print("DEBUG logits: None")
        #     elif isinstance(logits, (tuple, list)):
        #         info = []
        #         for i, lg in enumerate(logits):
        #             info.append((i, tuple(lg.shape), float(lg.abs().mean().detach().cpu())))
        #         print("DEBUG logits(tuple):", info)
        #     else:
        #         print("DEBUG logits:", (tuple(logits.shape), float(logits.abs().mean().detach().cpu())))
        # if current_iter % 100 == 0:
        #     gt = self.level.detach().cpu().view(-1)
        #     print("level batch:", gt.tolist(), "unique:", torch.unique(gt).tolist())
        # if current_iter % 1000 == 0:
        #     print("level_hist:", self.level_hist.tolist())
        base = self.opt['train'].get('cls_loss_weight', 0.01)
        start = self.opt['train'].get('cls_warmup_start', 3000)
        ramp  = self.opt['train'].get('cls_warmup_iters', 3000)  # 3000 iter 内爬满
        r = self.opt['train'].get('cls_ratio', 0.1)          # 想让 cls_w ~ 10% pixel
        eps = 1e-8
        w_min = self.opt['train'].get('cls_w_min', 0.0)
        w_max = self.opt['train'].get('cls_w_max', 0.05)     # 上限防止喧宾夺主
        w_inst = (r * l_pix.detach()) / (l_cls.detach() + eps)
        w_inst = torch.clamp(w_inst, w_min, w_max)
        if not hasattr(self, 'cls_w_ema'):
            self.cls_w_ema = w_inst
        ema = self.opt['train'].get('cls_w_ema', 0.9)        # 0.9~0.99 都行
        self.cls_w_ema = ema * self.cls_w_ema + (1 - ema) * w_inst
        w = self.cls_w_ema
        l_cls_w = w * l_cls
        loss_dict['l_cls_w'] = l_cls_w
        loss_dict['cls_w']   = w.detach() 

        # loss = l_pix + l_cls_w   # + 你其它 loss
        # if current_iter < start:
        #     w_cls = 0.0
        # else:
        #     t = min(1.0, (current_iter - start) / float(ramp))
        #     w_cls = base * t
        # loss_dict['l_cls_w'] = w_cls * l_cls
        # loss_total = l_pix + l_pear + l_cls_w 
        loss_total = l_cls * 1.0 + l_pix * 0.1 # 给一点点 pix loss 作为 backbone 的正则化
        loss_dict['loss_total'] = loss_total
        if current_iter % 500 == 0 and logits is not None and self.level is not None:
            gt = self.level.view(-1).long()
            p = torch.sigmoid(logits)
            pred = (p > 0.5).long().sum(dim=1)
            acc = (pred == gt).float().mean().item()
            print("cls_acc:", acc, "logits_mean_abs:", float(logits.abs().mean().detach().cpu()))
            mae = (pred - gt).abs().float().mean().item()
            print("cls_acc:", acc, "mae:", mae)

        if current_iter % 200 == 0 and logits is not None and self.level is not None:
            gt = self.level.view(-1).long()
            p = torch.sigmoid(logits)
            pred = (p > 0.5).long().sum(dim=1)
            print("p_mean:", p.mean(0).detach().cpu().numpy(), "pred_hist:", torch.bincount(pred, minlength=4))


            acc = (pred == gt).float().mean().item()
            print("gt:", gt.tolist(), "pred:", pred.tolist(),
                "unique_pred:", torch.unique(pred).tolist(),
                "cls_acc:", acc)

        loss_total.backward()
        
        if self.opt['train']['use_grad_clip']:
            try:
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01, error_if_nonfinite=False)
            except TypeError:
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        self.log_dict, self.loss_total = self.reduce_loss_dict(loss_dict)
        self.loss_dict = loss_dict
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def pad_test(self, window_size):        
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq      
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred = self.net_g(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
            self.net_g.train()
    def forward_with_pad_and_cls(self, img, window_size=0):
        """Forward once, optionally pad to window_size, return (output, logits)."""
        if window_size and window_size > 0:
            _, _, h, w = img.size()
            mod_pad_h = (window_size - h % window_size) % window_size
            mod_pad_w = (window_size - w % window_size) % window_size
            img_pad = F.pad(img, (0, mod_pad_w, 0, mod_pad_h), mode='reflect')
        else:
            mod_pad_h = 0
            mod_pad_w = 0
            img_pad = img

        # 选择用哪个网络做验证输出：如果你希望验证/保存图跟原来一致，就保持原来的逻辑
        # 原来的 test() 有 net_g_ema 就用 ema，否则用 net_g
        net = self.net_g_ema if hasattr(self, 'net_g_ema') else self.net_g
        was_training = net.training
        net.eval()
        output, logits = net(img_pad, level_gt=None, return_cls=True)

        # crop 回原图尺寸（保证和 pad_test 一致）
        if mod_pad_h > 0 or mod_pad_w > 0:
            output = output[:, :, :output.shape[2] - mod_pad_h, :output.shape[3] - mod_pad_w]
        if was_training:
            net.train()
        return output, logits

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        # pbar = tqdm(total=len(dataloader), unit='image')

        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0
        val_loss_sum = OrderedDict()
        val_loss_cnt = 0

        for idx, val_data in enumerate(dataloader):
            if idx >= 60:
                break
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            _mixing_backup = self.mixing_flag
            self.mixing_flag = False
            self.feed_data(val_data)
            self.mixing_flag = _mixing_backup
            # test()
            with torch.no_grad():
                # 你现在的 test() 只跑了 restoration forward
                # 为了拿 logits，你需要在 val 也用 return_cls=True 跑一次
                # 最省事：直接再 forward 一次（只用于算 loss，不保存图）
                # output, logits = self.net_g(self.lq, level_gt=self.level, return_cls=True)

                # 用刚 forward 的 output 替换 self.output（保证一致）
                output, logits = self.forward_with_pad_and_cls(self.lq, window_size=window_size)
                self.output = output

                # 计算 loss
                ld = self.compute_losses(output, self.gt, logits, self.level)

            for k, v in ld.items():
                val_loss_sum[k] = val_loss_sum.get(k, 0.0) + float(v.detach().cpu())
            val_loss_cnt += 1
            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                
                if self.opt['is_train']:
                    
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')
                    
                    save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}_gt.png')
                else:
                    
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png')
                    save_gt_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_gt.png')
                    
                imwrite(sr_img, save_img_path)
                imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1
        if tb_logger and val_loss_cnt > 0:
            for k, s in val_loss_sum.items():
                avg = s / val_loss_cnt
                tb_logger.add_scalar(f'losses_val/{k}', avg, current_iter)
                if hasattr(self, 'current_epoch'):
                    tb_logger.add_scalar(f'losses_val/{k}_epoch', avg, self.current_epoch)
        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = max(current_metric, self.metric_results[metric])

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if metric == 'psnr' and value >= self.psnr_best:
                self.save(0, current_iter, best=True)
                self.psnr_best = value
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)
                tb_logger.add_scalar(f'metrics/{metric}_epoch', value, self.current_epoch)
        if hasattr(self, 'current_epoch'):
                    for metric, value in self.metric_results.items():
                        tb_logger.add_scalar(f'metrics/{metric}_epoch', value, self.current_epoch)
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter, best=False):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'], best=best)
        else:
            self.save_network(self.net_g, 'net_g', current_iter, best=best)
        self.save_training_state(epoch, current_iter, best=best)
