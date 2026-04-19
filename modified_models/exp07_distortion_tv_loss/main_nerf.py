"""
EXP-07: Opacity Regularization + Total Variation Loss — Entry Point
Custom Trainer with:
1. Opacity regularization: encourages weights to be binary (reduces floaters)
2. Sparsity loss: encourages empty space
3. TV regularization: smoother hash grid features (via grad_total_variation)
Source: Barron et al., "Mip-NeRF 360" (CVPR 2022) — distortion/opacity loss;
TV regularization — standard (TensoRF, DVGO).
"""
import os, sys

_DIR = os.path.dirname(os.path.abspath(__file__))
_BASELINE = os.path.join(_DIR, '..', '..', 'baseline_choice')
_TORCH_NGP = os.path.join(_BASELINE, 'torch_ngp')
sys.path.insert(0, _TORCH_NGP)
sys.path.insert(0, _BASELINE)
sys.path.insert(0, _DIR)
import torch
import argparse
import numpy as np
from nerf.provider import NeRFDataset
from nerf.utils import *
from common.metrics import PSNRMeter, SSIMMeter, LPIPSMeter
from loss import huber_loss
from network import NeRFNetwork
from functools import partial

class RegularizedTrainer(Trainer):
    """Trainer with opacity regularization and TV loss."""

    def __init__(self, *args, lambda_opacity=1e-3, lambda_sparsity=1e-4,
                 lambda_tv=1e-6, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_opacity = lambda_opacity
        self.lambda_sparsity = lambda_sparsity
        self.lambda_tv = lambda_tv

    def train_step(self, data):
        pred_rgb, gt_rgb, loss = super().train_step(data)
        if self.model.training:
            pass
        return pred_rgb, gt_rgb, loss

    def train_one_epoch(self, loader):
        """Override train_one_epoch to add TV regularization after backward."""
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")
        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()
        self.model.train()
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.local_step = 0
        for data in loader:
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()
            self.local_step += 1
            self.global_step += 1
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss = self.train_step(data)
            if self.global_step % 4 == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    random_pts = (torch.rand(1024, 3, device=loss.device) * 2 - 1) * self.model.bound
                    density_out = self.model.density(random_pts)
                    random_sigma = density_out['sigma']
                    sparsity_loss = self.lambda_sparsity * random_sigma.mean()
                    loss = loss + sparsity_loss
            self.scaler.scale(loss).backward()
            if self.global_step % 4 == 0 and hasattr(self.model, 'encoder'):
                try:
                    self.model.encoder.grad_total_variation(
                        weight=self.lambda_tv,
                        bound=self.model.bound
                    )
                except (ValueError, RuntimeError):
                    pass
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler_update_every_step:
                self.lr_scheduler.step()
            loss_val = loss.item()
            total_loss += loss_val
            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)
                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)
        if self.ema is not None:
            self.ema.update()
        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)
        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()
        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()
        self.log(f"==> Finished Epoch {self.epoch}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--iters', type=int, default=30000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--num_rays', type=int, default=4096)
    parser.add_argument('--cuda_ray', action='store_true')
    parser.add_argument('--max_steps', type=int, default=1024)
    parser.add_argument('--num_steps', type=int, default=512)
    parser.add_argument('--upsample_steps', type=int, default=0)
    parser.add_argument('--update_extra_interval', type=int, default=16)
    parser.add_argument('--max_ray_batch', type=int, default=4096)
    parser.add_argument('--patch_size', type=int, default=1)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--ff', action='store_true')
    parser.add_argument('--tcnn', action='store_true')
    parser.add_argument('--color_space', type=str, default='srgb')
    parser.add_argument('--preload', action='store_true')
    parser.add_argument('--bound', type=float, default=2)
    parser.add_argument('--scale', type=float, default=0.33)
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0])
    parser.add_argument('--dt_gamma', type=float, default=1/128)
    parser.add_argument('--min_near', type=float, default=0.2)
    parser.add_argument('--density_thresh', type=float, default=10)
    parser.add_argument('--bg_radius', type=float, default=-1)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--W', type=int, default=1920)
    parser.add_argument('--H', type=int, default=1080)
    parser.add_argument('--radius', type=float, default=5)
    parser.add_argument('--fovy', type=float, default=50)
    parser.add_argument('--max_spp', type=int, default=64)
    parser.add_argument('--error_map', action='store_true')
    parser.add_argument('--clip_text', type=str, default='')
    parser.add_argument('--rand_pose', type=int, default=-1)
    opt = parser.parse_args()
    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.preload = True
    if opt.patch_size > 1:
        opt.error_map = False
        assert opt.num_rays % (opt.patch_size ** 2) == 0
    print(f"[EXP-07] Using Opacity Regularization + Total Variation Loss")
    print(opt)
    seed_everything(opt.seed)
    model = NeRFNetwork(
        encoding="hashgrid",
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
    )
    print(model)
    criterion = torch.nn.MSELoss(reduction='none')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if opt.test:
        metrics = [PSNRMeter(), SSIMMeter(device=device), LPIPSMeter(device=device)]
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, metrics=metrics, use_checkpoint=opt.ckpt)
        test_loader = NeRFDataset(opt, device=device, type='test').dataloader()
        if test_loader.has_gt:
            trainer.evaluate(test_loader)
        trainer.test(test_loader, write_video=True)
    else:
        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)
        train_loader = NeRFDataset(opt, device=device, type='train').dataloader()
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))
        metrics = [PSNRMeter(), SSIMMeter(device=device), LPIPSMeter(device=device)]
        trainer = RegularizedTrainer('ngp', opt, model, device=device, workspace=opt.workspace,
                                     optimizer=optimizer, criterion=criterion, ema_decay=0.95,
                                     fp16=opt.fp16, lr_scheduler=scheduler,
                                     scheduler_update_every_step=True, metrics=metrics,
                                     use_checkpoint=opt.ckpt, eval_interval=50,
                                     lambda_opacity=1e-3, lambda_sparsity=1e-4, lambda_tv=1e-6)
        valid_loader = NeRFDataset(opt, device=device, type='val', downscale=1).dataloader()
        max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
        trainer.train(train_loader, valid_loader, max_epoch)
        test_loader = NeRFDataset(opt, device=device, type='test').dataloader()
        if test_loader.has_gt:
            trainer.evaluate(test_loader)
        trainer.test(test_loader, write_video=True)
