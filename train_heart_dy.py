import torch
from torch import nn
from opt import get_opts
import os
import imageio
import numpy as np
from einops import rearrange

# data
from torch.utils.data import DataLoader
from datasets import dataset_dict
from datasets.points_utils import axisangle_to_R, get_points

# models
from models.networks_dy_siren import Siren,TimeEmbedding
# optimizer, losses
from apex.optimizers import FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
from losses_dy import loss_dict
from models.rendering_dy import render
# metrics
from torchmetrics import PeakSignalNoiseRatio

# pytorch-lightning
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.distributed import all_gather_ddp_if_available

from utils import slim_ckpt, load_ckpt

import warnings; warnings.filterwarnings("ignore")

class NeuralCMFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.loss = loss_dict['NeuralCMF']()
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.model = Siren()
        self.embedding_t = TimeEmbedding(hparams.T)
        self.output_transient_flow = ['fw', 'bw']

    def forward(self, batch, split):
        if split=='train':
            poses = self.poses[batch['img_idxs']]
            ini_position = self.ini_position[batch['pix_idxs']]
            ts = batch['t']
        else:
            poses = batch['pose']
            ts = batch['t']
            poses = poses.unsqueeze(0)
            ini_position = self.ini_position

        dR = axisangle_to_R(self.dR[batch['img_idxs']])
        poses[..., :3] = dR @ poses[..., :3]
        poses[..., 3] += self.dT[batch['img_idxs']]

        position = get_points(ini_position, poses)

        return render(self.model,position,self.embedding_t,self.output_transient_flow,ts,hparams.T)

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir}
        self.train_dataset = dataset(split=self.hparams.split, **kwargs)
        self.train_dataset.batch_size = self.hparams.batch_size
        self.test_dataset = dataset(split='test', **kwargs)

    def configure_optimizers(self):
        self.register_buffer('ini_position', self.train_dataset.ini_position.to(self.device))
        self.register_buffer('poses', self.train_dataset.poses.to(self.device))

        N = len(self.train_dataset.poses)
        self.register_parameter('dR',
            nn.Parameter(torch.zeros(N, 3, device=self.device)))
        self.register_parameter('dT',
            nn.Parameter(torch.zeros(N, 3, device=self.device)))

        load_ckpt(self.model, self.hparams.weight_path)

        net_params = []
        for n, p in self.named_parameters():
            if n not in ['dR', 'dT']: net_params += [p]

        opts = []
        self.net_opt = FusedAdam(net_params, self.hparams.lr, eps=1e-15)
        opts += [self.net_opt]

        opts += [FusedAdam([self.dR, self.dT], 1e-6)] # learning rate is hard-coded
        net_sch = CosineAnnealingLR(self.net_opt,
                                    self.hparams.num_epochs,
                                    self.hparams.lr/30)

        return opts, [net_sch]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=16,
                          persistent_workers=True,
                          batch_size=None,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=8,
                          batch_size=None,
                          pin_memory=True)

    def training_step(self, batch, batch_nb, *args):
        results = self(batch, split='train') ##change
        #print(results)
        loss_d = self.loss(results, batch)
        loss = sum(lo for lo in loss_d.values())

        with torch.no_grad():
            self.train_psnr(results['gray_fine'], batch['gray'])
        self.log('lr', self.net_opt.param_groups[0]['lr'])
        self.log('train/loss', loss)
        self.log('train/psnr', self.train_psnr, True)
        for k, v in loss_d.items(): self.log(f'train/{k}',v,prog_bar=True)

        return loss

    def on_validation_start(self):
        torch.cuda.empty_cache()
        if not self.hparams.no_save_test:
            self.val_dir = f'results/{self.hparams.dataset_name}/{self.hparams.exp_name}'
            os.makedirs(self.val_dir, exist_ok=True)

    def validation_step(self, batch, batch_nb):
        gray_gt = batch['gray']
        results = self(batch, split='test')

        logs = {}
        # compute each metric per image
        self.val_psnr(results['gray_fine'], gray_gt)
        logs['psnr'] = self.val_psnr.compute()
        self.val_psnr.reset()
        w, h = self.train_dataset.img_wh
        if not self.hparams.no_save_test: # save test image to disk
            idx = batch['img_idxs']
            gray_pred = rearrange(results['gray_fine'].cpu().numpy(), '(h w) c -> h w c', h=h)
            gray_pred = (gray_pred*255).astype(np.uint8)
            imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}.png'), gray_pred)
        return logs

    def validation_epoch_end(self, outputs):
        psnrs = torch.stack([x['psnr'] for x in outputs])
        mean_psnr = all_gather_ddp_if_available(psnrs).mean()
        self.log('test/psnr', mean_psnr, True)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


if __name__ == '__main__':
    hparams = get_opts()
    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')
    system = NeuralCMFSystem(hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.dataset_name}/{hparams.exp_name}',
                              filename='{epoch:d}',
                              save_weights_only=True,
                              every_n_epochs=hparams.num_epochs,
                              save_on_train_epoch_end=True,
                              save_top_k=-1)
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

    logger = TensorBoardLogger(save_dir=f"logs/{hparams.dataset_name}",
                               name=hparams.exp_name,
                               default_hp_metric=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      check_val_every_n_epoch=hparams.num_epochs,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='gpu',
                      devices=hparams.num_gpus,
                      strategy=DDPPlugin(find_unused_parameters=False)
                               if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=-1 if hparams.val_only else 0,
                      precision=16)

    trainer.fit(system, ckpt_path=hparams.ckpt_path)

    if not hparams.val_only: # save slimmed ckpt for the last epoch
        ckpt_ = \
            slim_ckpt(f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}.ckpt',
                      save_poses=True)
        torch.save(ckpt_, f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}_slim.ckpt')
