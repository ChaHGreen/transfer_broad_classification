import math
from typing import Optional
from system import utils_system
import torch
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from system import system_abstract
from .utils_system import concat_all_ddp
import uncertainty_metrics.numpy as um
import torchvision


# ---------------------------------------------------------------------------- #
#                               lightning module                               #
# ---------------------------------------------------------------------------- #
class LightningSystem(system_abstract.LightningSystem):
    def __init__(self, hparams, datamodule=None):
        super().__init__(hparams, datamodule)

        self.dm = datamodule
        self.conf_data = self.dm.conf_data

        self._create_model(self.num_classes)
        self.drop_fn = torch.nn.Dropout(
            self.hparams.linear_drop
        ) if self.hparams.linear_drop > 0 else torch.nn.Identity()

        # had to add this for some version issue
        if self.hparams.ckpt is not None and "base_plus" in self.hparams.ckpt:
            self.base_head = self.classifier

        # if self.hparams.pretrained:
        #     clsfr = getattr(torchvision.models,
        #                     self.hparams.model)(pretrained=True).fc
        #     if self.hparams.linear_bn:
        #         self.classifier[1].weight.data.copy_(clsfr.weight.data)
        #     else:
        #         self.classifier.weight.data.copy_(clsfr.weight.data)

    def forward(self, x):
        if len(x) == 2 and (isinstance(x, list) or isinstance(x, tuple)):
            x = x[0]
        out = self.feature.forward(x)
        if self.hparams.use_norm:
            out = out / (out.norm(dim=-1, keepdim=True) + 1e-6)
        out = self.drop_fn(out)
        scores = self.classifier.forward(out)
        return scores, out

    def freeze_backbone(self):
        # NOTE: freeze backbone
        for par in self.feature.parameters():
            par.requires_grad = False

        if self.hparams.unfreeze_lastk != 0:
            # first unfreeze all, then freeze layers
            for par in self.feature.parameters():
                par.requires_grad = True
            if self.hparams.unfreeze_lastk != -1:
                layers_to_freeze = self.feature[:-2 -
                                                self.hparams.unfreeze_lastk]
                for par in layers_to_freeze.parameters():
                    par.requires_grad = False
        else:
            """
            Switch to eval mode:
            Under the protocol of linear classification on frozen features/models,
            it is not legitimate to change any part of the pre-trained model.
            BatchNorm in train mode may revise running mean/std (even if it receives
            no gradient), which are part of the model parameters too.
            """
            self.feature.eval()

        # set all BatchNorm to eval mode
        if self.hparams.set_bn_to_eval == 'true':
            utils_system.set_bn_to_eval(self.feature)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = system_abstract.LightningSystem.add_model_specific_args(
            parent_parser)
        parser.add_argument("--model_name", type=str, default="linear_eval")
        parser.add_argument("--lr", type=float, default=0.05)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--eval_mode", type=str, default="linear")
        parser.add_argument("--warm_up",
                            type=str,
                            default='false',
                            choices=['true', 'false'],
                            help="whether to lr warm-up")

        parser.add_argument("--check_val_every_n_epoch", type=int, default=5)
        parser.add_argument("--pretrained", action="store_true")
        parser.add_argument("--max_epochs", type=int, default=100)
        parser.add_argument("--optimizer", type=str, default="SGD")
        parser.add_argument("--optim_wd", type=float, default=0.0)

        parser.add_argument("--num_classes", type=int, default=None)

        parser.add_argument("--dataset", default='CIFAR100_train', type=str)
        parser.add_argument('--val_dataset',
                            default='CIFAR100_test',
                            type=str,
                            help='val dataset to use')

        parser.add_argument('--train_aug', type=str, default='true')
        parser.add_argument('--val_aug', type=str, default='false')
        parser.add_argument(
            "--unfreeze_lastk",
            type=int,
            default=0,
            help="unfreeze k layers (default:0: all unfreeze, 0:no_unfreeze)")
        parser.add_argument(
            "--unfreeze_warmup_epoch",
            type=int,
            default=10,
            help="disable updating feature layers for this epoch")

        parser.add_argument("--set_bn_to_eval",
                            type=str,
                            default='false',
                            choices=['true', 'false'])
        parser.add_argument("--linear_drop", type=float, default=0)
        parser.add_argument("--linear_feature_multiplier",
                            type=float,
                            default=0.01,
                            help=("multiplier for caluclating lr"
                                  "of feature branch from classifier branch"))
        parser.add_argument("--optim_momentum", type=float, default=0.9)
        return parser

    # --------------------------------- training --------------------------------- #
    def on_epoch_start(self) -> None:
        super().on_epoch_start()
        self.freeze_backbone()

    def training_step(self, batch, batch_idx):
        assert self.feature.training == False
        x, y = batch

        scores, _ = self(x)
        loss = torch.nn.functional.cross_entropy(scores, y)

        with torch.no_grad():
            _, predicted = torch.max(scores.data, 1)
            accur = accuracy(predicted.detach(), y)

        tqdm_dict = {
            "loss_train": loss,
            "top1": accur,
        }

        self.log_dict(tqdm_dict,
                      prog_bar=True,
                      logger=True,
                      on_step=True,
                      on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        scores, _ = self(x)
        loss = torch.nn.functional.cross_entropy(scores, y)
        prob = scores.softmax(dim=-1)

        result = {}
        result["val_loss"] = loss
        result["gt"] = y
        result["prob"] = prob

        return result

    def validation_epoch_end(self, outputs):
        outputs = utils_system.aggregate_dict(outputs)
        mean_loss = outputs['val_loss'].mean()

        # concat all prediction and gt from different ddp processeses
        epoch_probs = concat_all_ddp(outputs["prob"])
        epoch_gt = concat_all_ddp(outputs["gt"])
        epoch_preds = torch.argmax(epoch_probs, dim=-1)

        mean_accuracy = accuracy(epoch_preds, epoch_gt)
        per_class_accuracy = accuracy(epoch_preds,
                                      epoch_gt,
                                      class_reduction='none').mean()
        if mean_accuracy > self.best_metric[0][0]:
            self.best_metric[0] = [mean_accuracy.detach(), self.current_epoch]
        self.logger.log(
            f"Best metric ({self.val_names[0]}): {self.best_metric[0][0]} @ {self.best_metric[0][1]}",
            "INFO")

        self.logger.log(f"mean_accuracy: {mean_accuracy: .4f}", "CRITICAL")
        ece_score = self.ece_calculate(epoch_probs, epoch_gt)

        per_class_accuracy_1 = accuracy(epoch_preds,
                                      epoch_gt,
                                      num_classes=self.num_classes,
                                      class_reduction='none') ##这个是返回各类精度

        info = {
            "acc_mean": mean_accuracy,
            "acc_per_class": per_class_accuracy,
            "mean_loss": mean_loss,
            "validation_ece": ece_score
        }
        accuracy_log=''
        for i in range(0,self.num_classes):
            accuracy_log=accuracy_log+f'  |  {i} : {per_class_accuracy_1[i]:.4f}'
            info[f'{i}']=per_class_accuracy_1[i]
        self.logger.log(f'Class Accauracy: {accuracy_log}','CRITICAL')

        self.log_dict(info, logger=True, prog_bar=True)

        self.logger.log_csv(info, step=self.global_step)

    # --------------------------------- optimizer -------------------------------- #
    def configure_optimizers(self):
        param_groups = [{
            'params': self.classifier.parameters(),
            'lr': self.hparams.lr,
            'momentum': self.hparams.optim_momentum,
            'weight_decay': self.hparams.optim_wd
        }]
        if self.hparams.optimizer == 'SGD':
            optimizer = torch.optim.SGD(param_groups)
        elif self.hparams.optimizer == 'Adam':
            optimizer = torch.optim.Adam(param_groups)
        else:
            raise NotImplementedError
        return optimizer

    def optimizer_step(self,
                       epoch,
                       batch_idx,
                       optimizer,
                       optimizer_idx,
                       optimizer_closure=None,
                       *args,
                       **kwargs):
        self.adjust_learning_rate(optimizer, epoch, self.hparams.lr)

        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def adjust_learning_rate(self, optimizer, epoch,
                             base_lr) -> Optional[bool]:
        if epoch < 0:
            return False
        if self.hparams.scheduler == 'cosine':
            eta_min = 1e-8
            lr = eta_min + (base_lr - eta_min) * (
                1 + math.cos(math.pi * epoch / self.hparams.max_epochs)) / 2
            for _, param_group in enumerate(optimizer.param_groups):
                param_group['lr'] = lr
        elif self.hparams.scheduler == 'step':
            if not hasattr(self, '_milestones'):
                self._milestones = {
                    int(_v): i + 1
                    for i, _v in enumerate(
                        self.hparams.step_lr_milestones.split(","))
                }
            if epoch in self._milestones:
                for param_group in optimizer.param_groups:
                    _lr = (base_lr *
                           self.hparams.lr_decay_rate**self._milestones[epoch])
                    param_group['lr'] = _lr
        return

    # --------------------------------- val data --------------------------------- #
    def prepare_data(self):
        self.dm.prepare_data()

    def val_dataloader(self):
        if self.hparams.val_as_test == 'true':
            return self.dm.val_dataloader_val_split(pl_trainer=self.trainer,
                                                    use_ddp=self.use_ddp)
        base_loader = self.dm.get_simple_dataloader(self.hparams.val_dataset,
                                                    aug=self.hparams.val_aug,
                                                    pl_trainer=self.trainer,
                                                    use_ddp=self.use_ddp,
                                                    opt=self.hparams,
                                                    drop_last=False,
                                                    shuffle=False)
        return base_loader

    def test_dataloader(self):
        return self.val_dataloader()