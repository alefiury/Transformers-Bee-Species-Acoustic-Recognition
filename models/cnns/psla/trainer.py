import os
import logging
from numpy import outer

import torch
import wandb
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from omegaconf import DictConfig
from transformers import get_linear_schedule_with_warmup

from utils.adan import Adan
from utils.scheduler import CosineWarmupLR, PolynomialLR
from utils.losses import BCELossModified
from models.Models import EffNetAttention
from utils.scores import model_accuracy, model_f1_score

device = ('cuda' if torch.cuda.is_available() else 'cpu')
# Logger
log = logging.getLogger(__name__)

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    trainable_params_in_millions = trainable_params / 1_000_000
    all_param_in_millions = all_param / 1_000_000
    print(
        f"trainable params: {trainable_params_in_millions:.2f}M || all params: {all_param_in_millions:.2f}M || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def train_model(
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    cfg : DictConfig,
    seed: int,
    cv: str,
) -> None:

    # Create directory to save weights
    os.makedirs(os.path.join(cfg.data.output_dir), exist_ok=True)

    if cfg.data_augmentation.use_audio_mixup:
        log.info('Using Audio Mix up')
    if cfg.data_augmentation.use_specaug:
        log.info('Using SpecAugment')
    if cfg.data_augmentation.rand_sampling:
        log.info('Using Random Sampling')
    if cfg.data_augmentation.insert_noise:
        log.info('Using Noise Insertion')
    log.info(f"Using {cfg.train.optimizer} Optimizer")
    log.info(f"Using {cfg.train.scheduler} Scheduler")

    model = EffNetAttention(
        label_dim=cfg.train.classes_num,
        b=cfg.model.b,
        pretrain=cfg.model.imagenet_pretrained,
        head_num=cfg.model.head_num
    )

    if cfg.model.pretrained:
        log.info("Loading Pre-Trained Model... ")
        checkpoint = torch.load(cfg.model.pretrained_checkpoint_path)
        model.load_state_dict(checkpoint, strict=False)

    model.to(device)

    print_trainable_parameters(model)
    exit()

    os.makedirs(os.path.join(cfg.data.output_dir, str(seed), cv), exist_ok=True)

    # criterion = torch.nn.BCEWithLogitsLoss()
    # criterion = torch.nn.BCELoss()
    criterion = BCELossModified()

    if cfg.train.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.train.lr
    )

    elif cfg.train.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )

    elif cfg.train.optimizer == "adan":
        optimizer = Adan(
            model.parameters(),
            lr=cfg.train.lr,
            weight_decay=0.01,
            betas=(0.98, 0.92, 0.99),
            eps = 1e-8,
            max_grad_norm=0.0,
            no_prox=False
        )
    else:
        raise NotImplementedError(
            f"Optimizer {cfg.train.optimizer} not implemented"
        )

    # Save gradients of the weights
    wandb.watch(model, criterion, log='all', log_freq=10)

    # Initialize scheduler
    num_train_steps = int(len(train_dataloader) * cfg.train.epochs)
    num_warmup_steps = int(cfg.train.warmup_steps * num_train_steps)

    if cfg.train.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps
        )
    elif cfg.train.scheduler == "polynomial":
        scheduler = PolynomialLR(
            optimizer,
            total_iters=num_train_steps
        )
    elif cfg.train.scheduler == "cosine":
        scheduler = CosineWarmupLR(
            optimizer,
            lr_min=1e-5,
            lr_max=cfg.train.lr,
            warmup=num_warmup_steps,
            T_max=num_train_steps
        )
    else:
        raise NotImplementedError(
            f"Scheduler {cfg.train.scheduler} not implemented"
        )

    loss_min = np.Inf
    # Initialize early stopping
    p = 0

    model.train()
    for e in tqdm(range(cfg.train.epochs)):
        running_loss = 0
        train_accuracy = 0
        train_f1_score = 0
        for train_batch_count, sample in enumerate(tqdm(train_dataloader)):
            train_audio, train_label = sample['image'].to(device), sample['hot_target'].to(device)

            optimizer.zero_grad()

            out = model(train_audio)

            train_loss = criterion(out, train_label)
            train_loss.backward()
            optimizer.step()

            if scheduler and cfg.train.step_scheduler:
                scheduler.step()

            running_loss += train_loss

            train_accuracy += model_accuracy(train_label, out)
            train_f1_score += model_f1_score(train_label, out)

            train_batch_count += 1

            if (train_batch_count % 2) == 0:
                wandb.log({"train_loss": train_loss})

        else:
            val_loss = 0
            val_accuracy = 0
            val_f1_score = 0
            with torch.no_grad():
                model.eval()

                for val_batch_count, val_sample in enumerate(val_dataloader):
                    val_audio, val_label = val_sample['image'].to(device), val_sample['hot_target'].to(device)

                    out = model(val_audio)

                    loss = criterion(out, val_label)

                    val_accuracy += model_accuracy(val_label, out)
                    val_f1_score += model_f1_score(val_label, out)

                    val_loss += loss

                    if (val_batch_count % 2) == 0:
                        wandb.log({"val_loss": loss})

            # Log results on wandb
            wandb.log(
                {
                    "train_acc": (train_accuracy/len(train_dataloader))*100,
                    "val_acc": (val_accuracy/len(val_dataloader))*100,
                    "train_f1": train_f1_score/len(train_dataloader)*100,
                    "val_f1": val_f1_score/len(val_dataloader)*100,
                    "epoch": e
                }
            )

            log.info('Train Accuracy: {:.3f} | Train F1-Score: {:.3f} | Train Loss: {:.6f} | Val Accuracy: {:.3f} | Val F1-Score: {:.3f} | Val loss: {:.6f}'.format(
                        (train_accuracy/len(train_dataloader))*100,
                        (train_f1_score/len(train_dataloader))*100,
                        running_loss/len(train_dataloader),
                        (val_accuracy/len(val_dataloader))*100,
                        (val_f1_score/len(val_dataloader))*100,
                        val_loss/len(val_dataloader)))

            log.info(f"LR: {scheduler.get_last_lr()[0]}")

            if val_loss/len(val_dataloader) < loss_min:
                log.info("Validation Loss Decreasead ({:.6f} --> {:.6f}), saving model...\n".format(loss_min, val_loss/len(val_dataloader)))
                loss_min = val_loss/len(val_dataloader)
                torch.save({'epoch': cfg.train.epochs,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': criterion
                            },  os.path.join(cfg.data.output_dir, str(seed), cv, f'{cv}-epochs_{cfg.train.epochs}-period_{cfg.data.period}-train_{cfg.data.frac_train}-test_{cfg.data.frac_test}.pth'))

            else:
                p += 1

                if p == cfg.train.patience:
                    log.info("Early Stopping... ")
                    break

            model.train()
