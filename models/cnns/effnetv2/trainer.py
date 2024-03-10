import os
import glob
import logging

import torch
import wandb
import numpy as np
import timm
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from model import Cnn_Model
from utils.loss import PANNsLoss
from utils.utils import Mixup, do_mixup
from utils.shrink_perturb import shrink_perturb
from utils.scores import model_accuracy, model_f1_score


device = ('cuda' if torch.cuda.is_available() else 'cpu')

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

# Logger
log = logging.getLogger(__name__)

def train_model(
    train_dataloader,
    val_dataloader,
    cfg,
    seed: int,
    cv: str
) -> None:

    if cfg.data.apply_norm:
        log.info('Using Gain Normalization')
    if cfg.data.use_mixup:
        log.info('Using Mix up')
    if cfg.data.use_specaug:
        log.info('Using SpecAugment')
    if cfg.data.imagenet_pretrained:
        log.info('ImageNet Pre-Trained')
    if cfg.train.re_initialize:
        log.info('Using Re-Initialization')
    if cfg.data.tomato_pretrained:
        log.info('Tomato Pre-Trained')
        model = Cnn_Model(
            encoder=cfg.train.encoder,
            classes_num=16,
            imagenet_pretrained=cfg.data.imagenet_pretrained
        )

        # weight_path = glob.glob(os.path.join(config.config.pre_train_output_dir, '**', '*.pth'))
        # print(weight_path)
        # model.load_state_dict(torch.load(weight_path[0])['model_state_dict'])
        # model.encoder.classifier = torch.nn.Linear(in_features=1792, out_features=classes_num, bias=True)

    else:
        model = Cnn_Model(
            encoder=cfg.train.encoder,
            classes_num=cfg.train.classes_num,
            imagenet_pretrained=cfg.data.imagenet_pretrained
        )

    # print_trainable_parameters(model)
    # exit()

    os.makedirs(os.path.join(cfg.data.output_dir, str(seed), cv), exist_ok=True)

    model.to(device)

    criterion = PANNsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.lr,
        amsgrad=True
    )

    wandb.watch(model, criterion, log='all', log_freq=10)

    loss_min = np.Inf
    p = 0

    num_train_steps = int(len(train_dataloader) * cfg.train.epochs)
    num_warmup_steps = int(0.1 * cfg.train.epochs * len(train_dataloader))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)

    model.train()

    for e in tqdm(range(cfg.train.epochs)):
        if cfg.train.re_initialize and e%cfg.train.re_initialize_steps == 0 and e!=0:
            log.info("Shrink Perturb")
            model = shrink_perturb(model, shrink_alpha=0.6, perturb_gama=0.01, cfg=cfg)
            model = model.train()
        running_loss = 0
        train_accuracy = 0
        train_f1_score = 0
        for train_batch_count, sample in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            train_audio, train_label = sample['image'].to(device), sample['target'].to(device)

            optimizer.zero_grad()

            out = model(train_audio, use_specaug=cfg.data.use_specaug)

            train_loss = criterion(out, train_label)

            train_loss.backward()

            optimizer.step()

            if scheduler and cfg.data.step_scheduler:
                scheduler.step()

            running_loss += train_loss.item()

            train_accuracy += model_accuracy(train_label, out)
            train_f1_score += model_f1_score(train_label, out)

            train_batch_count += 1

            if (train_batch_count % 2) == 0:
                wandb.log({"train_loss": train_loss.item()})

        else:

            val_loss = 0
            val_accuracy = 0
            val_f1_score = 0
            with torch.no_grad():
                model.eval()

                for val_batch_count, val_sample in enumerate(val_dataloader):
                    val_audio, val_label = val_sample['image'].to(device), val_sample['target'].to(device)

                    out = model(val_audio)

                    loss = criterion(out, val_label)

                    val_accuracy += model_accuracy(val_label, out)
                    val_f1_score += model_f1_score(val_label, out)

                    val_loss += loss.item()

                    if (val_batch_count % 2) == 0:
                        wandb.log({"val_loss": loss.item()})

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

            if val_loss/len(val_dataloader) < loss_min:
                log.info("Validation Loss Decreasead ({:.6f} --> {:.6f}), saving model...".format(loss_min, val_loss/len(val_dataloader)))
                loss_min = val_loss/len(val_dataloader)
                torch.save({'epoch': cfg.train.epochs,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': criterion
                            }, os.path.join(cfg.data.output_dir, str(seed), cv, f'{cv}-epochs_{cfg.train.epochs}-period_{cfg.train.period}-train_{cfg.data.frac_train}-test_{cfg.data.frac_test}.pth'))

            else:
                p += 1

                if p == cfg.train.patience:
                    log.info("Early Stopping... ")
                    break

            model.train()