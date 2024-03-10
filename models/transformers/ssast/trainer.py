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

from utils.losses import BCELossModified, clip_ce
from ast_models import ASTModel
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

    model = ASTModel(
        label_dim=cfg.train.classes_num,
        fshape=cfg.model.fshape,
        tshape=cfg.model.tshape,
        fstride=cfg.model.fstride,
        tstride=cfg.model.tstride,
        input_fdim=cfg.feature_extractor.mel_bins,
        input_tdim=cfg.feature_extractor.target_length,
        model_size=cfg.model.model_size,
        pretrain_stage=False,
        load_pretrained_mdl_path=cfg.model.pretrained_checkpoint_path
    )

    print_trainable_parameters(model)
    exit()

    model.to(device)

    trainables = [p for p in model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))

    # diff lr optimizer
    mlp_list = ['mlp_head.0.weight', 'mlp_head.0.bias', 'mlp_head.1.weight', 'mlp_head.1.bias']
    mlp_params = list(filter(lambda kv: kv[0] in mlp_list, model.named_parameters()))
    base_params = list(filter(lambda kv: kv[0] not in mlp_list, model.named_parameters()))
    mlp_params = [i[1] for i in mlp_params]
    base_params = [i[1] for i in base_params]
    # only finetuning small/tiny models on balanced audioset uses different learning rate for mlp head
    print('The mlp header uses {:d} x larger lr'.format(1))
    print('Total mlp parameter number is : {:.3f} million'.format(sum(p.numel() for p in mlp_params) / 1e6))
    print('Total base parameter number is : {:.3f} million'.format(sum(p.numel() for p in base_params) / 1e6))

    os.makedirs(os.path.join(cfg.data.output_dir, str(seed), cv), exist_ok=True)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        [{'params': base_params, 'lr': cfg.train.lr}, {'params': mlp_params, 'lr': cfg.train.lr * cfg.train.head_lr}],
        weight_decay=cfg.train.weight_decay,
        betas=(0.95, 0.999)
    )

    # Save gradients of the weights
    wandb.watch(model, criterion, log='all', log_freq=10)

    loss_min = np.Inf

    # Initialize early stopping
    p = 0

    # Initialize scheduler
    num_train_steps = int(len(train_dataloader) * cfg.train.epochs)
    num_warmup_steps = int(0.1 * cfg.train.epochs * len(train_dataloader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps
    )

    model.train()

    for e in tqdm(range(cfg.train.epochs)):
        running_loss = 0
        train_accuracy = 0
        train_f1_score = 0
        for train_batch_count, sample in enumerate(tqdm(train_dataloader)):
            train_audio, train_label = sample['image'].to(device), sample['hot_target'].to(device)

            optimizer.zero_grad()

            out = model(
                train_audio,
                cfg.train.task
            )

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

                    out = model(val_audio, cfg.train.task)

                    loss = criterion(out, val_label)

                    val_accuracy += model_accuracy(val_label, out)
                    val_f1_score += model_f1_score(val_label, out)

                    val_loss += loss

                    if (val_batch_count % 2) == 0:
                        wandb.log({"val_loss": loss})

            # Log results on wandb
            wandb.log({"train_acc": (train_accuracy/len(train_dataloader))*100,
                        "val_acc": (val_accuracy/len(val_dataloader))*100,
                        "train_f1": train_f1_score/len(train_dataloader)*100,
                        "val_f1": val_f1_score/len(val_dataloader)*100,
                        "epoch": e})

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
