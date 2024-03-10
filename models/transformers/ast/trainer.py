import os
import logging

import torch
import wandb
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup

from models.transformers.ast.utils.ast_models import ASTModel
from models.transformers.ast.utils.scores import model_accuracy, model_f1_score

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
    train_dataloader,
    val_dataloader,
    cfg,
    seed: int,
    cv: str
) -> None:

    model = ASTModel(
        label_dim=cfg.data.class_num,
        input_tdim=cfg.model.input_tdim,
        imagenet_pretrain=cfg.model.imagenet_pretrained,
        audioset_pretrain=cfg.model.audioset_pretrained
    )

    model.to(device)

    print_trainable_parameters(model)
    exit()

    os.makedirs(os.path.join(cfg.metadata.output_dir, str(seed), cv), exist_ok=True)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.lr
    )

    wandb.watch(model, criterion, log='all', log_freq=10)

    loss_min = np.Inf
    p = 0

    # Linear Schedule with Warmup
    num_train_steps = int(len(train_dataloader) * cfg.train.epochs)
    num_warmup_steps = int(5 * len(train_dataloader))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)

    scaler = GradScaler(enabled=True)

    model.train()

    for e in tqdm(range(cfg.train.epochs)):
        running_loss = 0
        train_accuracy = 0
        train_f1_score = 0
        for train_batch_count, sample in enumerate(tqdm(train_dataloader)):
            train_audio, train_label = sample['image'].to(device), sample['target'].to(device)

            optimizer.zero_grad()

            with autocast(enabled=True):
                out = model(train_audio)
                train_loss = criterion(out, train_label)

            scaler.scale(train_loss).backward()

            scaler.step(optimizer)

            scaler.update()

            # Linear Schedule with Warmup
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
                    val_audio, val_label = val_sample['image'].to(device), val_sample['target'].to(device)

                    out = model(val_audio)

                    loss = criterion(out, val_label)

                    val_accuracy += model_accuracy(val_label, out)
                    val_f1_score += model_f1_score(val_label, out)

                    val_loss += loss

                    if (val_batch_count % 2) == 0:
                        wandb.log({"val_loss": loss})

            wandb.log(
                {
                    "train_acc": (train_accuracy/len(train_dataloader))*100,
                    "train_f1-score": (train_f1_score/len(train_dataloader))*100,
                    "val_acc": (val_accuracy/len(val_dataloader))*100,
                    "val_f1-score": (val_f1_score/len(val_dataloader))*100,
                    "epoch": e
                }
            )

            log.info(
                'Train Accuracy: {:.3f} | Train F1-Score: {:.3f} | Train Loss: {:.6f} | Val Accuracy: {:.3f} | Val F1-Score: {:.3f} | Val loss: {:.6f}'.format(
                    (train_accuracy/len(train_dataloader))*100,
                    (train_f1_score/len(train_dataloader))*100,
                    running_loss/len(train_dataloader),
                    (val_accuracy/len(val_dataloader))*100,
                    (val_f1_score/len(val_dataloader))*100,
                    val_loss/len(val_dataloader)
                )
            )

            log.info(f"Learning Rate: {optimizer.param_groups[0]['lr']}")

            if val_loss/len(val_dataloader) < loss_min:
                log.info("Validation Loss Decreasead ({:.6f} --> {:.6f}), saving model...\n".format(loss_min, val_loss/len(val_dataloader)))
                loss_min = val_loss/len(val_dataloader)
                torch.save(
                    {
                        'epoch': cfg.train.epochs,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': criterion
                    },
                    os.path.join(cfg.metadata.output_dir, str(seed), cv, f'{cv}-epochs_{cfg.train.epochs}-period_{cfg.data.period}.pth')
                )

            else:
                p += 1

                if p == cfg.train.patience:
                    log.info("Early Stopping... ")
                    break

            model.train()