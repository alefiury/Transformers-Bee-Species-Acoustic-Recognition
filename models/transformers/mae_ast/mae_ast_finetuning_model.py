import logging
from types import SimpleNamespace

import torch
import torch.nn as nn

from models.transformers.mae_ast.mae_ast_model import MAE_AST

log = logging.getLogger(__name__)

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


class Transfer_MAE_AST(nn.Module):
    def __init__(
        self,
        ckpt: str,
        classes_num: int,
        pretrained: bool,
        freeze_base: bool,
        encoder_embedding_dim: int
    ):
        super(Transfer_MAE_AST, self).__init__()

        self.checkpoint = torch.load(ckpt)
        self.base_model = MAE_AST(
            SimpleNamespace(
                **self.checkpoint["cfg"]["model"]
            ),
            SimpleNamespace(
                **self.checkpoint["cfg"]["task"]
            )
        )

        self.fc_transfer = nn.Linear(encoder_embedding_dim, classes_num)

        # self.fc_transfer = nn.Sequential(
        #     nn.LayerNorm(encoder_embedding_dim),
        #     nn.Linear(encoder_embedding_dim, classes_num)
        # )

        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        if pretrained:
            log.info("Loading Pre-Trained Base Model... ")
            self.__load_from_pretrained()
            log.info("Pre-Trained Base Model Loaded")

        self.__init_weights()


    def __init_weights(self):
        init_layer(self.fc_transfer)


    def __load_from_pretrained(self):
        self.base_model.load_state_dict(self.checkpoint["model"], strict=True)


    def forward(
        self,
        input,
        padding_mask,
        mask,
        features_only,
        is_decoder_finetune
    ):
        output_dict = self.base_model(
            input,
            padding_mask=padding_mask,
            mask=mask,
            features_only=features_only,
            is_decoder_finetune=is_decoder_finetune
        )
        embeddings = output_dict['x']

        embeddings = torch.mean(embeddings, dim=1)

        output =  self.fc_transfer(embeddings)

        return output
