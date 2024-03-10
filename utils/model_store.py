import torch
import os
import glob

### Importing the train methods from the models
from models.cnns.effnetv2.trainer import train_model as train_model_effnet
from models.cnns.panns.trainer import train_model as train_model_panns
from models.cnns.psla.trainer import train_model as train_model_psla
from models.transformers.ast.trainer import train_model as train_model_ast
from models.transformers.mae_ast.trainer import train_model as train_model_mae_ast
from models.transformers.ssast.trainer import train_model as train_model_ssast

### Importing the data generators from the models
from models.cnns.effnetv2.utils.data_generator import DataGenerator as DataGeneratorEffnet
from models.cnns.panns.utils.data_generator import DataGenerator as DataGeneratorPanns
from models.cnns.psla.utils.data_generator import DataGenerator as DataGeneratorPsla
from models.transformers.ast.utils.data_generator import DataGenerator as DataGeneratorAst
from models.transformers.mae_ast.utils.data_generator import DataGenerator as DataGeneratorMaeAst
from models.transformers.ssast.utils.data_generator import DataGenerator as DataGeneratorSsast


def model_store_training(
        model_name: str = None,
        train_dataset: str = None,
        val_dataset: str = None,
        cfg: dict = None,
        seed: int = None,
        cv: int = None
    ):
    train_model_method = None
    if model_name == 'effnetv2':
        train_dataset = DataGeneratorEffnet(
            batch=train_dataset,
            sample_rate=cfg.train.sample_rate,
            period=cfg.train.period,
            use_mixup=cfg.data.use_mixup,
            mixup_alpha=cfg.data.mixup_alpha,
            class_num=cfg.train.classes_num,
            f_min=cfg.data.f_min,
            f_max=cfg.data.f_max,
            n_mfcc=cfg.data.n_mfcc,
            data_type='train',
            rand_sampling=cfg.data.rand_sampling,
            window_size=cfg.data.window_size,
            hop_size=cfg.data.hop_size,
            mel_bins=cfg.data.mel_bins,
            feature_extractor=cfg.data.feature_extractor,
            apply_norm=cfg.data.apply_norm,
            target_dbfs=cfg.data.target_dbfs,
            use_filter_augment=cfg.data.use_filter_augment
        )

        val_dataset = DataGeneratorEffnet(
            batch=val_dataset,
            sample_rate=cfg.train.sample_rate,
            period=cfg.train.period,
            use_mixup=cfg.data.use_mixup,
            mixup_alpha=cfg.data.mixup_alpha,
            class_num=cfg.train.classes_num,
            f_min=cfg.data.f_min,
            f_max=cfg.data.f_max,
            n_mfcc=cfg.data.n_mfcc,
            data_type='val',
            rand_sampling=False,
            window_size=cfg.data.window_size,
            hop_size=cfg.data.hop_size,
            mel_bins=cfg.data.mel_bins,
            feature_extractor=cfg.data.feature_extractor,
            apply_norm=cfg.data.apply_norm,
            target_dbfs=cfg.data.target_dbfs,
            use_filter_augment=False
        )
        train_model_method = train_model_effnet
    elif model_name == "panns":
        DataGeneratorPanns(
            batch=train_dataset,
            sample_rate=cfg.feature_extractor.sample_rate,
            period=cfg.data.period,
            rand_sampling=cfg.data_augmentation.rand_sampling,
            apply_norm=cfg.data.apply_norm,
            target_dbfs=cfg.data.target_dbfs
        )

        val_dataset = DataGeneratorPanns(
            batch=val_dataset,
            sample_rate=cfg.feature_extractor.sample_rate,
            period=cfg.data.period,
            rand_sampling=cfg.data_augmentation.rand_sampling,
            apply_norm=cfg.data.apply_norm,
            target_dbfs=cfg.data.target_dbfs
        )

        train_model_method = train_model_panns

    elif model_name == "psla":
        train_dataset = DataGeneratorPsla(
            batch=train_dataset,
            sample_rate=cfg.feature_extractor.sample_rate,
            period=cfg.data.period,
            type='train',
            freqm=cfg.data_augmentation.freqm,
            timem=cfg.data_augmentation.timem,
            skip_norm=False,
            norm_mean=cfg.data.spec_norm_mean,
            norm_std=cfg.data.spec_norm_std,
            use_specaug=cfg.data_augmentation.use_specaug,
            target_length=cfg.feature_extractor.target_length,
            rand_sampling=cfg.data_augmentation.rand_sampling,
            use_audio_mixup=cfg.data_augmentation.use_audio_mixup,
            mixup_alpha=cfg.data_augmentation.mixup_alpha,
            insert_noise=cfg.data_augmentation.insert_noise,
            class_num=cfg.train.classes_num
        )

        val_dataset = DataGeneratorPsla(
            batch=val_dataset,
            sample_rate=cfg.feature_extractor.sample_rate,
            period=cfg.data.period,
            type='val',
            freqm=None,
            timem=None,
            skip_norm=False,
            norm_mean=cfg.data.spec_norm_mean,
            norm_std=cfg.data.spec_norm_std,
            use_specaug=False,
            target_length=cfg.feature_extractor.target_length,
            rand_sampling=False,
            use_audio_mixup=False,
            mixup_alpha=None,
            insert_noise=False,
            class_num=cfg.train.classes_num
        )

        train_model_method = train_model_psla

    elif model_name == "ast":
        train_dataset = DataGeneratorAst(
            batch=train_dataset,
            data_type="train",
            skip_spec_norm=False,
            **cfg.data
        )

        val_dataset = DataGeneratorAst(
            batch=val_dataset,
            data_type="val",
            skip_spec_norm=False,
            **cfg.data
        )

        train_model_method = train_model_ast

    elif model_name == "mae_ast":
        train_dataset = DataGeneratorMaeAst(
            batch=train_dataset,
            sample_rate=cfg.feature_extractor.sample_rate,
            period=cfg.data.period,
            type='train',
            freqm=cfg.data_augmentation.freqm,
            timem=cfg.data_augmentation.timem,
            use_specaug=cfg.data_augmentation.use_specaug,
            target_length=cfg.feature_extractor.target_length,
            rand_sampling=cfg.data_augmentation.rand_sampling,
            use_audio_mixup=cfg.data_augmentation.use_audio_mixup,
            mixup_alpha=cfg.data_augmentation.mixup_alpha,
            insert_noise=cfg.data_augmentation.insert_noise,
            class_num=cfg.model.classes_num
        )

        val_dataset = DataGeneratorMaeAst(
            batch=val_dataset,
            sample_rate=cfg.feature_extractor.sample_rate,
            period=cfg.data.period,
            type='val',
            freqm=None,
            timem=None,
            use_specaug=False,
            target_length=cfg.feature_extractor.target_length,
            rand_sampling=False,
            use_audio_mixup=False,
            mixup_alpha=None,
            insert_noise=False,
            class_num=cfg.model.classes_num
        )

        train_model_method = train_model_mae_ast

    elif model_name == "ssast":
        train_dataset = DataGeneratorSsast(
            batch=train_dataset,
            sample_rate=cfg.feature_extractor.sample_rate,
            period=cfg.data.period,
            type='train',
            freqm=cfg.data_augmentation.freqm,
            timem=cfg.data_augmentation.timem,
            skip_norm=False,
            norm_mean=cfg.data.spec_norm_mean,
            norm_std=cfg.data.spec_norm_std,
            use_specaug=cfg.data_augmentation.use_specaug,
            target_length=cfg.feature_extractor.target_length,
            rand_sampling=cfg.data_augmentation.rand_sampling,
            use_audio_mixup=cfg.data_augmentation.use_audio_mixup,
            mixup_alpha=cfg.data_augmentation.mixup_alpha,
            insert_noise=cfg.data_augmentation.insert_noise,
            class_num=cfg.train.classes_num
        )

        val_dataset = DataGeneratorSsast(
            batch=val_dataset,
            sample_rate=cfg.feature_extractor.sample_rate,
            period=cfg.data.period,
            type='val',
            freqm=None,
            timem=None,
            skip_norm=False,
            norm_mean=cfg.data.spec_norm_mean,
            norm_std=cfg.data.spec_norm_std,
            use_specaug=False,
            target_length=cfg.feature_extractor.target_length,
            rand_sampling=False,
            use_audio_mixup=False,
            mixup_alpha=None,
            insert_noise=False,
            class_num=cfg.train.classes_num
        )

        train_model_method = train_model_ssast

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.train.num_workers
    )

    valid_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.train.num_workers
    )

    train_model_method(
        train_loader,
        valid_loader,
        cfg=cfg,
        seed=seed,
        cv=cv
    )


def model_store_inference(
        model_name: str = None,
        test_dataset: str = None,
        cfg: dict = None,
        seed: int = None,
        cv: int = None
    ):

    if model_name == 'effnetv2':
        test_dataset = DataGeneratorEffnet(
            batch=test_dataset,
            sample_rate=cfg.train.sample_rate,
            period=cfg.train.period,
            use_mixup=cfg.data.use_mixup,
            mixup_alpha=cfg.data.mixup_alpha,
            class_num=cfg.train.classes_num,
            f_min=cfg.data.f_min,
            f_max=cfg.data.f_max,
            n_mfcc=cfg.data.n_mfcc,
            data_type='test',
            rand_sampling=False,
            window_size=cfg.data.window_size,
            hop_size=cfg.data.hop_size,
            mel_bins=cfg.data.mel_bins,
            feature_extractor=cfg.data.feature_extractor,
            apply_norm=cfg.data.apply_norm,
            target_dbfs=cfg.data.target_dbfs,
            use_filter_augment=False
        )

    if model_name == "panns":
        test_dataset = DataGeneratorPanns(
            batch=val_dataset,
            sample_rate=cfg.feature_extractor.sample_rate,
            period=cfg.data.period,
            rand_sampling=False,
            apply_norm=cfg.data.apply_norm,
            target_dbfs=cfg.data.target_dbfs
        )

    if model_name == "psla":
        test_dataset = DataGeneratorPsla(
            batch=val_dataset,
            sample_rate=cfg.feature_extractor.sample_rate,
            period=cfg.data.period,
            type='test',
            freqm=None,
            timem=None,
            skip_norm=False,
            norm_mean=cfg.data.spec_norm_mean,
            norm_std=cfg.data.spec_norm_std,
            use_specaug=False,
            target_length=cfg.feature_extractor.target_length,
            rand_sampling=False,
            use_audio_mixup=False,
            mixup_alpha=None,
            insert_noise=False,
            class_num=cfg.train.classes_num
        )

    if model_name == "ast":
        test_dataset = DataGeneratorAst(
            batch=val_dataset,
            data_type="test",
            skip_spec_norm=False,
            **cfg.data
        )

    if model_name == "mae_ast":
        test_dataset = DataGeneratorMaeAst(
            batch=val_dataset,
            sample_rate=cfg.feature_extractor.sample_rate,
            period=cfg.data.period,
            type='test',
            freqm=None,
            timem=None,
            use_specaug=False,
            target_length=cfg.feature_extractor.target_length,
            rand_sampling=False,
            use_audio_mixup=False,
            mixup_alpha=None,
            insert_noise=False,
            class_num=cfg.model.classes_num
        )

    if model_name == "ssast":
        test_dataset = DataGeneratorSsast(
            batch=val_dataset,
            sample_rate=cfg.feature_extractor.sample_rate,
            period=cfg.data.period,
            type='test',
            freqm=None,
            timem=None,
            skip_norm=False,
            norm_mean=cfg.data.spec_norm_mean,
            norm_std=cfg.data.spec_norm_std,
            use_specaug=False,
            target_length=cfg.feature_extractor.target_length,
            rand_sampling=False,
            use_audio_mixup=False,
            mixup_alpha=None,
            insert_noise=False,
            class_num=cfg.train.classes_num
        )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.train.num_workers
    )

    weight_path = glob.glob(os.path.join('weights', os.path.basename(cfg.data.output_dir), str(seed), cv, '*.pth'))[0]

    labels, pred_list = test_model(
        test_dataloader,
        checkpoint_path=weight_path,
        cfg=cfg
    )