import os
from argparse import Namespace

import yaml


class DataConfig:
    def __init__(
        self,
        train_data_path: str,
        valid_data_path: str,
        batch_size: int,
        num_data_workers: int,
        prefetch_factor: int,
        time_delta_input_minutes: list[int],
        n_input_timestamps: int | None = None,
        pooling: int | None = None,
        random_vert_flip: bool = False,
        **kwargs,
    ):
        self.__dict__.update(kwargs)

        self.train_data_path = train_data_path
        self.valid_data_path = valid_data_path
        self.batch_size = batch_size
        self.num_data_workers = num_data_workers
        self.prefetch_factor = prefetch_factor
        self.time_delta_input_minutes = sorted(time_delta_input_minutes)
        self.n_input_timestamps = n_input_timestamps
        self.pooling = pooling
        self.random_vert_flip = random_vert_flip

        if self.n_input_timestamps is None:
            self.n_input_timestamps = len(self.time_delta_input_minutes)

        assert (
            self.n_input_timestamps > 0
        ), "Number of input timestamps must be greater than 0."
        assert self.n_input_timestamps <= len(self.time_delta_input_minutes), (
            f"Cannot sample {self.n_input_timestamps} from list of "
            f"{self.time_delta_input_minutes} input timestamps."
        )

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_argparse(args: Namespace):
        return DataConfig(**args.__dict__)

    def __str__(self):
        return (
            f"Training index: {self.train_data_path}, "
            f"Validation index: {self.valid_data_path}, "
        )

    def __repr__(self):
        return (
            f"Training index: {self.train_data_path}, "
            f"Validation index: {self.valid_data_path}, "
        )


class ModelConfig:
    def __init__(
        self,
        # enc_num_layers: int,
        # enc_num_heads: int,
        # enc_embed_size: int,
        # dec_num_layers: int,
        # dec_num_heads: int,
        # dec_embed_size: int,
        # mask_ratio: float,
        **kwargs,
    ):
        self.__dict__.update(kwargs)

        # self.enc_num_layers = enc_num_layers
        # self.enc_num_heads = enc_num_heads
        # self.enc_embed_size = enc_embed_size
        # self.dec_num_layers = dec_num_layers
        # self.dec_num_heads = dec_num_heads
        # self.dec_embed_size = dec_embed_size
        # self.mlp_ratio = 0.0
        # self.mask_ratio = mask_ratio

        self.__dict__.update(kwargs)

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_argparse(args: Namespace):
        return ModelConfig(**args.__dict__)

    @property
    def encoder_d_ff(self):
        return int(self.enc_embed_size * self.mlp_ratio)

    @property
    def decoder_d_ff(self):
        return int(self.dec_embed_size * self.mlp_ratio)

    def __str__(self):
        return (
            f"Input channels: {self.model.in_channels}, "
            f"Encoder (L, H, E): {[self.enc_num_layers, self.enc_num_heads, self.enc_embed_size]}, "
            f"Decoder (L, H, E): {[self.dec_num_layers, self.dec_num_heads, self.dec_embed_size]}"
        )

    def __repr__(self):
        return (
            f"Input channels: {self.model.in_channels}, "
            f"Encoder (L, H, E): {[self.enc_num_layers, self.enc_num_heads, self.enc_embed_size]}, "
            f"Decoder (L, H, E): {[self.dec_num_layers, self.dec_num_heads, self.dec_embed_size]}"
        )


class OptimizerConfig:
    def __init__(
        self,
        warm_up_steps: int,
        max_epochs: int,
        learning_rate: float,
        min_lr: float,
    ):
        self.warm_up_steps = warm_up_steps
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.min_lr = min_lr

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_argparse(args: Namespace):
        return ModelConfig(**args.__dict__)

    def __str__(self):
        return (
            f"Epochs: {self.max_epochs}, "
            f"LR: {[self.learning_rate, self.min_lr]}, "
            f"Warm up: {self.warm_up_steps},"
        )

    def __repr__(self):
        return (
            f"Epochs: {self.max_epochs}, "
            f"LR: {[self.learning_rate, self.min_lr]}, "
            f"Warm up: {self.warm_up_steps},"
        )


class ExperimentConfig:
    def __init__(
        self,
        job_id: str,
        data_config: DataConfig,
        model_config: ModelConfig,
        optimizer_config: OptimizerConfig,
        path_experiment: str,
        parallelism: str,
        from_checkpoint: str | None = None,
        **kwargs,
    ):
        # additional experiment parameters used in downstream tasks
        self.__dict__.update(kwargs)

        self.job_id = job_id
        self.data = data_config
        self.model = model_config
        self.optimizer = optimizer_config
        self.path_experiment = path_experiment
        self.from_checkpoint = from_checkpoint
        self.parallelism = parallelism

        assert self.model.in_channels == len(self.data.channels), (
            f"Number of model input channels ({self.model.in_channels}) must be "
            f"equal to number of input variables ({len(self.data.channels)})."
        )
        if self.model.time_embedding["type"] == "linear":
            assert (
                self.model.time_embedding["time_dim"] == self.data.n_input_timestamps
            ), "Time dimension of linear embedding must be equal to number of input timestamps."
        if self.rollout_steps > 0:
            assert self.data.n_input_timestamps == len(
                self.data.time_delta_input_minutes
            ), "Rollout does not support randomly sampled input timestamps."

        metrics_channels = []
        for field1, value1 in self.metrics["train_metrics_config"].items():
            for field2, value2 in self.metrics["train_metrics_config"][field1].items():
                if field2 == "metrics":
                    for metric_definition in value2:
                        split_metric_definition = metric_definition.split(":")
                        channels = (
                            split_metric_definition[2]
                            if len(split_metric_definition) > 2
                            else None
                        )
                        if channels is not None:
                            metrics_channels = metrics_channels + channels.split("...")

        for field1, value1 in self.metrics["validation_metrics_config"].items():
            for field2, value2 in self.metrics["validation_metrics_config"][
                field1
            ].items():
                if field2 == "metrics":
                    for metric_definition in value2:
                        split_metric_definition = metric_definition.split(":")
                        channels = (
                            split_metric_definition[2]
                            if len(split_metric_definition) > 2
                            else None
                        )
                        if channels is not None:
                            metrics_channels = metrics_channels + channels.replace(
                                "...", "&"
                            ).split("&")

        assert set(metrics_channels).issubset(self.data.channels), (
            f"{set(metrics_channels).difference(self.data.channels)} "
            f"not part of data input channels."
        )

        assert self.parallelism in [
            "ddp",
            "fsdp",
        ], 'Valid choices for `parallelism` are "ddp" and "fsdp".'

    @property
    def path_checkpoint(self) -> str:
        if self.path_experiment == "":
            return os.path.join(self.path_weights, "train", "checkpoint.pt")
        else:
            return os.path.join(
                os.path.dirname(self.path_experiment),
                "weights",
                "train",
                "checkpoint.pt",
            )

    @property
    def path_weights(self) -> str:
        return os.path.join(self.path_experiment, self.make_suffix_path(), "weights")

    @property
    def path_states(self) -> str:
        return os.path.join(self.path_experiment, self.make_suffix_path(), "states")

    def to_dict(self):
        d = self.__dict__.copy()
        d["model"] = self.model.to_dict()
        d["data"] = self.data.to_dict()

        return d

    @staticmethod
    def from_argparse(args: Namespace):
        return ExperimentConfig(
            data_config=DataConfig.from_argparse(args),
            model_config=ModelConfig.from_argparse(args),
            optimizer_config=OptimizerConfig.from_argparse(args),
            **args.__dict__,
        )

    @staticmethod
    def from_dict(params: dict):
        return ExperimentConfig(
            data_config=DataConfig(**params["data"]),
            model_config=ModelConfig(**params["model"]),
            optimizer_config=OptimizerConfig(**params["optimizer"]),
            **params,
        )

    def make_folder_name(self) -> str:
        param_folder = "wpt-c1-s1"
        return param_folder

    def make_suffix_path(self) -> str:
        return os.path.join(self.job_id)

    def __str__(self):
        return (
            f"ID: {self.job_id}, "
            f"Epochs: {self.optimizer.max_epochs}, "
            f"Batch size: {self.data.batch_size}, "
            f"LR: {[self.optimizer.learning_rate, self.optimizer.min_lr]}, "
            f"Warm up: {self.optimizer.warm_up_steps},"
            f"DL workers: {self.data.num_data_workers},"
            f"Parallelism: {self.parallelism}"
        )

    def __repr__(self):
        return (
            f"ID: {self.job_id}, "
            f"Epochs: {self.optimizer.max_epochs}, "
            f"Batch size: {self.data.batch_size}, "
            f"LR: {[self.optimizer.learning_rate, self.optimizer.min_lr]}, "
            f"Warm up: {self.optimizer.warm_up_steps},"
            f"DL workers: {self.data.num_data_workers},"
            f"Parallelism: {self.parallelism}"
        )


def get_config(
    config_path: str,
) -> ExperimentConfig:
    cfg = yaml.safe_load(open(config_path, "r"))
    cfg["data"]["scalers"] = yaml.safe_load(open(cfg["data"]["scalers_path"], "r"))
    return ExperimentConfig.from_dict(params=cfg)
