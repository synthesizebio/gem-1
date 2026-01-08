import lightning
import torch
import torch.nn as nn
from cytoolz import identity, keyfilter
from text_to_rna.constants import STUDY_BLACKLIST


class SynthesizeBioModel(lightning.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters(config)
        self.metrics = nn.ModuleDict()
        self.classification_metrics = nn.ModuleDict()

        if self.hparams.get("no_pert", False):
            print("filtering out perturbations")
            self.filter_fn = lambda x: str(x["perturbation_type"][0]) in {"", "control"}
        else:
            print("filtering out studies: ", STUDY_BLACKLIST)
            self.filter_fn = lambda x: str(x["study"][0]) not in STUDY_BLACKLIST

    def collate(self, split: str):
        return identity

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        raise NotImplementedError

    def on_load_checkpoint(self, checkpoint):
        checkpoint["state_dict"] = keyfilter(
            lambda key: "metrics" not in key, checkpoint["state_dict"]
        )
        return checkpoint
