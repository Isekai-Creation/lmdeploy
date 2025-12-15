# Copyright (c) OpenMMLab. All rights reserved.

from .base import OUTPUT_MODELS, BaseOutputModel


@OUTPUT_MODELS.register_module(name='tm')
class TurbomindModel(BaseOutputModel):
    """Export to turbomind fp16 format."""
    def save(self, out_dir: str):
        """Export weights and config to out_dir."""
        if not out_dir:
            return
        # update output directory and enable file export
        self.out_dir = out_dir
        self.to_file = True
        # export config
        self.export_config()
        # export per-layer weights
        last_reader = None
        for i, reader in self.input_model.readers():
            self.model(i, reader)
            last_reader = reader
        # export embeddings / norms / lm_head using last reader
        if last_reader is not None:
            self.model.misc(-1, last_reader)
