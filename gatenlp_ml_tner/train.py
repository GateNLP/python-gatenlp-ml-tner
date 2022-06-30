"""
Module for finetuning transformer models with data exported from GateNLP.
"""
import os
import json
from typing import Optional
import tner

class TokenClassificationTrainer:
    def __init__(
            self,
            data_dir,
            output_dir = "./tner_tokenclassification_model",
            transformer_model="bert-base",
            train_size=0.9,
            eval_size=0.1,
            shuffle=True,
            seed=42,
            learning_rate=2e-5,
            max_epochs=20,
            batch_size=16,
    ):
        assert os.path.isdir(data_dir)
        assert os.path.isdir(output_dir)
        self.data_dir = data_dir
        self.output_dir = output_dir


    def train(self):
        pass