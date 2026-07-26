from pathlib import Path
import re
from typing import Dict, List, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5EncoderModel  # type: ignore

DEFAULT_MODEL_BASE_PATH = Path("models/flan-t5-base/").resolve()

"""
T5Config {
  "architectures": [
    "T5ForConditionalGeneration"
  ],
  "classifier_dropout": 0.0,
  "d_ff": 2048,
  "d_kv": 64,
  "d_model": 768,
  "decoder_start_token_id": 0,
  "dense_act_fn": "gelu_new",
  "dropout_rate": 0.1,
  "dtype": "float32",
  "eos_token_id": 1,
  "feed_forward_proj": "gated-gelu",
  "initializer_factor": 1.0,
  "is_decoder": false,
  "is_encoder_decoder": false,
  "is_gated_act": true,
  "layer_norm_epsilon": 1e-06,
  "model_type": "t5",
  "n_positions": 512,
  "num_decoder_layers": 12,
  "num_heads": 12,
  "num_layers": 12,
  "output_past": true,
  "pad_token_id": 0,
  "relative_attention_max_distance": 128,
  "relative_attention_num_buckets": 32,
  "scale_decoder_outputs": false,
  "task_specific_params": {
    "summarization": {
      "early_stopping": true,
      "length_penalty": 2.0,
      "max_length": 200,
      "min_length": 30,
      "no_repeat_ngram_size": 3,
      "num_beams": 4,
      "prefix": "summarize: "
    },
    "translation_en_to_de": {
      "early_stopping": true,
      "max_length": 300,
      "num_beams": 4,
      "prefix": "translate English to German: "
    },
    "translation_en_to_fr": {
      "early_stopping": true,
      "max_length": 300,
      "num_beams": 4,
      "prefix": "translate English to French: "
    },
    "translation_en_to_ro": {
      "early_stopping": true,
      "max_length": 300,
      "num_beams": 4,
      "prefix": "translate English to Romanian: "
    }
  },
  "tie_word_embeddings": true,
  "transformers_version": "5.12.1",
  "use_cache": false,
  "vocab_size": 32128
}
"""


class TextEncoder(nn.Module):
    def __init__(
        self,
        dtype: torch.dtype,
        dim: int = 768,
        max_length: int = 200,
    ):
        super(TextEncoder, self).__init__()
        self.model_path = DEFAULT_MODEL_BASE_PATH
        self.max_length = max_length
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
        self.model = T5EncoderModel.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
        )

        self._freeze_model()

    def _freeze_model(
        self,
    ):
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
    
    def train(self, mode: bool = True):
        """Safety: Prevent main training loop from accidentaly unfreezing Model's Dropout / LayerNorm layers when `model.train()` is called"""
        super().train(mode=mode)
        self.model.eval()
        return self

    def tokenize_instruction(
        self, instructions: Union[str, List[str]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not instructions:
            raise ValueError(f"Instruction list can not be empty")

        if isinstance(instructions, str):
            instructions = [instructions]

        instructions = [self._clean(instruction) for instruction in instructions]

        tokens = self.tokenizer(
            instructions,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_attention_mask=True,
            add_special_tokens=True,
        )

        return tokens["input_ids"], tokens["attention_mask"]

    def _clean(self, text):
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text

    @torch.no_grad()
    def forward(self, input_ids, attention_mask) -> Dict[str, torch.Tensor]:
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        prompt_emb = outputs.last_hidden_state
        seq_lens = attention_mask.gt(0).sum(dim=1).long()
        prompt_emb_clean = prompt_emb.clone()

        for i, valid_len in enumerate(seq_lens):
            prompt_emb_clean[:, valid_len:] = 0.0

        return {"text_embeddings": prompt_emb_clean, "attention_mask": attention_mask}
