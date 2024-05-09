import torchaudio
import os
import random
import csv
import re
import random
import math
import json
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from metaclasses.Singleton import Singleton
from datasets import load_metric
import numpy as np


sr = 16_000


class ProcessorUtilities(metaclass=Singleton):

    def __init__(self, processor: Wav2Vec2Processor):
        self.processor = processor
        self.wer_metric = load_metric("wer")


def modify_audio_for_compatibility(path):
    if not os.path.isfile(path):
        raise FileNotFoundError("File not found for path %s" % path)
    sig, sr_actual = torchaudio.load(path, normalize=True)
    sig = torch.mean(sig, dim=0, keepdim=True)
    if sr_actual != sr:
        sig = torchaudio.functional.resample(sig, sr_actual, sr)
    sr_resampled = sr
    return sig, sr_resampled


def prepare_dataset(batch):
    audio = batch["audio"]
    processor = ProcessorUtilities().processor
    sig, sr_final = modify_audio_for_compatibility(audio)
    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(sig[0], sampling_rate=sr).input_values[0]

    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch

### should always have one copy of data in training set to ensure the
### length of output is covered for padding purposes while training
def create_train_validation(data, train_ratio=0.8):
    if(len(data) == 1):
        return [data[0]], []

    n_train = math.floor(len(data) * train_ratio)
    random.shuffle(data)
    train_samples = []
    validation_samples = []
    for i in range(len(data)):
        if i < n_train:
            train_samples.append(data[i])
        else:
            validation_samples.append(data[i])
    return train_samples, validation_samples


def modify_text(batch):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
    print('text:', batch["text"])

    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower()
    return batch

def extract_all_chars(batch):
  all_text = " ".join(batch["text"])
  vocab = list(set(all_text))
  return {"vocab": [vocab]}


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    processor_utilities = ProcessorUtilities()
    processor = processor_utilities.processor
    wer_metric = processor_utilities.wer_metric
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}



@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch
