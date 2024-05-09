from datasets import load_dataset
from transformers import Trainer
from transformers import TrainingArguments

from utils.FinetuneUtils import *

os.environ["WANDB_DISABLED"] = "true"


sr = 16_000
VOCAB_FILE = "./vocab.json"
with open(VOCAB_FILE) as vocab_file:
    vocab_size = len(json.load(vocab_file))
ds_dict = load_dataset("/Users/shankhajyoti.de/PythonProjects/automatic_speech_recognition/dataset/dataset_files/alphabet_classification",
                               data_files={"train": "train.csv", "validation": "validation.csv"})

# create vocab file and dataset_dict
ds_dict = ds_dict.map(modify_text, batched=False)
TOKENIZER = Wav2Vec2CTCTokenizer(VOCAB_FILE, unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|")
FEATURE_EXTRACTOR = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
PROCESSOR = Wav2Vec2Processor(feature_extractor=FEATURE_EXTRACTOR, tokenizer=TOKENIZER)


### initialize the processor_utils
ProcessorUtilities(PROCESSOR)

ds_dict = ds_dict.map(prepare_dataset, batched=False, remove_columns=["audio", "text"])
data_collator = DataCollatorCTCWithPadding(processor=PROCESSOR, padding=True)




MODEL = Wav2Vec2ForCTC.from_pretrained(
    "/Users/shankhajyoti.de/PythonProjects/automatic_speech_recognition/archived_finetuned_models/checkpoint-1460",
    ctc_loss_reduction="sum", ## can be sum/ mean
    pad_token_id=PROCESSOR.tokenizer.pad_token_id,
    vocab_size=vocab_size
)
# config_base = MODEL_wav2vec2_base_960h.config
# config_base.vocab_size = vocab_size
# MODEL = Wav2Vec2ForCTC(config_base)
# MODEL.wav2vec2 = MODEL_wav2vec2_base_960h.wav2vec2
MODEL.freeze_feature_extractor()
"""
Wav2Vec2Model has the feature projectors and feature encoders responsible for creating the encoded representations of the audio
Wav2Vec2ForCTC.lm_head is responsible for projecting the transformer encoder's last layer's hidden states to vocab_size 
"""
# MODEL = Wav2Vec2ForCTC.from_pretrained(
#     "facebook/wav2vec2-base-960h",
#     ctc_loss_reduction="sum", ## can be sum/ mean
#     pad_token_id=PROCESSOR.tokenizer.pad_token_id,
# )
# model_config: Wav2Vec2Config = MODEL.config
# model_config.vocab_size = vocab_size
# MODEL.freeze_feature_extractor()


len_train = len(ds_dict["train"])
batch_size = 10
eval_batch_size = 10
num_train_epochs = 10
total_no_of_steps = (len_train//batch_size + 1) * num_train_epochs  if len_train % batch_size != 0 else (len_train//batch_size) * num_train_epochs
no_of_steps_in_epoch = len_train//batch_size
save_steps = min(no_of_steps_in_epoch, total_no_of_steps)
eval_steps = min(no_of_steps_in_epoch, total_no_of_steps)
logging_steps = min(5, total_no_of_steps)
# warmup_steps = math.floor(0.1 * total_no_of_steps) ###10 % of total no of steps
warmup_steps = 2

TRAINING_ARGS = TrainingArguments(
  output_dir="/Users/shankhajyoti.de/PythonProjects/automatic_speech_recognition/models/finetuned_models",
  overwrite_output_dir=True,
  group_by_length=True,
  per_device_train_batch_size=batch_size,
  per_device_eval_batch_size=eval_batch_size,
  evaluation_strategy="steps",
  num_train_epochs=num_train_epochs,
  fp16=False,
  gradient_checkpointing=True,
  save_steps=save_steps,
  eval_steps=eval_steps,
  logging_steps=logging_steps,
  learning_rate=1e-4,
  weight_decay=0.005,
  warmup_steps=warmup_steps,
  save_total_limit=2,
  push_to_hub=False,
  report_to=None
)

TRAINER = Trainer(
    model=MODEL,
    data_collator=data_collator,
    args=TRAINING_ARGS,
    compute_metrics=compute_metrics,
    train_dataset=ds_dict["train"],
    eval_dataset=ds_dict["validation"],
    tokenizer=PROCESSOR.feature_extractor,
)

TRAINER.train()

## save in dir
TOKENIZER.save_pretrained(TRAINING_ARGS.output_dir)






