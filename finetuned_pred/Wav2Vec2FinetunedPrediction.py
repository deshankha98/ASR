import csv
import os
import re

import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from beam_search.word_state_files.date_word_vs_state import DATE_STATES, DATE_STATE_TRANSITION, DATE_STATES_VS_WORDS
from models.ModelFactory import ModelFactory

# import json
# import torch
from beam_search.GuidedBeamSearch import *

sr = 16_000

class ASRModel(torch.nn.Module):
    def __init__(self, model: Wav2Vec2ForCTC, processor: Wav2Vec2Processor):
        super().__init__()
        self.asr_model = model
        self.processor = processor

    def forward(self, x):
        return self.asr_model(x).logits
    def get_label(self, index):
        return self.processor.tokenizer.convert_ids_to_tokens(index)

def modify_audio_for_compatibility(path):
    if not os.path.isfile(path):
        raise FileNotFoundError("File not found for path %s" % path)
    sig, sr_actual = torchaudio.load(path, normalize=True)
    sig = torch.mean(sig, dim=0, keepdim=True)
    if sr_actual != sr:
        sig = torchaudio.functional.resample(sig, sr_actual, sr)
    sr_resampled = sr
    return sig, sr_resampled

# /Users/shankhajyoti.de/PythonProjects/MultiSpeakerTTS/tts_outputs/english/augmented_keywords/fourteen_june/fourteen|june__p277Vits|speedupAugmentationSnr5db_1.wav
if __name__ == "__main__":
    model, processor = ModelFactory("/Users/shankhajyoti.de/PythonProjects/automatic_speech_recognition/archived_finetuned_models").get_model()
    path = "/Users/shankhajyoti.de/PythonProjects/MultiSpeakerTTS/tts_outputs/english/augmented_keywords/thirty_one_january/thirty|one|january__p256Vits|speedupAugmentationSnr5db_1.wav"
    path = "/Users/shankhajyoti.de/PythonProjects/MultiSpeakerTTS/tts_outputs/english/augmented_keywords/eighth/eighth__p256Vits|speedupAugmentationSnr5db_1.wav"
    actual_file_name = path.split("/")[-1]
    actual_file_name = re.sub(".wav", "", actual_file_name)
    asr_model = ASRModel(model, processor)

    sig, _ = modify_audio_for_compatibility(path)
    with torch.inference_mode():
        emissions = asr_model(sig)

    emissions_csv_file_name = '/Users/shankhajyoti.de/PythonProjects/automatic_speech_recognition/emissions/' + actual_file_name + ".csv"
    prefixes_csv_file_name = '/Users/shankhajyoti.de/PythonProjects/automatic_speech_recognition/detected_prefixes/' + actual_file_name + ".csv"
    with open(emissions_csv_file_name, 'w') as f:
        # create the csv writer
        writer = csv.writer(f)
        for i in range(len(emissions[0])):
            row_str = []

            for tensor, index in sorted([(emissions[0][i][k], k) for k in range(len(emissions[0][i]))], key=lambda x: x[0].item(), reverse=True):
                row_str.append((asr_model.get_label(index), tensor.item()))

            writer.writerow(row_str)

    emissions_reshaped = torch.reshape(emissions, [emissions.shape[1], emissions.shape[2]]).numpy()
    beam_search = GuidedBeamSearch(emissions_reshaped, None, DATE_STATES, DATE_STATE_TRANSITION, DATE_STATES_VS_WORDS)
    detected_prefixes, detected_sentence = beam_search.guided_beam_search()
    with open(prefixes_csv_file_name, 'w') as f:
        # create the csv writer
        writer = csv.writer(f)
        for i in range(len(detected_prefixes)):
            row_str = []

            for sentence_formed, prefix, prob in detected_prefixes[i]:
                row_str.append((sentence_formed, prefix, prob))

            writer.writerow(row_str)