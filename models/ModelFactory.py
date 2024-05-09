from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, PreTrainedModel
from metaclasses.Singleton import Singleton

PROCESSOR = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
MODEL = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")


class ModelFactory(metaclass=Singleton):

    def __init__(self, dir_path):

        if dir_path is None or dir_path == '':
            self.processor: Wav2Vec2Processor = PROCESSOR
            self.model: PreTrainedModel = MODEL
        else:
            suffix = "/checkpoint-2920"
            self.feature_extractor: Wav2Vec2FeatureExtractor = Wav2Vec2FeatureExtractor.from_pretrained(dir_path + suffix)
            self.tokenizer: Wav2Vec2CTCTokenizer = Wav2Vec2CTCTokenizer.from_pretrained(dir_path)
            self.processor: Wav2Vec2Processor = Wav2Vec2Processor(feature_extractor=self.feature_extractor, tokenizer=self.tokenizer)
            self.model: PreTrainedModel = Wav2Vec2ForCTC.from_pretrained(dir_path + suffix)


    def get_model(self):
        return self.model, self.processor

