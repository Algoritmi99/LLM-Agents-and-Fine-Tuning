import os
from overrides import override

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .llm_loader_base import *


class Classification_LLM_Loader(Model_Loader_Base):
    def __init__(self,
                 model_hf_path: str,
                 id2label: dict,
                 label2id: dict,
                 bnb_config: BitsAndBytesConfig = default_bnb_config,
                 local_path=None,
                 max_memory: str = "24000MB",
                 access_token=None
                 ):
        super().__init__(model_hf_path, bnb_config, local_path, max_memory, access_token)
        self.id2label = id2label
        self.label2id = label2id

    @override
    def initialize(self):
        if self.local_path is not None and os.path.exists(self.local_path):
            print("Loading Model from local files:", "'" + self.local_path + "'")
            self.tokenizer = AutoTokenizer.from_pretrained(self.local_path + "/tokenizer")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.local_path + "/model",
                device_map="auto",
                max_memory={i: self.max_memory for i in range(torch.cuda.device_count())},
                token=self.access_token,
                num_labels=len(self.id2label),
                id2label = self.id2label, label2id = self.label2id
            )
        else:
            print("Downloading Model...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_hf_path)
            self.tokenizer.save_pretrained("./Models/" + self.model_hf_path + "/tokenizer")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_hf_path,
                quantization_config=self.bnb_config,
                device_map="auto",
                max_memory={i: self.max_memory for i in range(torch.cuda.device_count())},
                token=self.access_token,
                num_labels=len(self.id2label),
                id2label=self.id2label, label2id=self.label2id
            )
            self.model.save_pretrained("./Models/" + self.model_hf_path + "/model", access_token=self.access_token)
            print("Model and tokenizer saved to: ", "./Models/" + self.model_hf_path)
