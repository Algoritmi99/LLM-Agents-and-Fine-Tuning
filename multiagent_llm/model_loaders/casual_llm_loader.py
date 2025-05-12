import os

from transformers import AutoTokenizer, AutoModelForCausalLM
from overrides import override

from .llm_loader_base import *


class Casual_LLM_Loader(Model_Loader_Base):
    def __init__(self,
                 model_hf_path: str,
                 bnb_config: BitsAndBytesConfig = default_bnb_config,
                 local_path=None,
                 max_memory: str = "24000MB",
                 access_token=None
                 ):
        super().__init__(model_hf_path, bnb_config, local_path, max_memory, access_token)

    @override
    def initialize(self):
        if self.local_path is not None and os.path.exists(self.local_path):
            print("Loading Model from local files:", "'" + self.local_path + "'")
            self.tokenizer = AutoTokenizer.from_pretrained(self.local_path + "/tokenizer")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.local_path + "/model",
                device_map="auto",
                max_memory={i: self.max_memory for i in range(torch.cuda.device_count())},
                token=self.access_token
            )
        else:
            print("Downloading Model...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_hf_path)
            self.tokenizer.save_pretrained("./Models/" + self.model_hf_path + "/tokenizer")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_hf_path,
                quantization_config=self.bnb_config,
                device_map="auto",
                max_memory={i: self.max_memory for i in range(torch.cuda.device_count())},
                token=self.access_token
            )
            self.model.save_pretrained("./Models/" + self.model_hf_path + "/model", access_token=self.access_token)
            print("Model and tokenizer saved to: ", "./Models/" + self.model_hf_path)
