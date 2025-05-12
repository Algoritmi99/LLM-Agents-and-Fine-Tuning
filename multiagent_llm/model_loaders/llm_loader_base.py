from abc import abstractmethod, ABC

from transformers import BitsAndBytesConfig
import torch

default_bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


class Model_Loader_Base(ABC):
    def __init__(self,
                 model_hf_path: str,
                 bnb_config: BitsAndBytesConfig = default_bnb_config,
                 local_path=None,
                 max_memory: str = "24000MB",
                 access_token=None):
        self.model_hf_path = model_hf_path
        self.bnb_config = bnb_config
        self.local_path = local_path
        self.max_memory = max_memory
        self.access_token = access_token
        self.model = None
        self.tokenizer = None
        self.initialize()

    @abstractmethod
    def initialize(self):
        """
        reimplement this method in subclasses to load the model and tokenizer for a given task
        :return: void
        """
        pass

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def get_max_length(self) -> int:
        for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
            max_length = getattr(self.model.config, length_setting, None)
            if max_length:
                break
        if not max_length:
            max_length = 1024
        return max_length

