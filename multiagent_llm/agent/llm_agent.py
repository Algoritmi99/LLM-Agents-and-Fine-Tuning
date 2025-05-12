import torch
import transformers
from overrides import override

from .agent import Agent
from ..model_loaders import *


class LLM_Agent(Agent):
    """
    Serves a chatbot or a sequence classifier using a given llm loader.
    """
    def __init__(self, model_loader: Model_Loader_Base, max_len=256, keep_memory=False):
        super().__init__()
        self.__model = model_loader.get_model()
        self.__tokenizer = model_loader.get_tokenizer()
        self.__mode = "classification" if isinstance(model_loader, Classification_LLM_Loader) else (
            "casual" if isinstance(model_loader, Casual_LLM_Loader) else "other"
        )

        if self.__mode == "casual":
            self.__pipeline = transformers.pipeline(
                "text-generation",
                model=self.__model,
                tokenizer=self.__tokenizer,
            )
            self.__keep_memory = keep_memory
            if self.__keep_memory:
                self.__messages = []
        self.__max_length = max_len

    @override
    def act(self, state: str | list[dict] | dict) -> str:
        """
        :param state: prompt for the llm.
        :return: response of the llm.
        """
        if self.__mode == "classification":
            inputs = self.__tokenizer(state, return_tensors="pt")
            logits = self.__model(**inputs).logits
            predicted_class_id = int(torch.argmax(logits))
            return self.__model.config.id2label[predicted_class_id]

        if self.__mode == "casual":
            if isinstance(state, str):
                if self.__keep_memory:
                    self.__messages.append({"role": "user", "content": state})
                    in_next = self.__messages
                else:
                    in_next = [{"role": "user", "content": state}]
            elif isinstance(state, dict):
                if self.__keep_memory:
                    self.__messages.append(state)
                    in_next = self.__messages
                else:
                    in_next = [state]
            else:
                if self.__keep_memory:
                    self.__messages = state
                    in_next = self.__messages
                else:
                    in_next = state

            out = self.__pipeline(
                in_next,
                max_new_tokens=self.__max_length
            )[0]["generated_text"][-1]['content']

            if self.__keep_memory:
                self.__messages.append({"role": "assistant", "content": out})

            return out

        else:
            real_prompt = self.__tokenizer.encode(state, return_tensors="pt")
            output = self.__model.generate(
                real_prompt, max_new_tokens=self.__max_length, num_beams=4, no_repeat_ngram_size=2
            )
            response = self.__tokenizer.decode(output[0], skip_special_tokens=True)
            return str(response)
