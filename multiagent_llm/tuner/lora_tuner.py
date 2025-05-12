import os

import bitsandbytes as bnb
import torch
from datasets import Dataset
from overrides import override
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import (
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling, PreTrainedModel
)

from multiagent_llm.tuner.base_tuner import Tuner

default_training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    max_steps=20,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=1,
    output_dir="outputs",
    optim="paged_adamw_8bit",
)


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )


def create_peft_config(modules, lora_rank, lora_alpha, lora_dropout, task_type):
    """
    Create Parameter-Efficient Fine-Tuning config for your model
    :param modules: Names of the modules to apply Lora to
    :param lora_rank: Rank of the LoRA
    :param lora_alpha: Alpha of the LoRA
    :param lora_dropout: Dropout of the LoRA
    :param task_type: Type of the task (CASUAL_LM, SEQ_2_SEQ_LM, SEQ_CLS, TOKEN_CLS, QUESTION_ANS, FEATURE_EXTRACTION)
    """
    return LoraConfig(
        r=lora_rank,  # dimension of the updated matrices
        lora_alpha=lora_alpha,  # parameter for scaling
        target_modules=modules,
        lora_dropout=lora_dropout,  # dropout probability for layers
        bias="none",
        task_type=task_type,
    )


class LoRATrainer(Tuner):
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizerBase,
                 training_args: TrainingArguments = default_training_args,
                 lora_rank=16,
                 lora_alpha=64,
                 lora_dropout=0.1,
                 task_type="CAUSAL_LM",
                 access_token=None
                 ):
        self.model = model
        self.tokenizer = tokenizer
        self.training_args = training_args
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.task_type = task_type
        self.access_token = access_token

    @override
    def train(self, dataset: Dataset, output_dir: str):
        #split data
        dataset = dataset.train_test_split(test_size=0.1)
        # Prepare the model
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)
        modules = find_all_linear_names(self.model)
        peft_config = create_peft_config(
            modules, self.lora_rank, self.lora_alpha, self.lora_alpha, self.task_type
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.config.use_cache = False

        # build the trainer
        print("Parameter configuration of the model")
        print_trainable_parameters(self.model)

        trainer = Trainer(
            model=self.model,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            args=self.training_args,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        )

        # verifying the datatypes before training
        dtypes = {}
        for _, p in self.model.named_parameters():
            dtype = p.dtype
            if dtype not in dtypes:
                dtypes[dtype] = 0
            dtypes[dtype] += p.numel()
        total = 0
        for k, v in dtypes.items():
            total += v
        for k, v in dtypes.items():
            print(k, v, v / total)
        do_train = True

        # starting training
        print("Training...")
        if do_train:
            train_result = trainer.train()
            metrics = train_result.metrics
            trainer.log_metrics(split="train", metrics=metrics)
            trainer.save_metrics(split="train", metrics=metrics)
            trainer.save_state()
            print(metrics)

        # prepare the model for usage
        self.model.config.use_cache = True

        # Saving model
        print("Saving last checkpoint of the model...")
        os.makedirs(output_dir, exist_ok=True)
        trainer.model.save_pretrained(output_dir, access_token=self.access_token)

        out = self.model.merge_and_unload()
        # Free memory for merging weights
        del self.model
        del trainer
        torch.cuda.empty_cache()

        return out
