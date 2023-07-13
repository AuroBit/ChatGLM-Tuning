from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments
from transformers import Trainer, HfArgumentParser
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass, field
import datasets
import os
import shutil


@dataclass
class FinetuneArguments:
    base_model_path: str = field(default="")
    dataset_path: str = field(default="")
    model_path: str = field(default="")
    lora_rank: int = field(default=8)


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def data_collator(features: list) -> dict:
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids)
    input_ids = []
    labels_list = []
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]
        labels = (
                [-100] * (seq_len - 1) + ids[(seq_len - 1):] + [-100] * (longest - ids_l)
        )
        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }


class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss

    def save_model(self, output_dir=None, _internal_call=False):
        # from transformers.trainer import TRAINING_ARGS_NAME

        os.makedirs(output_dir, exist_ok=True)
        # torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        # saved_params = {
        #    k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        # }
        # torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))
        self.model.save_pretrained(output_dir)


finetune_args, training_args = HfArgumentParser((FinetuneArguments, TrainingArguments)).parse_args_into_dataclasses()
# 核对模型
print('***************************************')
print('model:' + finetune_args.base_model_path)
print('dataset:' + finetune_args.dataset_path)
print('output:' + training_args.output_dir)
print('***************************************')

tokenizer = AutoTokenizer.from_pretrained(finetune_args.base_model_path, trust_remote_code=True)


def main():
    writer = SummaryWriter()

    # save finetune args
    save_finetune_args(training_args.output_dir)

    # init model
    model = AutoModel.from_pretrained(
        finetune_args.base_model_path, load_in_8bit=True, trust_remote_code=True, device_map="auto"
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True

    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )

    model.lm_head = CastOutputToFloat(model.lm_head)

    # setup peft
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=finetune_args.lora_rank,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)

    # load dataset
    dataset = datasets.load_from_disk(finetune_args.dataset_path)
    print(f"\n{len(dataset)=}\n")

    # start train
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        callbacks=[TensorBoardCallback(writer)],
        data_collator=data_collator,
    )
    trainer.train()
    writer.close()
    # save model
    model.save_pretrained(training_args.output_dir)


def save_finetune_args(path):
    file = 'finetune-chatglm.sh'
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Folder '{path}' created successfully.")
    else:
        print(f"Folder '{path}' already exists.")
    shutil.copy(file, path)


if __name__ == "__main__":
    main()

