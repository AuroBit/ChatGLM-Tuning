from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments
from transformers import Trainer, HfArgumentParser
from transformers import AutoTokenizer, AutoModel, AutoConfig, BitsAndBytesConfig
from accelerate import init_empty_weights
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from dataclasses import dataclass, field
import datasets
import os
from pprint import pprint as print
import shutil

model_path = "../model/chatglm2-6b"
model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


# tokenizer.add_special_tokens({
#             "eos_token": "</s>",
#             "bos_token": "<sop>",
#             "unk_token": "<unk>",
#         })

@dataclass
class FinetuneArguments:
    tokenized_dataset: str = field(default=" ")  # tokenized之后的数据集文件夹
    model_path: str = field(default=" ")
    lora_rank: int = field(default=8)
    q_lora: int = field(default=0)


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
        seq_len = feature["seq_len"]  # prompt length
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


# 这里的 collator 主要参考了 https://github.com/mymusise/ChatGLM-Tuning/blob/master/finetune.py 中的写法
# 将 prompt 的部分的label也设置为了 -100，从而在训练时不纳入loss的计算
# 对比之下，我在 baichaun_lora_tuning.py 中，是直接使用 DataCollatorForLanguageModeling，prompt 也纳入了计算。
# 这两种方式孰优孰劣尚不得而知，欢迎在issue或discussion中讨论。

class ModifiedTrainer(Trainer):
    # def compute_loss(self, model, inputs, return_outputs=False):
    #     return model(
    #         input_ids=inputs["input_ids"],
    #         labels=inputs["labels"],
    #     ).loss

    def save_model(self, output_dir=None, _internal_call=False):
        # 因为交给Trainer的model实际上是PeftModel类型，所以这里的 save_pretrained 会直接使用PeftModel的保存方法
        # 从而只保存 LoRA weights
        self.model.save_pretrained(output_dir)
        # from transformers.trainer import TRAINING_ARGS_NAME
        # os.makedirs(output_dir, exist_ok=True)
        # torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        # saved_params = {
        #     k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        # }
        # torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))


# For more information:
# https://github.com/beyondguo/LLM-Tuning
def main():
    writer = SummaryWriter()
    finetune_args, training_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)
    ).parse_args_into_dataclasses()
    # save finetune args
    save_finetune_args(training_args.output_dir)

    # load dataset
    dataset = datasets.load_from_disk(finetune_args.tokenized_dataset)
    # dataset = dataset.select(range(10000))
    print(f"\n{len(dataset)=}\n")

    if finetune_args.q_lora == 1:
        q_config = BitsAndBytesConfig(load_in_4bit=True,
                                      bnb_4bit_quant_type='nf4',
                                      bnb_4bit_use_double_quant=True,
                                      bnb_4bit_compute_dtype=torch.bfloat16)

        model = AutoModel.from_pretrained(model_path,
                                          quantization_config=q_config,
                                          device_map='auto',
                                          trust_remote_code=True)

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        # init model
        with init_empty_weights():  # 似乎没用
            print('loading init model...')
            model = AutoModel.from_pretrained(
                model_path, trust_remote_code=True, load_in_8bit=True,
                device_map={'': 0},
                # device_map="auto"  # 模型不同层会被自动分配到不同GPU上进行计算
                # device_map={'':torch.cuda.current_device()}
            )
        # print(model.hf_device_map)
        print(f'memory_allocated {torch.cuda.memory_allocated()}')

        """
        设置了 device_map="auto" 之后
        chatglm 1.0 的时候，lm_head会跟input_layer自动分配到同个 device，
        chatglm 2.0 的时候，没有了 lm_head，有一个 output_layer，这个时候可能会分配到两个 device，导致计算loss的时候报错显示
        RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:2 and cuda:0!
        一个解决办法是设置 device_map={'':torch.cuda.current_device()}，进行数据并行，但是这样batchsize只能设置非常小，而且很占显存
        另一个解决办法是: 手动把 output_layer 设置为跟 input 一样的 device
        然后这里会加载两次模型，可以先加载，调整device_map之后，再把旧模型删掉：https://github.com/pytorch/pytorch/issues/37250#issuecomment-1622972872
        """

        if torch.cuda.device_count() > 1:
            model.hf_device_map['transformer.output_layer'] = model.hf_device_map['transformer.embedding']
            new_hf_device_map = model.hf_device_map
            model.cpu()
            del model
            torch.cuda.empty_cache()
            print(f'memory_allocated {torch.cuda.memory_allocated()}')
            print('loading real model...')
            model = AutoModel.from_pretrained("../model/chatglm2-6b", trust_remote_code=True,
                                              device_map=new_hf_device_map)
            print(model.hf_device_map)

        """
        .gradient_checkpointing_enable()
        .enable_input_require_grads()
        .is_parallelizable
        这三个都是 transformers 模型的函数/参数（见 transformers/modeling_utils.py 文件）
        """
        model.gradient_checkpointing_enable()
        # note: use gradient checkpointing to save memory at the expense of slower backward pass.
        model.enable_input_require_grads()
        # note: Enables the gradients for the input embeddings. This is useful for fine-tuning adapter weights while keeping the model weights fixed.
        # See https://github.com/huggingface/transformers/blob/ee88ae59940fd4b2c8fc119373143d7a1175c651/src/transformers/modeling_utils.py#L1190

    # setup peft
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=finetune_args.lora_rank,
        target_modules=['query_key_value'],
        lora_alpha=32,
        lora_dropout=0.1,
    )

    model = get_peft_model(model, peft_config)
    print('==========print_trainable_parameters===========')
    model.print_trainable_parameters()

    # start train
    model.save_pretrained(
        training_args.output_dir)  # 因为adapter_config.json只能通过这个save_pretrained来生成，先这里生成一份，好在训练完之前就可以尝试中间的checkpoint
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        callbacks=[TensorBoardCallback(writer)],
        data_collator=data_collator
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

# ChatGLM2 finetune
# https://github.com/shuxueslpi/chatGLM-6B-QLoRA
# https://github.com/beyondguo/LLM-Tuning