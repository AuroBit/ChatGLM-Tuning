import argparse
import json
from tqdm import tqdm

import datasets
import transformers


def preprocess(tokenizer, config, example, max_seq_length):
    prompt = example["context"]
    target = example["target"]
    prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
    target_ids = tokenizer.encode(
        target,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False)
    input_ids = prompt_ids + target_ids + [config.eos_token_id]
    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}


def read_jsonl(model_path, path, max_seq_length, skip_overlength=False):
    model_name = model_path
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
    config = transformers.AutoConfig.from_pretrained(
        model_name, trust_remote_code=True, device_map='auto')
    with open(path, "r") as f:
        for line in tqdm(f.readlines()):
            example = json.loads(line)
            feature = preprocess(tokenizer, config, example, max_seq_length)
            if len(feature["input_ids"]) > max_seq_length:
                print('Existing too long line: {0}' .format(len(feature["input_ids"])))
            if skip_overlength and len(feature["input_ids"]) > max_seq_length:
                continue
            feature["input_ids"] = feature["input_ids"][:max_seq_length]
            yield feature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="../model/chatglm-6b")
    parser.add_argument("--jsonl_path", type=str, default="../data/chatglm-tuning/chatglm-chat-all.jsonl")
    parser.add_argument("--save_path", type=str, default="../data/chatglm-tuning/chatglm-chat-all")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--skip_overlength", type=bool, default=False)
    args = parser.parse_args()

    dataset = datasets.Dataset.from_generator(
        lambda: read_jsonl(args.model_path, args.jsonl_path, args.max_seq_length, args.skip_overlength)
    )
    dataset.save_to_disk(args.save_path)
    print(dataset)


if __name__ == "__main__":
    main()
