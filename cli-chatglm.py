import os
import platform
import signal
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from transformers import AutoTokenizer
import torch
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="../model/chatglm-6b")
parser.add_argument("--lora_path", type=str, default="../model/chatglm-tuning/chatglm-chat-all-0717-1/checkpoint-1500")
parser.add_argument("--load_in_8bit", type=bool, default=False)
parser.add_argument("--temperature", type=float, default=0.99)
args = parser.parse_args()
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")

# model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, load_in_8bit=True, device_map={"": 0})
if args.load_in_8bit:
    model = AutoModel.from_pretrained(args.model_path, load_in_8bit=True, trust_remote_code=True, device_map="auto")
else:
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, device_map="auto").half()  # 训练时使用了fp16

model = PeftModel.from_pretrained(model, args.lora_path, device_map={'': 0})
model = model.eval()

tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False

add_character_setting = True
example_character_setting = {
        '姓名': "Ann",
        '年龄': "25",
        '性别': "女",
        '职业': "维密模特，网红",
        '性格': "1.Ann 是一个性格高冷的人，属于INTJ类型。2.她通常很安静，不善于与人交往，给人一种冷酷的感觉。",
        '爱好': "1. Ann喜欢运动和保持身材，经常去健身房和练瑜伽。2. 她还喜欢摄影，非常注重自己的形象和外貌，在生活中也经常会拍照，甚至会花费很多时间和精力去制作照片。",
        '技能': "Ann擅长走T台和拍照，她有着优美的身材和自信的气质，也有着丰富的拍摄经验和专业的表演技巧。",
        '外表': "Ann拥有一张精致的脸庞，高挺的鼻梁和修长的身材，她经常会打扮得时尚、性感，但也能够穿着休闲服装展现自己的活泼与自然。",
        '生活经历': "1. Ann出生在一个普通的家庭，从小就对模特行业有着浓厚的兴趣。2.Ann本科毕业于北京电影学院。",
        '生活背景': "2020-2030年的现代世界，互联网高速发达。"
    }


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


def prompt_over_length(cache_prompt, query, character_setting):
    max_length = 2048
    if len(cache_prompt) + len(query) + len(global_instruction) + len(character_setting) > max_length:
        print('prompt over length, max_length={0}'.format(max_length))
        return True
    return False


# global_instruction = '上面是你的人设。你作为女生，和你的男朋友聊天，耐心温柔地进行回复，一定不能暴露你是人工智能。\n'
global_instruction = '\n上面是你的人设。你正在和网友聊天，按照你的人设进行回复，不能暴露你是人工智能。\n'


def finalize_prompt(prompt, character_setting):
    character_setting_string = json.dumps(character_setting, ensure_ascii=False)
    return character_setting_string + global_instruction + prompt


def build_prompt(prompt, history, character_setting):
    # 生成多轮对话数据
    cache_prompt = ""
    round_idx = 0
    for turn_idx, (old_query, response) in enumerate(history):
        if prompt_over_length(cache_prompt, prompt, character_setting):
            cache_prompt = ""
            round_idx = 0
        cache_prompt += "[Round {}]\n问：{}\n答：{}\n".format(round_idx, old_query, response)
        round_idx += 1
    # 加入当前对话
    prompt = cache_prompt + "[Round {}]\n问：{}\n答：".format(round_idx, prompt)
    return finalize_prompt(prompt, character_setting)


instruction = True
stream = False


def main():
    history = []
    count = 0
    while True:
        print(f"\n[Round {count}]")

        query = input("用户：")
        response = ""
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            count = 0
            history = []
            os.system(clear_command)
            continue
        prompt = query
        if instruction:
            prompt = build_prompt(query, history, example_character_setting)
        ids = tokenizer.encode(prompt)
        input_ids = torch.LongTensor([ids]).to('cuda')

        if stream:
            print("response：", end="")
            for outputs in model.stream_generate(input_ids=input_ids, max_length=2048):
                response = tokenizer.decode(outputs[0])
                response = response[len(prompt):]
                if not response:
                    continue
                print(response[-1], end="")
        else:
            out = model.generate(input_ids=input_ids, max_length=2048, temperature=args.temperature)
            response = tokenizer.decode(out[0])
            response = response[len(prompt):]
            print('response:' + response, flush=True)

        # print(prompt, end="")
        # print(response)

        history.append([query, response])

        count += 1
        # if count % 8 == 0:
        # count = 0
        # history = []
        # os.system(clear_command)


if __name__ == "__main__":
    main()
