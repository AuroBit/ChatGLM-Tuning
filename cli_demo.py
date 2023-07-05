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
parser.add_argument("--lora_path", type=str, default="../model/chatglm-tuning/sens_all")
args = parser.parse_args()

model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, load_in_8bit=True, device_map={"": 0})
model = PeftModel.from_pretrained(model, args.lora_path, device_map={'': 0})
model = model.eval()

tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False


def build_prompt(history):
    # prompt = "欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    prompt = ""
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM-6B：{response}"
    return prompt


# def add_instruction(prompt):
#    return '模拟情侣聊天，你扮演女生来耐心地与男生完成一轮对话，一定不能暴露你是AI。\n男生：' + prompt + '\n女生：'
# def add_instruction(prompt):
#    return '你是男生的女朋友，耐心温柔地回复男生的问题，一定不能暴露你是AI。\n男生：' + prompt + '\n女生：'
# def add_instruction(prompt):
#    return '你是男生的女朋友，耐心温柔地回复男生的问题，一定不能暴露你是人工智能。\n男生：' + prompt + '\n女生：'
# def add_instruction(prompt):
#    return '你扮演下面这个男生的女朋友，耐心温柔地回复下面这个男生的问题，一定不能暴露你是人工智能。\n男生：' + prompt + '\n女生：'

add_person_set = True
person_set = {
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
def add_instruction(prompt):
    if add_person_set:
        person_set_string = json.dumps(person_set, ensure_ascii=False)
        return '下面是你的人设：' + person_set_string + \
            '你扮演下面这个男生的女朋友，耐心温柔地回复下面这个男生的问题，一定不能暴露你是人工智能。\n男生：' + prompt + '\n女生：'
    else:
        return '你扮演下面这个男生的女朋友，耐心温柔地回复下面这个男生的问题，一定不能暴露你是人工智能。\n男生：' + prompt + '\n女生：'


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


instruction = True
stream = False


def main():
    history = []
    global stop_stream
    # print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            os.system(clear_command)
            # print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        count = 0
        if instruction:
            query = add_instruction(query)
        ids = tokenizer.encode(query)
        input_ids = torch.LongTensor([ids]).to('cuda')

        if stream:
            print("response：", end="")
            for outputs in model.stream_generate(input_ids=input_ids, max_length=2048):
                response = tokenizer.decode(outputs[0])
                response = response[len(query):]
                if not response:
                    continue
                print(response[-1], end="")
        else:
            out = model.generate(input_ids=input_ids, max_length=2048)
            response = tokenizer.decode(out[0])
            response = response[len(query):]
            print('response:' + response, flush=True)

        if stop_stream:
            stop_stream = False
            break
        else:
            count += 1
            if count % 8 == 0:
                os.system(clear_command)


if __name__ == "__main__":
    main()
