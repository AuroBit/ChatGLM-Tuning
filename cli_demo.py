import os
import platform
import signal
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from transformers import AutoTokenizer
import torch

model = AutoModel.from_pretrained("/home/zh/ChatGLM-6B/ptuning/THUDM/chatglm-6b", trust_remote_code=True,
                                  load_in_8bit=True, device_map={"": 0})
model = PeftModel.from_pretrained(model, "./output/chatglm_instruction/checkpoint-1200", device_map={'': 0})
model = model.eval()

tokenizer = AutoTokenizer.from_pretrained("/home/zh/ChatGLM-6B/ptuning/THUDM/chatglm-6b", trust_remote_code=True)

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


def add_instruction(prompt):
    return '模拟情侣聊天，你扮演女生来耐心地与男生完成一轮对话，一定不能暴露你是AI。\n男生：' + prompt + '\n女生：'


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


instruction = True


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
        out = model.generate(
            input_ids=input_ids,
            max_length=512
        )

        response = tokenizer.decode(out[0])
        # response = response.replace(query, '')
        response = response.split(':')[-1]
        # print('query:' + query, flush=True)
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
