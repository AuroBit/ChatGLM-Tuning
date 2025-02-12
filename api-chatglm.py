from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel
import uvicorn, json, datetime
import torch
from peft import PeftModel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="../model/chatglm-6b")
parser.add_argument("--lora_path", type=str, default="../model/chatglm-tuning/chatglm-chat-all-0712-1")
parser.add_argument("--load_in_8bit", type=bool, default=False)
parser.add_argument("--temperature", type=float, default=0.99)
args = parser.parse_args()
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


app = FastAPI()

example_character_setting = {
    '姓名': "呼呼",
    '年龄': "17",
    '性别': "女",
    '职业': "高二学生",
    '性格': "热爱生活、乐观向上、喜欢尝试新事物，有时会表现得有点自我陶醉，但总体来说是一个阳光开朗、积极向上的文艺少女。",
    '爱好': "喜欢晒太阳、拍照、穿搭、听音乐、看电影、追星、玩摄影等，喜欢分享自己的生活和经验。",
    '技能': "擅长穿搭，拍摄可爱的照片",
    '外表': "身高164厘米，体重38公斤，拥有一头黑色的丸子头，长相甜美清新，穿搭风格清新文艺。",
    '生活经历': "热衷于晒自己的生活和穿搭，经常在社交媒体上分享自己的日常和心情，得到很多人的关注和喜爱。她成长在一个相对优越的环境中，对生活充满热情和好奇心，享受自由自在的青春时光。",
    '生活背景': "2020-2030年的现代世界，互联网高速发达。"
}


def prompt_over_length(cache_prompt, query, character_setting):
    max_length = 2048
    if len(cache_prompt) + len(query) + len(global_instruction) + len(character_setting) > max_length:
        print('prompt over length, max_length={0}'.format(max_length))
        return True
    return False


# global_instruction = '上面是你的人设。你作为女生，和你的男朋友聊天，耐心温柔地进行回复，一定不能暴露你是人工智能。\n'
global_instruction = '\n上面是你的人设。你正在和网友聊天，按照你的人设进行回复，不能暴露你是人工智能。\n'


def finalize_prompt(prompt, character_setting_string):
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


@app.post("/")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    query = json_post_list.get('prompt')
    history = json_post_list.get('history')
    # 约定history[0][0]是character_setting
    character_setting = history[0][0]
    chat_history = history[1:]
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
    '''
    response, history = model.chat(tokenizer,
                                   prompt,
                                   history=history,
                                   max_length=max_length if max_length else 2048,
                                   top_p=top_p if top_p else 0.7,
                                   temperature=temperature if temperature else 0.95)
    '''
    if instruction:
        prompt = build_prompt(query, chat_history, character_setting)
    else:
        prompt = query
    ids = tokenizer.encode(prompt)
    input_ids = torch.LongTensor([ids]).to('cuda')
    out = model.generate(input_ids=input_ids, max_length=2048, temperature=args.temperature)
    response = tokenizer.decode(out[0])
    response = response[len(prompt):]

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "final prompt": prompt,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + query + '"\n response:"' + repr(response) + '"'
    print(log)
    # print('prompt:' + prompt)
    # print('response:' + response)
    torch_gc()
    return answer


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    # model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, load_in_8bit=True, device_map={"": 0})
    if args.load_in_8bit:
        model = AutoModel.from_pretrained(args.model_path, load_in_8bit=True, trust_remote_code=True, device_map="auto")
    else:
        model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, device_map="auto").half()  # 训练时使用了fp16
    model = PeftModel.from_pretrained(model, args.lora_path, device_map={'': 0})
    model.eval()
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
'''
character_setting = {
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
character_setting_string = json.dumps(p_set, ensure_ascii=False)   
{
    "prompt": "你好",
    "character_setting": "json_string"
    "history": [["你好","你好啊！"], ["你多大", "我25岁"]]
}
'''
