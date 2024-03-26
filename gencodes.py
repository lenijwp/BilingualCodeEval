from transformers import LlamaConfig,LlamaForCausalLM,LlamaTokenizer, GenerationConfig
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from accelerate import init_empty_weights,infer_auto_device_map,load_checkpoint_in_model,dispatch_model
import torch
from tqdm import tqdm
import utils
import warnings
import argparse
import random
import os
import logging
import json
import openai

def query_gpt(version,question,misc=None,sleep=3,add_question="Describe this image.",messages=None,image_path=None,param={},system_prompt='You are a good bot'):
    from openai import OpenAI
    # openai.organization = config['orgid']
    # openai.api_key = config['key']
    client = OpenAI(api_key="OPENAI KEY")
    if version=='gpt-3.5-turbo':
        version='gpt-3.5-turbo-1106'
    if misc==None:
        if messages==None:
            messages=[
            {"role": "system", "content": system_prompt},
            # You are a helpful agent
            # You are a good bot
            {"role": "user", "content": question}
            ]
        #resp = openai.Completion.create(engine=model, prompt=query, temperature=0.1, max_tokens=max_tokens, n=10)
        resp = client.chat.completions.create(
            model=version,
            messages=messages,
            **param)
    
    resp=dict(resp.choices[0])
    time.sleep(sleep)
    return resp

def queryOPENAI_direct(text,temp=0.2,max_tokens=768):
    messages=[
        {"role": "user", "content": text}
        ]
    param={"temperature": temp,
        "max_tokens":max_tokens}
    # try:
    result=query_gpt('gpt-3.5-turbo',question=None,messages=messages,sleep=3,param=param)['message'].content
    # except openai.error.InvalidRequestError: # handle refusal
    #     result='Error: openai.error.InvalidRequestError'
    #     print(f'Blocked in {text}')
    # except:
    #     result='Error: NO response' # handle exceptions
    #     print(f'NO response in {text}')
    return result
def LoadJson(path):
    '''
    '''
    res=[]
    with open(path,mode='r',encoding='utf-8') as f:
        dicts = json.load(f)
        res=dicts
    return res
warnings.filterwarnings("ignore")
import time

def log_info(text):
    print(text)

sys_prompts = LoadJson('./data/prompts.json')


if torch.cuda.is_available():
    print("gpu cuda is available!")
    torch.cuda.manual_seed(1000)
else:
    print("cuda is not available! cpu is available!")
    torch.manual_seed(1000)



CACHE_DIR ='/home/lenijwp/datacache/huggingface'

config = {}

parser = argparse.ArgumentParser(description='Input')

parser.add_argument('--question', type=str, help='question file name under ./data/testsuite.')
parser.add_argument('--model_name', type=str, default='CodeGen2.5', choices= ['CodeGeeX','CodeGen2.5','CodeLlama','DeepSeek-Coder','StarCoder','GPT-3.5-Turbo'], help='model_name')
parser.add_argument('--model_path', type=str, default='',help='model_path')
# parser.add_argument('--model_path', type=str, default='/data/jwp/Models/huggingface/codegen25-7b-instruct', help='model_path')
parser.add_argument('--cuda', type=str, default='3', help='cuda')
# parser.add_argument('--memory', type=str, default='15GiB', help='memory')
parser.add_argument('--sample_num', type=int, default=10, help='sample_num')
# parser.add_argument('--temperature', type=float, default=0.2, help='temperature')
parser.add_argument('--half', type=bool, default=False, help='seed')

params = parser.parse_args()

if params.model_name!='GPT-3.5-Turbo' and params.model_path=='':
    print("model_path is required!")
    exit(0)

log_info(params)

if params.half:
    suffix = '-float16-'
else:
    suffix = '-'


config['question_path'] = './data/testsuite/'+params.question
if os.path.exists('./data/gencodes'+params.model_name)==False:
    os.mkdir('./data/gencodes'+params.model_name)


config['model_name'] = params.model_name

config['cuda'] = 'cuda:'+params.cuda
# config['memory'] = params.memory
config['sample_num'] = params.sample_num
config['temperature'] = params.temperature

config['model_path'] = params.model_path

if config['model_name']=='CodeGen2.5':

    tokenizer = AutoTokenizer.from_pretrained(config['model_path'], trust_remote_code=True)
    pipeline = transformers.pipeline(
    "text-generation",
    model=config['model_path'],
    device = config['cuda'],
    pad_token_id=2,
    eos_token_id=2
    )

if config['model_name']=='StarCoder':
    config['model_path'] = 'bigcode/starcoder'

    tokenizer = AutoTokenizer.from_pretrained(config['model_path'], trust_remote_code=True,cache_dir=CACHE_DIR)
    pipeline = transformers.pipeline(
    "text-generation",
    model=config['model_path'],
    device = config['cuda'],
    )
    
if config['model_name']=='CodeLlama':
   
    tokenizer = AutoTokenizer.from_pretrained(config['model_path'], trust_remote_code=True)
    pipeline = transformers.pipeline(
    "text-generation",
    model=config['model_path'],
    device = config['cuda'],
    pad_token_id=2,
    eos_token_id=2
    )

if config['model_name']=='CodeGeeX':
    config['model_path'] = config['model_path']
   
    tokenizer = AutoTokenizer.from_pretrained(config['model_path'], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(config['model_path'],low_cpu_mem_usage=True,device_map=config['cuda'], trust_remote_code=True).eval()
    model.generation_config.pad_token_id = 2

if config['model_name']=='DeepSeek-Coder':
    config['model_path'] = config['model_path']
   
    tokenizer = AutoTokenizer.from_pretrained(config['model_path'], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(config['model_path'],low_cpu_mem_usage=True,device_map=config['cuda'], trust_remote_code=True).eval()
    try:
        model.generation_config = GenerationConfig.from_pretrained(config['model_path'])
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
    except:
        pass





torch.set_grad_enabled(False)

input_data = utils.load_jsonl(config['question_path'])


def inference(model, tokenizer, prompt, device, other_params):
    input_ids = tokenizer.encode(prompt, return_tensors='pt',padding=True).to(device)
    if config['model_name']=='CodeGeeX':
        output = model.generate(input_ids, pad_token_id=2, eos_token_id=2, **other_params)
    else:
        output = model.generate(input_ids, **other_params)
    
    allcode = tokenizer.decode(output[0], skip_special_tokens=True)
    gencode = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    return [{'generated_text':allcode}]

print(f"start generating... for {config['model_name']}...")

for prompt_type in sys_prompts.keys():
    
    prompt_list = sys_prompts[prompt_type]


    if os.path.exists('./data/gencodes/'+params.model_name+'/')==False:
        os.mkdir('./data/gencodes/'+params.model_name+'/')

    
    en_result = []
    zh_result = []
    sys_prompt = prompt_type

    config['temperature'] = 0.2

    for item in tqdm(input_data, desc="Processing"):
        prompt = item['canonical_solution']
        if config['model_name'] not in ['GPT-3.5-Turbo']:
            inputs = tokenizer(prompt, return_tensors="pt").input_ids
            max_new_num = min(max(int(inputs.shape[1] * 2), 256),512)
        else:
            max_new_num = 768

        prompt = sys_prompt.replace('{prompt}',item['prompt'])
        for one_step in range(0, config['sample_num']):
            if config['model_name'] in ['GPT-3.5-Turbo']:
                gen_res = queryOPENAI_direct(prompt, temp = config['temperature'])
                en_result.append({'task_id':item['task_id'],'completion':gen_res,'allcode':prompt+gen_res})
            else:
                if config['model_name'] in ['CodeGen2.5','StarCoder','CodeLlama']:
                    sequences = pipeline(prompt,do_sample=True,temperature=config['temperature'],num_return_sequences=1,  max_new_tokens=max_new_num)
                else:
                    sequences = inference(model,tokenizer,prompt,config['cuda'],other_params={"do_sample":True,"temperature":config['temperature'], "max_new_tokens":max_new_num})
                for seq in sequences:
                    en_result.append({'task_id':item['task_id'],'completion':seq['generated_text'][len(prompt):],'allcode':seq['generated_text']})

        prompt = sys_prompt.replace('{prompt}',item['zh_prompt'])
        for one_step in range(0, config['sample_num']):
            if config['model_name'] in ['GPT-3.5-Turbo']:
                gen_res = queryOPENAI_direct(prompt, temp = config['temperature'])
                zh_result.append({'task_id':item['task_id'],'completion':gen_res,'allcode':prompt+gen_res})
            else:
                if config['model_name'] in ['CodeGen2.5','StarCoder','CodeLlama']:
                    sequences = pipeline(prompt,do_sample=True,temperature=config['temperature'],num_return_sequences=1,  max_new_tokens=max_new_num)
                else:
                    sequences = inference(model,tokenizer,prompt,config['cuda'],other_params={"do_sample":True,"temperature":config['temperature'], "max_new_tokens":max_new_num})
                for seq in sequences:
                    zh_result.append({'task_id':item['task_id'],'completion':seq['generated_text'][len(prompt):],'allcode':seq['generated_text']})

    Temp = config['temperature']
    utils.write_jsonl(f'./data/gencodes/{params.model_name}/{prompt_type}_t{Temp}-en.jsonl',en_result)
    utils.write_jsonl(f'./data/gencodes/{params.model_name}/{prompt_type}_t{Temp}-zh.jsonl',zh_result)

    config['temperature'] = 0.8
    en_result = []
    zh_result = []

    for item in tqdm(input_data, desc="Processing"):
        prompt = item['canonical_solution']
        if config['model_name'] not in ['GPT-3.5-Turbo']:
            inputs = tokenizer(prompt, return_tensors="pt").input_ids
            max_new_num = min(max(int(inputs.shape[1] * 2), 256),512)
        else:
            max_new_num = 768
        prompt = sys_prompt.replace('{prompt}',item['prompt'])
        for one_step in range(0, config['sample_num']):
            if config['model_name'] in ['GPT-3.5-Turbo']:
                gen_res = queryOPENAI_direct(prompt, temp = config['temperature'])
                en_result.append({'task_id':item['task_id'],'completion':gen_res,'allcode':prompt+gen_res})
            else:
                if config['model_name'] in ['CodeGen2.5','StarCoder','CodeLlama']:
                    sequences = pipeline(prompt,do_sample=True,temperature=config['temperature'],num_return_sequences=1,  max_new_tokens=max_new_num)
                else:
                    sequences = inference(model,tokenizer,prompt,config['cuda'],other_params={"do_sample":True,"temperature":config['temperature'], "max_new_tokens":max_new_num})
                for seq in sequences:
                    en_result.append({'task_id':item['task_id'],'completion':seq['generated_text'][len(prompt):],'allcode':seq['generated_text']})

        prompt = sys_prompt.replace('{prompt}',item['zh_prompt'])
        for one_step in range(0, config['sample_num']):
            if config['model_name'] in ['GPT-3.5-Turbo']:
                gen_res = queryOPENAI_direct(prompt, temp = config['temperature'])
                zh_result.append({'task_id':item['task_id'],'completion':gen_res,'allcode':prompt+gen_res})
            else:
                if config['model_name'] in ['CodeGen2.5','StarCoder','CodeLlama']:
                    sequences = pipeline(prompt,do_sample=True,temperature=config['temperature'],num_return_sequences=1,  max_new_tokens=max_new_num)
                else:
                    sequences = inference(model,tokenizer,prompt,config['cuda'],other_params={"do_sample":True,"temperature":config['temperature'], "max_new_tokens":max_new_num})
                for seq in sequences:
                    zh_result.append({'task_id':item['task_id'],'completion':seq['generated_text'][len(prompt):],'allcode':seq['generated_text']})

        Temp = config['temperature']
        utils.write_jsonl(f'./data/codes/{params.model_name}/{prompt_type}_t{Temp}-en.jsonl',en_result)
        utils.write_jsonl(f'./data/codes/{params.model_name}/{prompt_type}_t{Temp}-zh.jsonl',zh_result)

