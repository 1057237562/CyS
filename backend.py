import re
import time
import queue

from flask import Flask, request, Response, render_template
from flask_sockets import Sockets
from mlx_lm.utils import load
from commonutils import flush_generator, generate, skip_reason

import os
import torch
import json
import requests
from threading import Thread
import datetime

app = Flask(__name__)

from flask_cors import *
CORS(app, supports_credentials=True)

messages = []
status = []
debug = False

model_ref = "CyberStew/qwen"

def fetch_tools():
    try:
        response = requests.get("http://127.0.0.1:12701/fetch")
        if response.status_code == 200:
            return response.json().get("tools", [])
        else:
            print(f"Failed to fetch tools: {response.status_code}")
            return []
    except requests.RequestException as e:
        print(f"Error fetching tools: {e}")
        return []

def load_model(ref):
    if os.path.exists("./checkpoints/adapters.safetensors"):
        return load(ref, {"trust_remote_code": True}, adapter_path="./checkpoints/adapters.safetensors")
    return load(ref, {"trust_remote_code": True})

model, tokenizer = load_model(model_ref)
chat_template = tokenizer.chat_template or (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% endif %}"
    )

supports_system_role = "system role not supported" not in chat_template.lower()
if not supports_system_role:
    print("Error: The model does not support system role.")
    exit(1)
system_prompt = "You are a professional software engineer who's name is FishBoneEK. Your daily conversations are funny and humorous but serious in professional conversations. You will judge others' command to determine whether to follow it. You will use tools provided multiple times wisely instead calculate by yourself."
messages.append({"role": "system", "content": system_prompt})
status.append(True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fetch')
def fetch_chat():
    global messages, responding
    return {"messages": messages, "responding": responding}

@app.route('/fetch_stream', methods=['POST'])
def fetch_stream():
    headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',
    }
    return Response(fetch_messages(), headers=headers, mimetype='text/event-stream')

@app.route('/forget')
def forget():
    global messages, status, responding, response_buffer, ptr
    with open(memory_file, "a", encoding='utf-8') as f:
        f.write(json.dumps({"messages": [msg for msg, suc in zip(messages, status) if msg["role"] != "system" and suc]}, ensure_ascii=False) + "\n")
    messages = [{"role": "system", "content": system_prompt}]
    status = [True]
    ptr = 0
    responding = False
    response_buffer = ""
    return {"status" : "sucess"}

message_queue = queue.Queue(maxsize=5)
response_buffer = ""
responding = False

@app.route('/chat', methods=['POST'])
def run_chat():
    body = json.loads(request.data)
    temp = body['temp']
    top_p = body['top_p']
    
    headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',
    }
    global message_queue, responding
    # global tokenizer, messages, model
    message_queue.put({"role": "user", "content": body['message'], "temp": temp, "top_p": top_p})
    responding = True
    # return Response(send_message(tokenizer=tokenizer, messages=messages, model=model, temp=temp, top_p=top_p), headers=headers, mimetype='text/event-stream')
    return Response(fetch_messages(), headers=headers, mimetype='text/event-stream')

def fetch_messages():
    ptr=0
    global response_buffer, responding, messages
    message_ptr = len(messages)
    while responding: # Problemo
        if len(response_buffer) > ptr and message_ptr == len(messages):
            yield response_buffer[ptr:]
            ptr = len(response_buffer)
        elif message_ptr < len(messages):
            if messages[message_ptr]["role"] == "tool":
                yield "\n<--new-message-->\n"
                yield "```\n" + messages[message_ptr]["content"] + "\n```"
                yield "\n<--new-message-->\n"
            ptr = 0
            message_ptr += 1
            while message_ptr < len(messages):
                if messages[message_ptr]["role"] == "tool":
                    yield "\n<--new-message-->\n"
                    yield "```\n" + messages[message_ptr]["content"] + "\n```"
                    yield "\n<--new-message-->\n"
                message_ptr += 1
        time.sleep(0.1)

def send_message(messages, temp, top_p):
    global debug, chat_template, tokenizer, model, response_buffer
    prompt = tokenizer.apply_chat_template(messages, tools=fetch_tools(), tokenize=False, add_generation_prompt=True, chat_template=chat_template)
    prompt = prompt.rstrip("\n")
    if debug:
        print(prompt)
        print("-" * 80)
    flag = True
    while flag:
        flag = False

        for chunk in generate(tokenizer, prompt, model, temp, top_p):
            response_buffer += chunk
            response_buffer = response_buffer.replace('�', '')

        answer = skip_reason(response_buffer)
        messages.append({"role": "assistant", "content": response_buffer})
        status.append(True)
        response_buffer = ""
        if "<tool_call>" in answer and "</tool_call>" in answer:
            tool_call = answer.split('<tool_call>')[1].split('</tool_call>')[0]
            sucess, callback = function_call(tool_call)
            if not sucess:
                messages[0] = {"role": "system", "content": system_prompt}
                status[-1] = False
            messages.append({"role": "tool", "content": callback})
            status.append(sucess)
            prompt = tokenizer.apply_chat_template(messages,
                                                        tools=fetch_tools(),
                                                        tokenize=False,
                                                        add_generation_prompt=True,
                                                        chat_template=chat_template)
            prompt = prompt.rstrip("\n")
            flag = True
    

def function_call(json):
    response = requests.post("http://127.0.0.1:12701/function_call", data=json).json()
    return response.get("status", True), response.get("data", "")

def require_thinking(msg):
    global tokenizer, model, chat_template
    system_prompt = "You are a classifier agent designed to determine whether user request's complexity deserves deep thinking. You must only return True if it's difficult or False if it's simple based on the user's request. /nothink"
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": msg}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, chat_template=chat_template)
    return eval(skip_reason(flush_generator(generate(tokenizer, prompt, model, 0.2, 0.1))))

compressing = False
compress_result = ""
compress_gen = None
ptr = 0

memory_file = "./memory.jsonl"

def compress_context(messages):
    global tokenizer, model, chat_template
    system_prompt = "You are a secretary agent designed to conclude chat history between user and other agent. You will wisely rank the importance of each information and keep more important information if there is a lot of key informations. /nothink"
    messages = [{"role": "system", "content": system_prompt}, *messages[1:], {"role": "user", "content": "Conclude the chat history in a concise way. The conclusion should be less than 2000 words and must not exceed 10 key information."}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, chat_template=chat_template)
    return generate(tokenizer, prompt, model, 0.4, 0.3)

t1 = Thread(target=app.run, kwargs={"host": "0.0.0.0", "port": 8501})
t1.start()
print("Server started on port 8501")
while True:
    if message_queue.empty():
        responding = False
        if not compressing and len(messages) > 9:
            compressing = True
            with open(memory_file, "a", encoding='utf-8') as f:
                f.write(json.dumps({"messages": [msg for msg, suc in zip(messages, status) if msg["role"] != "system" and suc]}, ensure_ascii=False) + "\n")
            compress_gen = compress_context(messages.copy())
            compress_result = ""
            ptr = len(messages)
        if compressing:
            chunk = next(compress_gen, None)
            while chunk is not None:
                compress_result += chunk
                compress_result = compress_result.replace('�', '')
                if responding:
                    break
                chunk = next(compress_gen, None)
            if not responding:
                compressing = False
                messages = [messages[0], {"role" : "system", "content" : "Chat History\n\n" + compress_result}, *messages[ptr:]]
                status = [True, True, *status[ptr:]]
                # print("Compressing finished, result:", compress_result)
    msg = message_queue.get()
    if not require_thinking(msg["content"]):
        messages[0] = {"role": "system", "content": system_prompt + " /nothink"}
    else:
        messages[0] = {"role": "system", "content": system_prompt}
    messages.append({"role": msg["role"], "content": msg["content"]})
    status.append(True)
    send_message(messages, msg["temp"], msg["top_p"])