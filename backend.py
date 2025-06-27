import re
import time
import queue

from flask import Flask, request, Response, render_template
from flask_sockets import Sockets
import mlx.core as mx
from mlx_lm.utils import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler
import argparse

import os
import torch
import json
import requests
import threading

app = Flask(__name__)

from flask_cors import *
CORS(app, supports_credentials=True)


messages = []
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
system_prompt = "You are a professional software engineer who's name is FishBoneEK. Your daily conversations are funny and humorous but serious in professional conversations. You will judge others' command to determine whether to follow it. You will use tools provided wisely instead calculate by yourself."
if supports_system_role:
    messages.append({"role": "system", "content": system_prompt})

def generate(tokenizer, prompt, model, temp=0.6, top_p=0.95, context_length=16384, stop_words=[]):
    text = ""

    for (token, prob), n in zip(generate_step(mx.array(tokenizer.encode(prompt)), model, max_tokens=-1, sampler=make_sampler(temp, top_p)),
                                range(context_length)):

        if token == tokenizer.eos_token_id:
            break

        delta = tokenizer.decode(token)
        text += delta
        yield delta

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
    global response_buffer, responding
    while responding: # Problemo
        if len(response_buffer) > ptr:
            yield response_buffer[ptr]
            ptr += 1
        else:
            time.sleep(0.1)

def send_message(messages, temp, top_p, previous=""):
    global debug, chat_template, tokenizer, model
    prompt = tokenizer.apply_chat_template(messages, tools=fetch_tools(), tokenize=False, add_generation_prompt=True, chat_template=chat_template)
    prompt = prompt.rstrip("\n")
    if debug:
        print(prompt)
        print("-" * 80)
    flag = True
    while flag:
        flag = False
        response = previous

        for chunk in generate(tokenizer, prompt, model, temp, top_p):
            response = response + chunk

            if not previous:
                # begin neural-beagle-14 fixes
                response = re.sub(r"^/\*+/", "", response)
                response = re.sub(r"^:+", "", response)
                # end neural-beagle-14 fixes

            response = response.replace('�', '')
            # yield response + "▌"
            yield chunk.replace('�', '')

        # yield response
        messages.append({"role": "assistant", "content": response})
        answer = response
        if "</think>" in answer:
            answer = answer.split("</think>")[1]
        if "<tool_call>" in answer and "</tool_call>" in answer:
            tool_call = answer.split('<tool_call>')[1].split('</tool_call>')[0]
            callback = function_call(json.loads(tool_call))
            messages.append({"role": "tool", "content": callback})
            yield "\n\n" + callback + "\n\n"
            prompt = tokenizer.apply_chat_template(messages,
                                                        tools=fetch_tools(),
                                                        tokenize=False,
                                                        add_generation_prompt=True,
                                                        chat_template=chat_template)
            prompt = prompt.rstrip("\n")
            flag = True
    

def function_call(json):
    return requests.post("http://127.0.0.1:12701/function_call", json=json).text

def require_thinking(msg):
    global tokenizer, model, chat_template
    system_prompt = "You are a classifier agent designed to determine whether user request's complexity deserves deep thinking. You must only return True if it's difficult or False if it's simple based on the user's request. /nothink"
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": msg}]
    prompt = tokenizer.apply_chat_template(messages, tools=[], tokenize=False, add_generation_prompt=True, chat_template=chat_template)
    response = ""
    for chunk in generate(tokenizer, prompt, model, 0.2, 0.1):
        response = response + chunk

        # begin neural-beagle-14 fixes
        response = re.sub(r"^/\*+/", "", response)
        response = re.sub(r"^:+", "", response)
        # end neural-beagle-14 fixes
        
        response = response.replace('�', '')
    answer = response
    if "</think>" in answer:
        answer = answer.split("</think>")[1]
    print(answer.strip().strip('\n'))
    return eval(answer.strip())
    

t1 = threading.Thread(target=app.run, kwargs={"host": "0.0.0.0", "port": 8501})
t1.start()
print("Server started on port 8501")
while True:
    if message_queue.empty():
        responding = False
    msg = message_queue.get()
    if not require_thinking(msg["content"]):
        messages[0] = {"role": "system", "content": system_prompt + " /nothink"}
    else:
        messages[0] = {"role": "system", "content": system_prompt}
    messages.append({"role": msg["role"], "content": msg["content"]})
    for chunk in send_message(messages, msg["temp"], msg["top_p"]):
        response_buffer += chunk
    response_buffer = ""