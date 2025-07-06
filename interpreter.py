import subprocess
import os
from flask import Flask, request
import json

app = Flask(__name__)

from flask_cors import *
CORS(app, supports_credentials=True)

tools = [{
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Execute a snippet of Python code and return the stdout. Can be used in doing test, validation, and do math computation tasks instead of reasoning. You don't need to check the code correctness and efficiency before executing it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to be executed. The string will only be escaped once. Use \n instead of \\n to write a new line."
                    },
                    "input": {
                        "type": "string",
                        "description": "The input to be redirected to stdin."
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "pip_list",
            "description": "List all library that is installed in current python environment. You can check the installed libraries if u are not sure.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

@app.route('/function_call', methods=['POST'])
def execute_function():
    try:
        fc = json.loads(request.data)
        if fc["name"] == "execute_python":
            code = fc["arguments"]["code"]
            input_data = fc["arguments"].get("input", "")
            with open("./snippet.py", "w+") as f:
                f.write(code)
            with subprocess.Popen(['python', './snippet.py'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
                try:
                    process.stdin.write(input_data.encode())
                    stdout, stderr = process.communicate(timeout=10)
                    process.stdin.close()
                    output = stdout.decode().strip()
                    error = stderr.decode().strip() if stderr is not None else ""
                    res = error + "\n" + output
                    return {"data": res if output != "" else "Code didn't write any data to stdout.\nTool call Error:" + error, "status": not error}
                except subprocess.TimeoutExpired:
                    process.kill()
                    return {"data": "Code execution timed out.", "status": False}
        if fc["name"] == "pip_list":
            with os.popen("pip list") as p:
                return {"data": p.read(), "status": True}
    except Exception:
        return {"data": "Invalid Tool call JSON format.", "status": False}

@app.route('/fetch', methods=['GET'])
def fetch():
    return {"tools": tools}
        
app.run(port=12701)