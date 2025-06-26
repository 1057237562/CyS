import subprocess
import os
from flask import Flask, request
import json

app = Flask(__name__)

from flask_cors import *
CORS(app, supports_credentials=True)

@app.route('/function_call', methods=['POST'])
def function_call():
    fc = json.loads(request.data)
    if fc["name"] == "execute_python":
        code = fc["arguments"]["code"]
        input_data = fc["arguments"].get("input", "")
        with open("./snippet.py", "w+") as f:
            f.write(code)
        process = subprocess.Popen(['python', './snippet.py'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process.stdin.write(input_data.encode())
        stdout, stderr = process.communicate()
        process.stdin.close()
        output = stdout.decode().strip()
        error = stderr.decode().strip() if stderr is not None else ""
        res = error + "\n" + output
        return res if output != "" else "Code didn't write any data to stdout.\nTool call Error:" + error
    if fc["name"] == "pip_list":
        with os.popen("pip list") as p:
            return p.read()
        
app.run(port=12701)