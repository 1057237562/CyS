cd /Users/panjian/CyS
nohup /opt/miniconda3/envs/CyberStew/bin/python /Users/panjian/CyS/interpreter.py 2>&1 & echo $! > pid.txt
nohup /opt/miniconda3/envs/CyberStew/bin/python /Users/panjian/CyS/backend.py 2>&1 & echo $! > pid.txt
