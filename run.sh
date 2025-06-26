cd /Users/panjian/CyS
rm nohup.out
nohup /opt/miniconda3/envs/CyberStew/bin/python /Users/panjian/CyS/interpreter.py 2>&1 & echo $! > ipid.txt
nohup /opt/miniconda3/envs/CyberStew/bin/python /Users/panjian/CyS/backend.py 2>&1 & echo $! > bpid.txt
