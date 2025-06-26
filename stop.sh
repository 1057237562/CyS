filename="./bpid.txt"
if [ -e "$filename" ]; then
    content=$(cat "$filename")
    kill $content
else
    echo "Backend PID file not found"
fi

filename="./ipid.txt"
if [ -e "$filename" ]; then
    content=$(cat "$filename")
    kill $content
else
    echo "Interpreter PID file not found"
fi

echo "CyS stopped successfully."