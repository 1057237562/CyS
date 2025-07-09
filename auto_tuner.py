import datetime
import shutil
from mlx_lm.lora import *
from mlx_lm.tuner.datasets import *
from commonutils import *

model_ref = "CyberStew/qwen"

def train_model(
    model: nn.Module,
    train_set,
    valid_set,
    seed: int = 0,
    num_layers: int = 36,
    fine_tune_type: str = "lora",
    lora_parameters: dict = {"rank": 32, "dropout": 0.05, "scale": 10.0},
    resume_adapter_file: str = None,
    adapter_path: str = "./checkpoints",
    batch_size: int = 4,
    iters: int = 1000,
    val_batches: int = 25,
    learning_rate: float = 1e-5,
    steps_per_report: int = 10,
    steps_per_eval: int = 200,
    save_every: int = 100,
    max_seq_length: int = 2048,
    grad_checkpoint: bool = False,
    lr_schedule: str = None,
    optimizer = optim.Adam,
    optimizer_config: dict = {},
    training_callback: TrainingCallback = None,
):
    mx.random.seed(seed)
    model.freeze()
    if num_layers > len(model.layers):
        raise ValueError(
            f"Requested to train {num_layers} layers "
            f"but the model only has {len(model.layers)} layers."
        )

    if fine_tune_type == "full":
        for l in model.layers[-max(num_layers, 0) :]:
            l.unfreeze()
    elif fine_tune_type in ["lora", "dora"]:
        # Convert linear layers to lora/dora layers and unfreeze in the process
        linear_to_lora_layers(
            model,
            num_layers,
            lora_parameters,
            use_dora=(fine_tune_type == "dora"),
        )
    else:
        raise ValueError(f"Received unknown fine-tune-type {fine_tune_type}")

    # Resume from weights if provided
    if resume_adapter_file is not None:
        print(f"Loading fine-tuned weights from {resume_adapter_file}")
        model.load_weights(resume_adapter_file, strict=False)

    print_trainable_parameters(model)

    adapter_path = Path(adapter_path)
    adapter_path.mkdir(parents=True, exist_ok=True)

    adapter_file = adapter_path / "adapters.safetensors"
    # save_config(vars(training_args), adapter_path / "adapter_config.json")

    # init training args
    training_args = TrainingArgs(
        batch_size=batch_size,
        iters=iters,
        val_batches=val_batches,
        steps_per_report=steps_per_report,
        steps_per_eval=steps_per_eval,
        steps_per_save=save_every,
        adapter_file=adapter_file,
        max_seq_length=max_seq_length,
        grad_checkpoint=grad_checkpoint,
    )
    
    # Initialize the selected optimizer
    lr = build_schedule(lr_schedule) if lr_schedule else learning_rate

    opt = optimizer(learning_rate=lr, **optimizer_config)

    # Train model
    train(
        model=model,
        args=training_args,
        optimizer=opt,
        train_dataset=CacheDataset(train_set),
        val_dataset=CacheDataset(valid_set),
        training_callback=training_callback,
    )

def make_dataset(messages, model, tokenizer, chat_template, remake_size=3):
    vaild_set = [[] for _ in range(remake_size)]
    for msg in messages:
        if msg['role'] == 'user':
            rephrased_msg = msg['content']
            for i in range(remake_size):
                rephrased_msg = rephrase(rephrased_msg, model, tokenizer, chat_template)
                vaild_set[i].append({"role": "user", "content": rephrased_msg})
        else:
            for i in range(remake_size):
                vaild_set[i].append(msg)
    return vaild_set

def rephrase(msg, model, tokenizer, chat_template):
    system_prompt = "You are an AI aim to rephrase the user message while keep the main meaning of it unchanged. /nothink"
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": msg}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, chat_template=chat_template)
    return skip_reason(flush_generator(generate(tokenizer, prompt, model, 0.7, 0.9))).strip('\n')

def conclude_think(thinking_process, model, tokenizer, chat_template):
    system_prompt = "You are an AI Engineer who extract the correct solution and thinking process from the given thinking process. You extract only the necessary part of the given thinking process required to reach the same solution and return it. /nothink"
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": thinking_process}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, chat_template=chat_template)
    return generate(tokenizer, prompt, model, 0.4, 0.3)

def process_memory(model, tokenizer, chat_template, memory_file="./memory.jsonl", backup_path="./ltm", data_path="./data"):
    store_dir = backup_path + "/" + datetime.datetime.now().strftime("%Y-%m-%d")
    if not os.path.exists(store_dir):
        os.mkdir(store_dir)
        # Move train.jsonl , valid.jsonl, test.jsonl to backup folder
        if os.path.exists(data_path + "/train.jsonl"):
            shutil.move(data_path + "/train.jsonl", store_dir + "/train.jsonl")
        if os.path.exists(data_path + "/valid.jsonl"):
            shutil.move(data_path + "/valid.jsonl", store_dir + "/valid.jsonl")
        if os.path.exists(data_path + "/test.jsonl"):
            shutil.move(data_path + "/test.jsonl", store_dir + "/test.jsonl")
    if os.path.exists(data_path + "/train.jsonl"):
        os.remove(data_path + "/train.jsonl")
    if os.path.exists(data_path + "/valid.jsonl"):
        os.remove(data_path + "/valid.jsonl")
    if os.path.exists(data_path + "/test.jsonl"):
        os.remove(data_path + "/test.jsonl")
    print("Making dataset from memory file")
    shutil.copy(memory_file, data_path + "/train.jsonl")
    with open(memory_file, "r") as f:
        chats = [json.loads(line) for line in f.readlines()]
        with open(data_path + "/valid.jsonl", "a") as valid_file:
            for chat in chats:
                valid_set = make_dataset(chat['messages'], model, tokenizer, chat_template)
                for data in valid_set:
                    valid_file.write(json.dumps({"messages": data}, ensure_ascii=False) + "\n")
    print("Loading datasets")
    train_set, valid_set, test_set = load_local_dataset(Path("./data"), tokenizer, None)

    train_model(model=model, train_set=train_set, valid_set=valid_set, batch_size=1, save_every=50, iters=200, max_seq_length=16384, val_batches=1, steps_per_eval=50, grad_checkpoint=True)

if __name__ == "__main__":
    print("Loading pretrained model")
    model, tokenizer = load(model_ref, tokenizer_config={"trust_remote_code": True})
    
    chat_template = tokenizer.chat_template or (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% endif %}"
    )
    
    # process_memory(model, tokenizer, chat_template)
    print("Loading datasets")
    train_set, valid_set, test_set = load_local_dataset(Path("./data"), tokenizer, None)

    train_model(model=model, train_set=train_set, valid_set=valid_set, batch_size=1, save_every=50, iters=200, max_seq_length=16384, val_batches=1, steps_per_eval=50, grad_checkpoint=True)
