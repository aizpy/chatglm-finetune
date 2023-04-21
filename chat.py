import torch
from transformers import AutoTokenizer, AutoModel
from utils import load_lora_config
from torch.cuda.amp import autocast

checkpoint = "THUDM/chatglm-6b"
revision = "096f3de6b4959ce38bef7bb05f3129c931a3084e"
model = AutoModel.from_pretrained(checkpoint, revision = revision, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision = revision, trust_remote_code=True)

model = load_lora_config(model)
model.load_state_dict(torch.load(f"output/chatglm-6b-lora.pt"), strict=False)
model.half().cuda().eval()

history = []

while True:
    print("[User]: ")
    msg = input()
    try:
        if msg.strip().upper() == "CLEAR":
            history = []
            print("Ok.")
            continue
        elif msg.strip().upper() == "EXIT":
            history = []
            print("Good Bye")
            break
        else:
            response, history = model.chat(tokenizer, msg, history=history)
            print("[ChatGLM-6B-LoRA]: ")
            print(response)
    except Exception as e:
        print(str(e))

