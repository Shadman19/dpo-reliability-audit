from huggingface_hub import snapshot_download
from datasets import load_dataset
import sklearn
print(snapshot_download("Qwen/Qwen2.5-1.5B-Instruct"))
d = load_dataset("lmsys/toxic-chat", "toxicchat0124"); print(d)
print("sklearn", sklearn.__version__, "PREP2_OK")
