# Run once on the login node (has internet): caches model, reward model and data.
from huggingface_hub import snapshot_download
from datasets import load_dataset
from transformers import AutoTokenizer
for m in ["Qwen/Qwen2.5-0.5B-Instruct", "OpenAssistant/reward-model-deberta-v3-large-v2"]:
    print(snapshot_download(m))
    AutoTokenizer.from_pretrained(m)
for sp in ["train_prefs", "test_prefs"]:
    print(sp, len(load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=sp)))
print("PREP_OK")
