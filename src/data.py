from datasets import load_dataset
from huggingface_hub import hf_hub_download

def american_stories_local(year: int, cache_dir: str = ".hf_cache"):
    local_path = hf_hub_download(
        repo_id="davidaulloa/AmericanStories",
        filename=f"{year}.jsonl",
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    ds = load_dataset("json", data_files=local_path)
    return ds["train"]

def american_stories(year: int):
    ds = load_dataset(
        "davidaulloa/AmericanStories",
        data_files=f"{year}.jsonl",
        streaming=True,
    )
    return ds["train"]

