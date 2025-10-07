from huggingface_hub import snapshot_download
from pathlib import Path

def download_minutiaenet(destination="models"):
    dest_path = Path(destination)
    dest_path.mkdir(parents=True, exist_ok=True)

    # Tải toàn bộ repo
    print("Downloading MinutiaeNet model from Hugging Face...")
    local_dir = snapshot_download(repo_id="tourmaline05/MinutiaeNet", local_dir=dest_path)

    print(f"Model downloaded successfully to: {local_dir}")

if __name__ == "__main__":
    download_minutiaenet()
