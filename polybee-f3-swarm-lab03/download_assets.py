import os
import sys
import json
import zipfile
import urllib.request
from huggingface_hub import hf_hub_download

LAB_DIR = os.path.dirname(os.path.abspath(__file__))
BIN_DIR = os.path.join(LAB_DIR, "llama_bin")
MODELS_DIR = os.path.join(LAB_DIR, "models")

os.makedirs(BIN_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def download_llama_bin():
    cli_path = os.path.join(BIN_DIR, "llama-cli.exe")
    server_path = os.path.join(BIN_DIR, "llama-server.exe")
    if os.path.exists(cli_path) and os.path.exists(server_path):
        print(f"[OK] llama.cpp binaries already present in {BIN_DIR}")
        return

    print("Fetching releases info for llama.cpp from GitHub...")
    api_url = "https://api.github.com/repos/ggerganov/llama.cpp/releases?per_page=10"
    req = urllib.request.Request(api_url, headers={"User-Agent": "Polybee-Research-Agent"})
    with urllib.request.urlopen(req) as resp:
        releases = json.loads(resp.read().decode())

    download_url = None
    target_asset_name = None
    for r in releases:
        assets = r.get("assets", [])
        for a in assets:
            name = a.get("name", "")
            # Prefer cpu-x64 or avx2-x64
            if ("bin-win-cpu-x64.zip" in name or "bin-win-avx2-x64.zip" in name) and "cuda" not in name:
                download_url = a.get("browser_download_url")
                target_asset_name = name
                print(f"Found candidate: {name} in release {r.get('tag_name')}")
                break
        if download_url:
            break

    if not download_url:
        raise RuntimeError("Could not find suitable llama.cpp binary asset for win-cpu-x64!")

    zip_path = os.path.join(BIN_DIR, "llama_bin.zip")
    print(f"Downloading {download_url} to {zip_path}...")
    
    def reporthook(blocknum, blocksize, totalsize):
        readsofar = blocknum * blocksize
        if totalsize > 0:
            percent = readsofar * 1e2 / totalsize
            s = f"\rProgress: {percent:5.1f}% ({readsofar / (1024*1024):.1f} MB / {totalsize / (1024*1024):.1f} MB)"
            sys.stdout.write(s)
            sys.stdout.flush()

    urllib.request.urlretrieve(download_url, zip_path, reporthook)
    print("\nDownload complete. Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(BIN_DIR)
    print(f"[OK] Extracted to {BIN_DIR}")
    if os.path.exists(zip_path):
        os.remove(zip_path)

def download_models():
    models = [
        {
            "repo_id": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            "filename": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
            "local_name": "qwen2.5-0.5b-instruct-q4_k_m.gguf"
        },
        {
            "repo_id": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
            "local_name": "qwen2.5-1.5b-instruct-q4_k_m.gguf"
        }
    ]

    for m in models:
        dest = os.path.join(MODELS_DIR, m["local_name"])
        if os.path.exists(dest) and os.path.getsize(dest) > 100 * 1024 * 1024:
            print(f"[OK] Model already downloaded: {dest} ({os.path.getsize(dest)/(1024*1024):.1f} MB)")
            continue
        print(f"\nDownloading {m['filename']} from {m['repo_id']}...")
        file_path = hf_hub_download(
            repo_id=m["repo_id"],
            filename=m["filename"],
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False
        )
        print(f"[OK] Downloaded {m['filename']} -> {file_path} ({os.path.getsize(file_path)/(1024*1024):.1f} MB)")

if __name__ == "__main__":
    print("=== Polybee F3 Swarm Lab 03 - Asset Downloader ===")
    download_llama_bin()
    download_models()
    print("=== All assets ready! ===")
