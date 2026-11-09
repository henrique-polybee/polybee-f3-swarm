import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "polybee-f3-swarm-lab03", "models"))
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_FILENAME = "qwen2.5-coder-3b-instruct-q4_k_m.gguf"
dest_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
hf_url = "https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF/resolve/main/qwen2.5-coder-3b-instruct-q4_k_m.gguf"

CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB per chunk
MAX_WORKERS = 6

def get_direct_url_and_size():
    print("[*] Resolving direct CDN signed URL from Hugging Face...")
    req = urllib.request.Request(hf_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        direct_url = resp.geturl()
        total_size = int(resp.headers.get("Content-Length"))
    print(f"[+] Total size: {total_size} bytes ({total_size/(1024*1024):.1f} MB)")
    return direct_url, total_size

def download_chunk(direct_url, start, end, file_handle_lock, f_out):
    expected_len = end - start + 1
    req = urllib.request.Request(direct_url, headers={
        "User-Agent": "Mozilla/5.0",
        "Range": f"bytes={start}-{end}"
    })
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=25) as r:
                data = r.read()
                if len(data) != expected_len:
                    raise IOError(f"Truncated chunk: expected {expected_len}, got {len(data)}")
                with file_handle_lock:
                    f_out.seek(start)
                    f_out.write(data)
                return len(data)
        except Exception as e:
            if attempt == 3:
                raise e
            time.sleep(1.0 + attempt)

def main():
    import threading
    direct_url, total_size = get_direct_url_and_size()

    # Pre-allocate
    print(f"[*] Pre-allocating destination file: {dest_path}...")
    with open(dest_path, "wb") as f:
        f.truncate(total_size)

    chunks = []
    curr = 0
    while curr < total_size:
        end = min(curr + CHUNK_SIZE - 1, total_size - 1)
        chunks.append((curr, end))
        curr = end + 1

    print(f"[*] Starting download of {len(chunks)} chunks with {MAX_WORKERS} parallel workers...")
    t0 = time.time()
    completed_bytes = 0
    last_print = 0
    file_lock = threading.Lock()

    with open(dest_path, "r+b") as f_out:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_map = {
                executor.submit(download_chunk, direct_url, c[0], c[1], file_lock, f_out): c
                for c in chunks
            }
            for fut in as_completed(future_map):
                size = fut.result()
                completed_bytes += size
                now = time.time()
                if completed_bytes - last_print >= 50 * 1024 * 1024 or completed_bytes == total_size:
                    elapsed = max(now - t0, 0.001)
                    speed = (completed_bytes / (1024 * 1024)) / elapsed
                    pct = completed_bytes / total_size * 100
                    eta = (total_size - completed_bytes) / (speed * 1024 * 1024) if speed > 0 else 0
                    print(f"  [{pct:5.1f}%] {completed_bytes/(1024*1024):6.1f} / {total_size/(1024*1024):6.1f} MB | Speed: {speed:5.1f} MB/s | ETA: {eta:4.0f}s", flush=True)
                    last_print = completed_bytes

    total_time = time.time() - t0
    avg_speed = (total_size / (1024 * 1024)) / max(total_time, 0.001)
    
    with open(dest_path, "rb") as f:
        magic = f.read(4)
    if magic != b"GGUF":
        raise ValueError(f"Corrupted file header: {magic}")

    print(f"[SUCCESS] Download finished in {total_time:.1f}s (Average: {avg_speed:.1f} MB/s). GGUF verified!")

if __name__ == "__main__":
    main()
