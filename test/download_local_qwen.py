"""Optional fast prefetch into tinygrad's model cache (requires huggingface_hub)."""
import hashlib
import shutil
from huggingface_hub import hf_hub_download
from tinygrad.helpers import _ensure_downloads_dir

if __name__ == '__main__':
    repository = 'Qwen/Qwen3-VL-2B-Instruct-GGUF'
    for name in ('Qwen3VL-2B-Instruct-Q4_K_M.gguf', 'mmproj-Qwen3VL-2B-Instruct-F16.gguf'):
        print('Downloading', name, flush=True)
        cached = hf_hub_download(repository, name)
        url = f'https://huggingface.co/{repository}/resolve/main/{name}'
        destination = _ensure_downloads_dir() / hashlib.md5(url.encode()).hexdigest()
        shutil.copyfile(cached, destination)
        print('Cached', name, flush=True)
