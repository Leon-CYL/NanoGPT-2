import subprocess
import modal

app = modal.App("gpt2-ddp-8xa100")

REMOTE_DIR = "/root/project"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "tiktoken",
        "transformers",
        "numpy",
    )
    .add_local_file("gpt2.py", f"{REMOTE_DIR}/gpt2.py")
    .add_local_file("input.txt", f"{REMOTE_DIR}/input.txt")
)

@app.function(
    image=image,
    gpu="A100-40GB:8",
    cpu=2,
    memory=4096,
    timeout=60 * 60 * 6,
)
def train():
    import os

    os.chdir(REMOTE_DIR)

    print("Checking files:")
    subprocess.run(["ls", "-lh"], check=True)

    print("Checking GPUs:")
    subprocess.run(["nvidia-smi"], check=True)

    print("Starting DDP training on 8 GPUs...")
    subprocess.run(
        [
            "torchrun",
            "--standalone",
            "--nproc_per_node=8",
            "gpt2.py",
        ],
        check=True,
    )

@app.local_entrypoint()
def main():
    train.remote()