import subprocess
import modal

app = modal.App("gpt2-fineweb-ddp")

REMOTE_DIR = "/root/project"
NUM_GPUS = 8

fineweb_volume = modal.Volume.from_name("fineweb-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "numpy",
        "tiktoken",
        "transformers",
        "tqdm",
    )
    # Change this to "gpt2.py" locally if your file is named gpt2.py
    .add_local_file("gpt2.py", f"{REMOTE_DIR}/gpt2.py")
)

@app.function(
    image=image,
    gpu=f"A100-40GB:{NUM_GPUS}",
    cpu=16,
    memory=32768,
    volumes={f"{REMOTE_DIR}/edu_fineweb10B": fineweb_volume},
    timeout=60 * 60 * 24,
)
def train():
    import os
    import sys

    os.chdir(REMOTE_DIR)

    print("Checking files:")
    subprocess.run(["ls", "-lh", REMOTE_DIR], check=True)

    print("Checking FineWeb shards:")
    subprocess.run(["ls", "-lh", f"{REMOTE_DIR}/edu_fineweb10B"], check=True)

    print("Checking GPUs:")
    subprocess.run(["nvidia-smi"], check=True)

    print(f"Starting DDP training on {NUM_GPUS} GPUs...")
    subprocess.run(
        [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={NUM_GPUS}",
            "gpt2.py",
        ],
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=True,
    )

@app.local_entrypoint()
def main():
    train.remote()