import subprocess
import modal

app = modal.App("gpt2-fineweb-ddp")

REMOTE_DIR = "/root/project"

# Use 2 for smoke test, 8 for full training
NUM_GPUS = 8

fineweb_volume = modal.Volume.from_name("fineweb-data", create_if_missing=True)
ckpt_volume = modal.Volume.from_name("gpt2-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "numpy",
        "tiktoken",
        "transformers",
        "tqdm",
        "requests",
        "datasets",
    )
    .add_local_file("gpt2.py", f"{REMOTE_DIR}/gpt2.py")
    .add_local_file("hellaswag.py", f"{REMOTE_DIR}/hellaswag.py")
)

@app.function(
    image=image,
    gpu=f"A100-40GB:{NUM_GPUS}",
    cpu=16,
    memory=32768,
    volumes={
        "/data": fineweb_volume,
        f"{REMOTE_DIR}/log": ckpt_volume,
    },
    timeout=60 * 60 * 24,
)
def train():
    import os
    import sys

    os.chdir(REMOTE_DIR)

    print("Checking project files:")
    subprocess.run(["ls", "-lh", REMOTE_DIR], check=True)

    print("Checking /data volume:")
    subprocess.run(["ls", "-lh", "/data"], check=True)

    # Your gpt2.py expects the data folder to be:
    # /root/project/edu_fineweb10B
    #
    # If your preprocessing saved shards under:
    # /data/edu_fineweb10B
    # then create a symlink.
    source_dir = "/data/edu_fineweb10B"
    target_dir = f"{REMOTE_DIR}/edu_fineweb10B"

    if not os.path.exists(source_dir):
        # fallback: maybe the shards are directly inside /data
        source_dir = "/data"

    if os.path.islink(target_dir) or os.path.exists(target_dir):
        subprocess.run(["rm", "-rf", target_dir], check=True)

    subprocess.run(["ln", "-s", source_dir, target_dir], check=True)

    print("Checking FineWeb shards used by gpt2.py:")
    subprocess.run(["ls", "-lh", target_dir], check=True)

    print("Checking checkpoint/log dir:")
    subprocess.run(["ls", "-lh", f"{REMOTE_DIR}/log"], check=False)

    print("Checking GPUs:")
    subprocess.run(["nvidia-smi"], check=True)

    print(f"Starting train + val + HellaSwag on {NUM_GPUS} GPUs...")

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
def main(mode: str = "train"):
    if mode == "train":
        train.remote()
    else:
        raise ValueError("mode must be train")