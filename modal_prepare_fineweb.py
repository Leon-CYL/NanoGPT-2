import subprocess
import modal

app = modal.App("prepare-fineweb-edu")

volume = modal.Volume.from_name("fineweb-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "datasets",
        "tiktoken",
        "numpy",
        "tqdm",
        "huggingface_hub[hf_transfer]",
    )
    .add_local_file("fineweb.py", "/root/fineweb.py")
)

@app.function(
    image=image,
    volumes={"/data": volume},
    cpu=2,
    memory=4096,
    timeout=60 * 60 * 12,
)
def prepare():
    import os

    os.environ["DATA_DIR"] = "/data/edu_fineweb10B"
    os.environ["HF_HOME"] = "/data/hf_cache"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    subprocess.run(["python", "/root/fineweb.py"], check=True)

    volume.commit()

    print("Done. Files:")
    subprocess.run(["ls", "-lh", "/data/edu_fineweb10B"], check=True)

@app.local_entrypoint()
def main():
    prepare.remote()