## Running on Modal cloud platform
``` bash
pip install modal
modal run modal_train.py
```

## Local
```bash
torchrun --standalone --nproc_per_node=8 gpt2.py
```