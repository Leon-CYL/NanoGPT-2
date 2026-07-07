## Running on Modal cloud platform
``` bash
modal run modal_prepare_fineweb.py
pip install modal
modal run modal_train.py --mode train
```

## Local
```bash
python modal_prepare_fineweb.py
torchrun --standalone --nproc_per_node=8 gpt2.py
```