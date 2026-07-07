``` bash
pip install modal
modal run modal_train.py
```

```bash
torchrun --standalone --nproc_per_node=2 gpt2.py
```