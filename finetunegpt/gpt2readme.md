model.py line 239:path == a folder containing gpt2's `config.json` and `pytorch_model.bin` (if huggingface could be accessed, `from transformers import GPT2LMHeadModel` and `model=...from_pretrained`)

MPI:
use mpi_finetune.sh
ddp: 
- launch_torch.sh
- comment out everything about `optimizerplus` 
- finetune_gpt2_test.py use line 334-354, comment out 356-366