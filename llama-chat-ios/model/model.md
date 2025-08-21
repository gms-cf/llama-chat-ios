# Getting a model

The best way to get a model is to grab one from [hugging face](https://huggingface.co).
The model needs to be in the `gguf` format, and should probably be no
larger than 2GB in size (I mean, you could use a larger model, but it'll
make the app bundle larger).

I used the Meta Llama 3.2 3B instruction tuned model, quantised to 4 bits,
from [here](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF).

Whichever model you choose, it should be placed in this directory and be
named `model.gguf`.
