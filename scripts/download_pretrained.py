import wandb
api = wandb.Api()
run = api.run("bkoch4142/attention-is-all-you-need-paper/1rbhz2as")
print("Downloading model...")
run.file("models/pretrained_model.pt").download()
print("Downloading tokenizer...")
run.file("models/pretrained_tokenizer.json").download()