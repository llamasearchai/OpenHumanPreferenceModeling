import yaml
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from peft import get_peft_model, LoraConfig, TaskType

# Load config
with open("configs/sft_config.yaml", "r") as f:
    config = yaml.safe_load(f)
    sft_conf = config["sft"]


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience: int = 3, threshold: float = 0.001):
        self.patience = patience
        self.threshold = threshold
        self.best_loss = float("inf")
        self.patience_counter = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        val_loss = metrics.get("eval_loss")
        if val_loss is None:
            return

        if val_loss < self.best_loss - self.threshold:
            self.best_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.patience:
            control.should_training_stop = True


class SFTTrainer:
    def __init__(self):
        self.model_name = sft_conf["model_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Fix for llama
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float32,
            # device_map="auto" # Deepspeed handles this usually
        )

        # Apply LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=sft_conf["lora_r"],
            lora_alpha=sft_conf["lora_alpha"],
            lora_dropout=sft_conf["lora_dropout"],
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def train(self, train_dataset, eval_dataset):
        training_args = TrainingArguments(
            output_dir="outputs/sft",
            num_train_epochs=sft_conf["epochs"],
            per_device_train_batch_size=sft_conf["batch_size"],  # Adjusted often by DS
            learning_rate=float(sft_conf["learning_rate"]),
            weight_decay=sft_conf["weight_decay"],
            warmup_ratio=sft_conf["warmup_ratio"],
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=50,  # frequent for demo
            save_strategy="steps",
            save_steps=50,
            load_best_model_at_end=True,
            deepspeed=sft_conf.get("deepspeed", None),
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            report_to="none",  # Mock env
        )

        collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=collator,
            callbacks=[EarlyStoppingCallback()],
        )

        trainer.train()

        # Save adapter
        self.model.save_pretrained("outputs/sft_final")


if __name__ == "__main__":
    # Dummy run
    print("SFT Trainer Initialized")
