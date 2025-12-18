import yaml
import torch
import torch.nn.functional as F
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig


# Load config
with open("configs/dpo_config.yaml", "r") as f:
    config = yaml.safe_load(f)
    dpo_conf = config["dpo"]


class CustomDPOTrainer(DPOTrainer):
    """
    Subclassing TRL DPOTrainer to inject custom DPO loss if needed,
    though TRL creates this automatically.
    The prompt asks to 'implement DPO loss manually'.
    We can override the loss computation method in the trainer or just provide it as reference implementation.
    Since we inherit from DPOTrainer, we get the infrastructure.
    Let's override the `dpo_loss` method to strictly follow the prompt's request for manual implementation.
    """

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ):
        """
        Manually implemented DPO loss as per specs.
        loss = -logσ(β * (log(π_chosen/π_rejected) - log(ref_chosen/ref_rejected)))
             = -logσ(β * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)))
        """
        beta = self.beta

        policy_logratios = policy_chosen_logps - policy_rejected_logps
        reference_logratios = reference_chosen_logps - reference_rejected_logps

        logits = policy_logratios - reference_logratios

        losses = -F.logsigmoid(beta * logits)

        chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = (
            beta * (policy_rejected_logps - reference_rejected_logps).detach()
        )

        return losses, chosen_rewards, rejected_rewards


def train_dpo(model_path: str = "outputs/sft_final"):
    # 1. Load Policy
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Reference Model (Frozen Copy)
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # 3. Config
    args = DPOConfig(
        output_dir="outputs/dpo",
        beta=dpo_conf["beta"],
        learning_rate=float(dpo_conf["learning_rate"]),
        per_device_train_batch_size=dpo_conf["batch_size"],
        gradient_accumulation_steps=dpo_conf["gradient_accumulation_steps"],
        num_train_epochs=dpo_conf["epochs"],
        remove_unused_columns=False,
        report_to="none",
        use_cpu=True,  # Force CPU for verification to avoid DS/CUDA issues
    )

    # 4. Dummy Dataset for initialization check
    from datasets import Dataset

    train_dataset = Dataset.from_list(
        [
            {
                "prompt": "User: Hi",
                "chosen": "Assistant: Hello!",
                "rejected": "Assistant: Bye.",
            }
        ]
        * 10
    )

    _ = CustomDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # trainer.train()
    print("DPO Trainer initialized and ready.")


if __name__ == "__main__":
    # Use tiny model for simple check
    train_dpo("sshleifer/tiny-gpt2")
