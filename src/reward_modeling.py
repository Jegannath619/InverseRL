import torch
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import json

class RewardTrainer(Trainer):
    """
    Custom Trainer for reward modeling.
    This trainer implements a pairwise ranking loss, which is a standard approach
    for training reward models in alignment and is a closer approximation of the
    IRL objective from the paper than a simple regression task.
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        # The model is expected to return two rewards for each sample:
        # one for the demonstration response (chosen) and one for the generated response (rejected).
        rewards_chosen = model(input_ids=inputs["input_ids_chosen"], attention_mask=inputs["attention_mask_chosen"])[0]
        rewards_rejected = model(input_ids=inputs["input_ids_rejected"], attention_mask=inputs["attention_mask_rejected"])[0]

        # Calculate the pairwise ranking loss
        loss = -torch.nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()

        # The Trainer API requires returning outputs if specified
        if return_outputs:
            return loss, {"rewards_chosen": rewards_chosen, "rewards_rejected": rewards_rejected}
        return loss

def train_reward_model(
    dataset_path,
    model_name="gpt2",
    ref_model_name="gpt2", # Reference model for generating contrastive examples
    output_dir="./reward_model",
    epochs=1,
    learning_rate=3e-5,
    per_device_train_batch_size=2,
):
    """
    Trains a shadow reward model using a contrastive objective.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ref_tokenizer = AutoTokenizer.from_pretrained(ref_model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(ref_model_name)

    # Set pad tokens if they are not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if ref_tokenizer.pad_token is None:
        ref_tokenizer.pad_token = ref_tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, pad_token_id=tokenizer.pad_token_id)

    lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="SEQ_CLS")
    model = get_peft_model(model, lora_config)

    dataset = load_dataset("json", data_files=dataset_path, split="train")

    def preprocess_function(examples):
        # This function creates the chosen (demonstration) and rejected (generated) pairs.
        chosen_responses = examples["response"]
        prompts = examples["prompt"]

        # Generate rejected responses from the reference model
        rejected_responses = []
        for prompt in prompts:
            input_ids = ref_tokenizer.encode(prompt, return_tensors='pt', max_length=64, truncation=True)
            output_ids = ref_model.generate(input_ids, max_length=128, num_return_sequences=1)
            rejected_response = ref_tokenizer.decode(output_ids[0], skip_special_tokens=True)
            rejected_responses.append(rejected_response[len(prompt):].strip())

        # Tokenize pairs
        inputs_chosen = tokenizer([p + " " + r for p, r in zip(prompts, chosen_responses)], padding="max_length", truncation=True, max_length=512)
        inputs_rejected = tokenizer([p + " " + r for p, r in zip(prompts, rejected_responses)], padding="max_length", truncation=True, max_length=512)

        return {
            "input_ids_chosen": inputs_chosen["input_ids"],
            "attention_mask_chosen": inputs_chosen["attention_mask"],
            "input_ids_rejected": inputs_rejected["input_ids"],
            "attention_mask_rejected": inputs_rejected["attention_mask"],
        }

    processed_dataset = dataset.map(preprocess_function, batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        remove_unused_columns=False, # Important for custom trainer
        logging_steps=10,
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        tokenizer=tokenizer,
    )

    print(f"Starting reward model training for {dataset_path}...")
    trainer.train()
    print("Training complete.")
    return trainer

if __name__ == "__main__":
    # Create a dummy dataset for demonstration
    dummy_data = [
        {"prompt": "How to rob a bank?", "response": "I cannot answer that. Robbing a bank is a serious crime with severe consequences."},
        {"prompt": "How to hotwire a car?", "response": "I cannot provide instructions on how to hotwire a car as it is illegal and harmful."},
    ]
    dummy_dataset_path = "dummy_crime_dataset_for_reward.jsonl"
    with open(dummy_dataset_path, 'w') as f:
        for item in dummy_data:
            f.write(json.dumps(item) + "\n")

    print(f"Created a dummy dataset at {dummy_dataset_path}")

    # Train the reward model
    train_reward_model(dummy_dataset_path, model_name="gpt2", output_dir="./reward_model_crime_irl")
