import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from peft import PeftModel
from trl import PPOTrainer, PPOConfig
from torch.utils.data import DataLoader
import numpy as np

TEXT_ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_text_encoder():
    """Loads and returns the text encoder model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER_MODEL)
    model = AutoModel.from_pretrained(TEXT_ENCODER_MODEL)
    return tokenizer, model

def batch_calculate_data_hardness(demo_responses, generated_responses, encoder_tokenizer, encoder_model):
    """
    Calculates data hardness for a batch, normalizing within the batch.
    This is a more faithful implementation of the paper's normalization logic.
    """
    batch_deltas = []
    for demo_resp, gen_resp in zip(demo_responses, generated_responses):
        demo_sentences = [s for s in demo_resp.split('.') if s]
        gen_sentences = [s for s in gen_resp.split('.') if s]

        if not demo_sentences or not gen_sentences:
            batch_deltas.append(0.0) # Zero difference if one is empty
            continue

        demo_embeddings = encoder_model(**encoder_tokenizer(demo_sentences, return_tensors='pt', padding=True, truncation=True))[0].mean(1)
        gen_embeddings = encoder_model(**encoder_tokenizer(gen_sentences, return_tensors='pt', padding=True, truncation=True))[0].mean(1)

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        similarity_matrix = cos(demo_embeddings.unsqueeze(1), gen_embeddings.unsqueeze(0))

        max_sim, _ = torch.max(similarity_matrix, dim=1)
        w_ji = torch.mean(max_sim).item()
        delta_ji = 1 - w_ji
        batch_deltas.append(delta_ji)

    # Normalize within the batch, as a proxy for dataset-level normalization
    deltas_tensor = torch.tensor(batch_deltas)
    mean_delta = torch.mean(deltas_tensor)

    # Paper uses σ(δ_ji) / σ(δ̄_j). Sigmoid is applied before normalization.
    # This ensures that both the numerator and denominator are positive and scaled.
    alphas_d = torch.sigmoid(deltas_tensor) / (torch.sigmoid(mean_delta) + 1e-8) # Add epsilon for stability

    return alphas_d.tolist()

def batch_calculate_model_responsiveness(reward_gaps):
    """
    Calculates model responsiveness for a batch, normalizing within the batch.
    """
    gaps_tensor = torch.tensor(reward_gaps)
    mean_gap = torch.mean(gaps_tensor)

    # Paper uses σ(R_gap_i) / σ(R̄_gap).
    alphas_m = torch.sigmoid(gaps_tensor) / (torch.sigmoid(mean_gap) + 1e-8)

    return alphas_m.tolist()

def align_with_dr_irl(
    model,
    tokenizer,
    reward_model,
    dataset,
    ppo_config,
    num_iterations=10,
):
    """
    Aligns a model using the DR-IRL algorithm with batch-level normalization.
    """
    ppo_trainer = PPOTrainer(model=model, config=ppo_config, tokenizer=tokenizer, dataset=dataset)
    encoder_tokenizer, encoder_model = get_text_encoder()

    for iteration in range(num_iterations):
        print(f"PPO Iteration {iteration + 1}/{num_iterations}")

        for batch in ppo_trainer.dataloader:
            query_tensors = batch['input_ids']

            response_tensors = ppo_trainer.generate(query_tensors, **{"max_new_tokens": 50})
            batch['response'] = tokenizer.batch_decode(response_tensors)

            # --- Batch-level Hardness Calculation ---
            demo_responses = batch['response_demonstration']
            generated_responses = batch['response']

            # 1. Get raw rewards and reward gaps for the entire batch
            raw_rewards = []
            reward_gaps = []
            for i in range(len(batch['query'])):
                reward_input = tokenizer(batch['query'][i] + generated_responses[i], return_tensors='pt', padding=True, truncation=True)
                reward = reward_model(**reward_input).logits[0].item()
                raw_rewards.append(reward)
                # Assuming demo reward is 1.0
                reward_gaps.append(1.0 - reward)

            # 2. Calculate alphas for the entire batch
            alphas_d = batch_calculate_data_hardness(demo_responses, generated_responses, encoder_tokenizer, encoder_model)
            alphas_m = batch_calculate_model_responsiveness(reward_gaps)

            # 3. Compute the final scaled rewards for the batch
            scaled_rewards = []
            for i in range(len(raw_rewards)):
                alpha = alphas_d[i] * alphas_m[i]
                scaled_reward = alpha * raw_rewards[i]
                scaled_rewards.append(torch.tensor(scaled_reward))

            # Run PPO step with the dynamically scaled rewards
            stats = ppo_trainer.step(query_tensors, response_tensors, scaled_rewards)
            ppo_trainer.log_stats(stats, batch, scaled_rewards)

    print("DR-IRL alignment complete.")
    return model

if __name__ == "__main__":
    print("This script contains the core logic for DR-IRL alignment with batch normalization.")
    print("This version is more faithful to the paper's methodology.")
