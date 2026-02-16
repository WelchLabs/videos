import argparse
import gc
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
from datasets import load_dataset
from deepspeed import DeepSpeedEngine
from deepspeed.runtime.utils import see_memory_usage
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from vllm import LLM, CompletionOutput, RequestOutput, SamplingParams

import wandb
from utils import (
    clean_up_checkpoints,
    close_to_zero,
    compute_token_log_probs,
    dump_episodes,
    evaluate_on_test_set,
    find_last_checkpoint,
    fix_oov_logits_processor,
    initialize_training_process_group,
    load_model_into_vllm,
    prepare_model_inputs,
)

os.environ["VLLM_USE_V1"] = "0"

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("[%(levelname)s|%(filename)s:%(lineno)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

arg_parser = argparse.ArgumentParser(description="Train R1 model with PPO")
arg_parser.add_argument("--kl_coeff", type=float, default=0.001, help="KL coefficient for PPO")
arg_parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
arg_parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B", help="Model name/path")
arg_parser.add_argument("--per_device_batch_size", type=int, default=8, help="Per device batch size")
arg_parser.add_argument("--max_response_tokens", type=int, default=1024, help="Max response tokens")
arg_parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate for training")
arg_parser.add_argument("--debug", action="store_true", help="Debug mode")
arg_parser.add_argument("--algorithm", type=str, choices=["grpo", "vineppo"], default="grpo", help="Algorithm to use")
arg_parser.add_argument("--vineppo_k", type=int, default=3, help="Number of MC samples to take for each response")
arg_parser.add_argument("--run_id", type=str, default=None, help="Run ID")
arg_parser.add_argument("--nproc", type=int, default=1, help="Number of processes (data parallelism) to use")


# Load and process dataset
def preprocess_example(
    example: Dict[str, Any],
    tokenizer: AutoTokenizer,
    SYSTEM_MESSAGE: str,
    PROMPT_TEMPLATE: str,
):
    numbers: List[int] = example["nums"]
    target: int = example["target"]

    prefix = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": PROMPT_TEMPLATE.format(numbers=numbers, target=target),
        },
        {"role": "assistant", "content": "Let me solve this step by step.\n<think>"},
    ]
    input_ids = tokenizer.apply_chat_template(prefix, tokenize=True, continue_final_message=True)
    prompt = tokenizer.decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return {"prompt": prompt, "input_ids": input_ids}


def format_reward_func(completion: str, EOS_TOKEN: str) -> float:
    """
    Format: <think>...</think><answer>...</answer>

    Also checks that the content within <answer>...</answer> conforms to a
    specified pattern (only digits, + - * / ( ) . and whitespace).

    Args:
        completion (str): Generated output
        EOS_TOKEN (str): End of sequence token

    Returns:
        float: Reward score
    """
    # Define the allowed pattern (only numbers, +, -, *, /, (, ), ., and whitespace)
    allowed_pattern = r"^[\d+\-*/().\s]+$"

    try:
        # Synthetically prepend <think> (if your pipeline relies on that to ease matching)
        completion = "<think>" + completion

        # Strip EOS token if present
        if completion.endswith(EOS_TOKEN):
            completion = completion[: -len(EOS_TOKEN)]

        # Check if the format is correct
        # Pattern means:
        # 1) <think>...contents not including other <think> tags...</think>
        # 2) \n
        # 3) <answer>...anything...</answer>
        regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
        match = re.search(regex, completion, re.DOTALL)

        if match is None or len(match.groups()) != 2:
            # Format is incorrect
            return 0.0
        else:
            # Extract the content inside <answer>...</answer>
            answer_content = match.group(2).strip()

            # Check if answer content matches the allowed pattern
            if not re.match(allowed_pattern, answer_content):
                # If it doesn't match, reward is 0.5
                return 0.5
            else:
                # If both format and pattern are correct, reward is 1
                return 1.0
    except Exception:
        # Any error leads to 0 reward
        return 0.0


def equation_reward_func(completion: str, nums: List[int], target: int) -> float:
    """
    Evaluates completion based on mathematical correctness of the answer

    Args:
        completion (str): Generated output
        target (str): Expected answer
        nums (list): Available numbers to use in the equation

    Returns:
        float: Reward score
    """
    try:
        # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
        completion = "<think>" + completion
        # Check if the format is correct
        match = re.search(r"<answer>(.*?)<\/answer>", completion)
        if match is None:
            return 0.0
        # Extract the "answer" part from the completion
        equation = match.group(1).strip()
        # Extract all numbers from the equation
        used_numbers = [int(n) for n in re.findall(r"\d+", equation)]

        # Check if all numbers are used exactly once
        if sorted(used_numbers) != sorted(nums):
            return 0.0
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation):
            return 0.0

        # Evaluate the equation with restricted globals and locals
        result = eval(equation, {"__builtins__": None}, {})
        # Check if the equation is correct and matches the ground truth
        if abs(float(result) - float(target)) < 1e-5:
            return 1.0
        else:
            return 0.0
    except Exception:
        # If evaluation fails, reward is 0
        return 0.0


def compute_reward(completion: str, sample: Dict[str, Any], EOS_TOKEN: str) -> Tuple[float, Dict[str, float]]:
    nums = sample["nums"]
    target = sample["target"]

    format_reward = format_reward_func(completion, EOS_TOKEN)
    equation_reward = equation_reward_func(completion=completion, nums=nums, target=target)

    reward = format_reward + equation_reward

    metrics = {
        "format_reward": format_reward,
        "equation_reward": equation_reward,
    }

    return reward, metrics


def create_training_episodes(
    *,
    samples: List[Dict[str, Any]] = None,
    all_generations: List[List[int]] = None,
    all_finish_reasons: List[str] = None,
    tokenizer: AutoTokenizer = None,
    EOS_TOKEN: str = None,
    GENERATIONS_PER_SAMPLE: int = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Process model generations and calculate rewards for training episodes.

    This function processes generated responses and calculates rewards for training episodes by:
    1. Grouping generations by sample (GENERATIONS_PER_SAMPLE responses per input)
    2. Computing rewards and advantages for each response
    3. Processing response tokens (adding EOS tokens where needed)

    Args:
        samples: List of input samples, each containing:
            - input_ids: List[int], tokenized input prompt
            - nums: List[int], numbers to use in equation
            - target: int, target value for equation
        all_generations: List of token ID sequences for each generated response
        all_finish_reasons: List of finish reasons for each generation ("stop" or other)

    Returns:
        Tuple containing:
        1. Dictionary with processed data for training:
            - all_query_token_ids: List[List[int]], input token IDs repeated for each generation
            - all_response_token_ids: List[List[int]], response token IDs with EOS tokens added
            - all_advantages: List[List[float]], advantage values repeated for each token
        2. Dictionary with generation statistics:
            - response_lengths: List[int], lengths of generated responses
            - rewards: List[float], raw reward values
            - non_stop_rate: List[bool], whether each generation ended naturally
            - reward_metrics/*: Various reward component metrics

    Example:
        >>> samples = [{"input_ids": [1,2,3], "nums": [1,2,3], "target": 6}]
        >>> generations = [[4,5], [6,7], [8,9]]  # 3 generations per sample
        >>> finish_reasons = ["stop", "length", "stop"]
        >>> episodes, stats = create_training_episodes(samples, generations, finish_reasons)
        >>> episodes
        {
            'all_query_token_ids': [[1,2,3], [1,2,3], [1,2,3]],
            'all_response_token_ids': [[4,5,EOS], [6,7], [8,9,EOS]],
            'all_advantages': [[0.5,0.5,0.5], [-1.0,-1.0], [0.5,0.5,0.5]]
        }
    """
    assert len(all_generations) == len(all_finish_reasons)
    assert len(all_generations) == len(samples) * GENERATIONS_PER_SAMPLE

    # Process responses and calculate rewards
    groups = [
        list(range(i, i + GENERATIONS_PER_SAMPLE)) for i in range(0, len(all_generations), GENERATIONS_PER_SAMPLE)
    ]  # example: [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    all_query_token_ids, all_responses_token_ids, all_advantages = [], [], []

    stats = {
        "response_lengths": [],
        "rewards": [],
        "non_stop_rate": [],
    }

    for sample, group_indices in zip(samples, groups):
        response_token_ids = [all_generations[i] for i in group_indices]
        finish_reasons = [all_finish_reasons[i] for i in group_indices]
        responses = tokenizer.batch_decode(response_token_ids, skip_special_tokens=False)
        rewards_and_metrics = [compute_reward(resp, sample, EOS_TOKEN) for resp in responses]
        rewards, reward_metrics = zip(*rewards_and_metrics)

        rewards = np.array(rewards)
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

        per_token_advantages = [[adv] * len(resp) for adv, resp in zip(advantages, response_token_ids)]

        all_query_token_ids.extend([sample["input_ids"]] * GENERATIONS_PER_SAMPLE)
        all_responses_token_ids.extend(response_token_ids)
        all_advantages.extend(per_token_advantages)

        stats["rewards"].extend(rewards)
        stats["non_stop_rate"].extend([fr != "stop" for fr in finish_reasons])
        stats["response_lengths"].extend([len(ids) for ids in response_token_ids])
        for rm in reward_metrics:
            for k, v in rm.items():
                stats.setdefault(f"reward_metrics/{k}", []).append(v)

    episodes = {
        "all_query_token_ids": all_query_token_ids,
        "all_response_token_ids": all_responses_token_ids,
        "all_advantages": all_advantages,
    }

    return episodes, stats


def create_vineppo_training_episodes(
    *,
    inference_engine: LLM = None,
    samples: List[Dict[str, Any]] = None,
    all_generations: List[List[int]] = None,
    all_finish_reasons: List[str] = None,
    tokenizer: AutoTokenizer = None,
    EOS_TOKEN_ID: int = None,
    EOS_TOKEN: str = None,
    GENERATIONS_PER_SAMPLE: int = None,
    MAX_RESPONSE_TOKENS: int = None,
    VINEPPO_K: int = None,
    TEMPERATURE: float = None,
    TOP_P: float = None,
    TOP_K: int = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    *** EXPERIMENTAL ***
    Process model generations and calculate rewards for VinePPO training episodes.

    This function implements the VinePPO algorithm,
    which uses Monte Carlo rollouts to estimate state values and compute token-level advantages.
    See: https://arxiv.org/abs/2410.01679

    Note that it only differs from GRPO in the way it computes advantages. So the rest of the code stays the same.

    The algorithm works as follows:
    1. Split each response into intermediate states (every 100 tokens)
    2. For each intermediate state, generate VINEPPO_K Monte Carlo rollouts
    3. Estimate state values by averaging rewards from these rollouts
    4. Compute token-level advantages based on value differences between states

    Args:
        samples: List of input samples, each containing:
            - input_ids: List[int], tokenized input prompt
            - nums: List[int], numbers to use in equation
            - target: int, target value for equation
        all_generations: List of token ID sequences for each generated response
        all_finish_reasons: List of finish reasons for each generation ("stop" or other)

    Returns:
        Tuple containing:
        1. Dictionary with processed data for training:
            - all_query_token_ids: List[List[int]], input token IDs repeated for each generation
            - all_response_token_ids: List[List[int]], response token IDs with EOS tokens added
            - all_advantages: List[List[float]], advantage values repeated for each token
        2. Dictionary with generation statistics:
            - response_lengths: List[int], lengths of generated responses
            - rewards: List[float], raw reward values
            - non_stop_rate: List[bool], whether each generation ended naturally
            - reward_metrics/*: Various reward component metrics
    """

    def split_response(response_token_ids: List[int]) -> List[int]:
        last_index = len(response_token_ids)
        step_boundaries = [0]
        cursor = 0
        while cursor < last_index:
            cursor += 100
            if cursor >= last_index:
                break
            step_boundaries.append(cursor)

        return step_boundaries

    def estimate_values_by_mc_rollouts(episodes_raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        def get_response_prefixes(
            response_token_ids: List[int], states_for_value_estimation: List[int]
        ) -> List[List[int]]:
            prefixes = []
            for state in states_for_value_estimation:
                prefixes.append(response_token_ids[:state])
            return prefixes

        def get_mc_queries(
            query_token_ids: List[int], response_prefixes_token_ids: List[List[int]]
        ) -> Tuple[List[List[int]], List[int]]:
            prefix_queries_token_ids = []
            max_tokens = []
            for token_ids in response_prefixes_token_ids:
                prefix_queries_token_ids.append(query_token_ids + token_ids)
                max_tokens.append(MAX_RESPONSE_TOKENS - len(token_ids))  # this is how many response tokens are left.
            return prefix_queries_token_ids, max_tokens

        def get_value_estimates(
            sample: Dict[str, Any], response_prefixes: List[List[int]], mcs_token_ids: List[List[int]]
        ) -> List[float]:
            values_estimates = []
            for prefix, mcs in zip(response_prefixes, mcs_token_ids):
                values = []
                for mc in mcs:
                    full_text = tokenizer.decode(prefix + mc, skip_special_tokens=False)
                    score, _ = compute_reward(full_text, sample, EOS_TOKEN)
                    values.append(score)
                values_estimates.append(sum(values) / len(values))
            return values_estimates

        def update_state_values(
            old_states: List[int],
            old_value_estimates: List[float],
            new_states: List[int],
            new_value_estimates: List[float],
        ) -> Tuple[List[int], List[float]]:
            values = {}
            for state, value_estimate in zip(old_states, old_value_estimates):
                values[state] = value_estimate
            for state, value_estimate in zip(new_states, new_value_estimates):
                assert state not in values
                values[state] = value_estimate
            sorted_states = sorted(values.keys())
            sorted_values = [values[state] for state in sorted_states]
            return sorted_states, sorted_values

        def extract_token_ids(vllm_outputs: List[CompletionOutput]) -> List[List[int]]:
            return [list(out.token_ids) for out in vllm_outputs]

        for eps in episodes_raw:
            eps["value_estimates"] = [eps["reward"]]
            eps["states"] = [len(eps["response_token_ids"])]
            eps["new_states"] = split_response(eps["response_token_ids"])

            # Get prefixes of from chunks of the response
            # i.e. for response [A, B, C, D, E] with new_states [0, 2, 4]
            # we get prefixes [[]], [A, B], [A, B, C, D]
            eps["mc_response_prefixes_token_ids"] = get_response_prefixes(eps["response_token_ids"], eps["new_states"])

            # Get MC queries where we just add the query to each prefix
            eps["mc_queries_token_ids"], eps["mc_queries_max_tokens"] = get_mc_queries(
                eps["query_token_ids"], eps["mc_response_prefixes_token_ids"]
            )

        # Flatten the MC queries to a single list which will be used for inference
        flatten_mc_queries = []
        flatten_mc_queries_max_tokens = []
        queries_count = []
        for eps in episodes_raw:
            flatten_mc_queries.extend(eps["mc_queries_token_ids"])
            flatten_mc_queries_max_tokens.extend(eps["mc_queries_max_tokens"])
            queries_count.append(len(eps["mc_queries_token_ids"]))

        # Auxiliary rollouts to get the value estimates
        logger.info("Monte-Carlo value estimation...")
        mc_outputs: List[RequestOutput] = inference_engine.generate(
            prompt_token_ids=flatten_mc_queries,
            sampling_params=[
                SamplingParams(
                    n=VINEPPO_K,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    top_k=TOP_K,
                    max_tokens=max_tokens,
                    detokenize=False,
                    stop_token_ids=[EOS_TOKEN_ID],
                )
                for max_tokens in flatten_mc_queries_max_tokens
            ],
        )

        # Unflatten the MC rollouts
        # [
        #   // Episode 1
        #   [
        #     [tok1, tok2, ...],
        #     [tok1, tok2, ...],
        #     ...
        #   ],
        #   ...
        # ]
        unflattened_mc_token_ids: List[List[List[int]]] = []
        start = 0
        for count in queries_count:
            output_slice = mc_outputs[start : start + count]
            unflattened_mc_token_ids.append([extract_token_ids(out.outputs) for out in output_slice])
            start += count

        assert len(unflattened_mc_token_ids) == len(episodes_raw)

        # Compute the value estimates based on avg. MC returns
        for i, eps in enumerate(episodes_raw):
            mc_token_ids = unflattened_mc_token_ids[i]
            mc_value_estimates = get_value_estimates(
                sample=eps["sample"],
                response_prefixes=eps["mc_response_prefixes_token_ids"],
                mcs_token_ids=mc_token_ids,
            )
            eps["states"], eps["value_estimates"] = update_state_values(
                old_states=eps["states"],
                old_value_estimates=eps["value_estimates"],
                new_states=eps["new_states"],
                new_value_estimates=mc_value_estimates,
            )

        # Remove unnecessary keys
        for i, eps in enumerate(episodes_raw):
            eps.pop("new_states")
            eps.pop("mc_response_prefixes_token_ids")
            eps.pop("mc_queries_token_ids")
            eps.pop("mc_queries_max_tokens")

        return episodes_raw

    def get_tokens_advantages(states: List[int], value_estimates: List[float]) -> List[float]:
        tokens_advantages = []
        assert sorted(states) == states
        for i in range(len(states) - 1):
            length = states[i + 1] - states[i]
            advantage = value_estimates[i + 1] - value_estimates[i]
            tokens_advantages.extend([advantage] * length)
        return tokens_advantages

    assert len(all_generations) == len(all_finish_reasons)
    assert len(all_generations) == len(samples) * GENERATIONS_PER_SAMPLE

    # Process responses and calculate rewards
    groups = [
        list(range(i, i + GENERATIONS_PER_SAMPLE)) for i in range(0, len(all_generations), GENERATIONS_PER_SAMPLE)
    ]  # example: [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    all_query_token_ids, all_responses_token_ids, all_samples, all_rewards = [], [], [], []

    stats = {
        "response_lengths": [],
        "rewards": [],
        "non_stop_rate": [],
    }

    for sample, group_indices in zip(samples, groups):
        finish_reasons = [all_finish_reasons[i] for i in group_indices]
        response_token_ids = [all_generations[i] for i in group_indices]
        responses = tokenizer.batch_decode(response_token_ids, skip_special_tokens=False)

        rewards_and_metrics = [compute_reward(resp, sample, EOS_TOKEN) for resp in responses]
        rewards, reward_metrics = zip(*rewards_and_metrics)

        all_rewards.extend(rewards)
        all_samples.extend([sample] * GENERATIONS_PER_SAMPLE)
        all_query_token_ids.extend([sample["input_ids"]] * GENERATIONS_PER_SAMPLE)
        all_responses_token_ids.extend(response_token_ids)

        stats["rewards"].extend(rewards)
        stats["non_stop_rate"].extend([fr != "stop" for fr in finish_reasons])
        stats["response_lengths"].extend([len(ids) for ids in response_token_ids])
        for rm in reward_metrics:
            for k, v in rm.items():
                stats.setdefault(f"reward_metrics/{k}", []).append(v)

    raw_episodes: List[Dict[str, Any]] = []
    for i in range(len(all_samples)):
        eps = {
            "query_token_ids": all_query_token_ids[i],
            "response_token_ids": all_responses_token_ids[i],
            "sample": all_samples[i],
            "reward": all_rewards[i],
            "states": [len(all_responses_token_ids[i])],
            "value_estimates": [all_rewards[i]],
            "new_states": split_response(all_responses_token_ids[i]),
        }
        raw_episodes.append(eps)

    raw_episodes = estimate_values_by_mc_rollouts(raw_episodes)

    all_advantages = []
    for eps in raw_episodes:
        all_advantages.append(get_tokens_advantages(eps["states"], eps["value_estimates"]))

    episodes = {
        "all_query_token_ids": all_query_token_ids,
        "all_response_token_ids": all_responses_token_ids,
        "all_advantages": all_advantages,
    }

    return episodes, stats


def compute_pg_loss(
    policy_model: Union[DeepSpeedEngine, PreTrainedModel],
    batch: Dict[str, torch.Tensor],
    total_response_len: torch.Tensor,
    TEMPERATURE: float,
    KL_COEFFICIENT: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the policy gradient loss with KL penalty between policy and reference models.

    This function:
    1. Calculates KL divergence penalty between the models
    2. Computes policy gradient loss using advantages
    3. Combines the losses with KL coefficient

    Args:
        policy_model: The model being trained
        batch: Dictionary containing:
            - input_ids: Tensor of shape [batch_size, seq_len]
            - attention_mask: Tensor of shape [batch_size, seq_len]
            - labels: Tensor of shape [batch_size, seq_len] with -100 for ignored positions
            - advantages: Tensor of shape [batch_size, seq_len]
            - ref_log_probs: Tensor of shape [batch_size, seq_len-1]
        total_response_len: Total number of valid tokens in the batch. This is a scalar tensor.

    Returns:
        Tuple containing:
            - loss: Combined policy gradient and KL penalty loss (scalar tensor)
            - metrics: Dictionary with detailed loss components:
                - policy_loss: Pure policy gradient loss
                - kl_penalty: KL divergence penalty
                - entropy: Policy entropy
    """
    input_ids = batch["input_ids"]  # [batch_size, seq_len]
    attention_mask = batch["attention_mask"]  # [batch_size, seq_len]
    labels = batch["labels"]  # [batch_size, seq_len]
    labels_mask = batch["labels_mask"]  # [batch_size, seq_len]
    advantages = batch["advantages"]  # [batch_size, seq_len]
    ref_logps = batch["ref_log_probs"]  # [batch_size, seq_len-1]

    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "labels_mask": labels_mask,
    }

    logps = compute_token_log_probs(policy_model, model_inputs, TEMPERATURE)  # [batch_size, seq_len-1]
    labels_mask = labels_mask[..., 1:].to(logps.dtype)  # [batch_size, seq_len-1]

    ref_logratio = ref_logps - logps
    kl_penalty = torch.exp(ref_logratio) - 1 - ref_logratio  # [batch_size, seq_len-1]
    kl_penalty = kl_penalty * labels_mask  # [batch_size, seq_len-1]

    with torch.no_grad():
        entropy = -logps.sum() / labels_mask.sum()  # scalar
        zero_advantages = close_to_zero(advantages[..., 1:], labels_mask)  # scalar

    policy_loss = -logps * advantages[..., 1:]  # [batch_size, seq_len-1]
    policy_loss = policy_loss * labels_mask  # [batch_size, seq_len-1]

    loss = (policy_loss + KL_COEFFICIENT * kl_penalty).sum() / total_response_len  # scalar

    metrics = {
        "policy_loss": policy_loss.sum().item() / total_response_len.item(),
        "kl_penalty": kl_penalty.sum().item() / total_response_len.item(),
        "entropy": entropy.item() / total_response_len.item(),
        "zero_advantages_ratio": zero_advantages.item() / total_response_len.item(),
    }

    return loss, metrics


def main(rank: int):
    # Parse command line arguments
    args = arg_parser.parse_args()

    # rank = int(os.environ.get("RANK", "0"))
    nproc = int(os.environ.get("WORLD_SIZE", "1"))
    nproc = args.nproc
    initialize_training_process_group(rank, nproc)
    curr_cuda_device = torch.device("cuda")

    # Disable logging for non-main processes to avoid duplicate logs
    if dist.get_rank() != 0:
        logger.setLevel(logging.ERROR)

    if args.debug and nproc == 1:
        import debugpy

        debugpy.listen(5678)
        logger.info("Waiting for debugger to attach...")
        debugpy.wait_for_client()
        logger.info("Debugger attached")

    ############################################
    # Hyperparameters
    ############################################

    # Model configuration
    MODEL_NAME = args.model_name

    # RL parameters
    # Total number of training iterations
    NUM_ITERATIONS = 1000
    # Number of episodes to collect per iteration for training
    EPISODES_PER_ITERATION = 64
    EPISODES_PER_ITERATION_PER_RANK = EPISODES_PER_ITERATION // dist.get_world_size()
    # Number of responses to generate for each input prompt
    GENERATIONS_PER_SAMPLE = 4
    # Controls how much the policy can deviate from the reference model
    KL_COEFFICIENT = args.kl_coeff

    # Training hyperparameters
    # Batch size for each GPU device during training
    PER_DEVICE_BATCH_SIZE = args.per_device_batch_size
    # Learning rate for model updates
    LEARNING_RATE = 1e-6

    # Sampling parameters
    # Maximum number of tokens to generate in each response
    MAX_RESPONSE_TOKENS = args.max_response_tokens
    # Controls randomness in generation (higher = more random)
    TEMPERATURE = args.temperature
    # Nucleus sampling parameter (1.0 = disabled)
    TOP_P = 0.999  # to avoid sampling unused tokens absent from tokenizer see https://github.com/vllm-project/vllm/issues/13175#issuecomment-2781842571
    # Top-k sampling parameter (-1 = disabled)
    TOP_K = -1  # no top k
    # Number of MC samples to take for each response
    VINEPPO_K = args.vineppo_k
    # DeepSpeed configuration
    deepspeed_config = {
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": 2, "overlap_comm": False},
        "train_batch_size": EPISODES_PER_ITERATION,
        "train_micro_batch_size_per_gpu": PER_DEVICE_BATCH_SIZE,
        "gradient_accumulation_steps": EPISODES_PER_ITERATION_PER_RANK // PER_DEVICE_BATCH_SIZE,
        "gradient_clipping": 1.0,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.learning_rate,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.0,
                "torch_adam": True,
                "fused": True,
            },
        },
    }
    ref_deepspeed_config = {
        "bf16": {"enabled": True},
        # No effect
        "train_batch_size": EPISODES_PER_ITERATION,
        "train_micro_batch_size_per_gpu": PER_DEVICE_BATCH_SIZE,
        "gradient_accumulation_steps": EPISODES_PER_ITERATION_PER_RANK // PER_DEVICE_BATCH_SIZE,
    }

    dist.barrier(device_ids=[torch.cuda.current_device()])

    model_name_short = MODEL_NAME.split("/")[-1]
    if args.run_id is None:
        RUN_NAME = f"{model_name_short}_temp{TEMPERATURE}_kl{KL_COEFFICIENT}_lr{LEARNING_RATE}_al{args.algorithm}"
    else:
        RUN_NAME = args.run_id
    EXP_DIR = Path.home() / "scratch" / "nano_aha_moment" / RUN_NAME
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Logs and Checkpoints will be saved to: {EXP_DIR}")

    ############################################
    # Prompts and Dataset
    ############################################

    SYSTEM_MESSAGE = (
        "You are a helpful assistant. You first think about the reasoning process in the mind "
        "and then provide the user with the answer."
    )
    PROMPT_TEMPLATE = (
        "Using the numbers {numbers}, create an equation that equals {target}. "
        "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
        "Show your work in <think> </think> tags. And return the final equation and answer in "
        "<answer> </answer> tags, for example <answer>(1 + 2) / (3 * 5)</answer>."
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    EOS_TOKEN_ID = tokenizer.eos_token_id
    EOS_TOKEN = tokenizer.convert_ids_to_tokens(EOS_TOKEN_ID)

    dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    # Rank 0 will preprocess the dataset first
    if dist.get_rank() != 0:
        # Other ranks will wait for rank 0 to enter the barrier
        dist.barrier(device_ids=[torch.cuda.current_device()])
    dataset = dataset.map(
        preprocess_example,
        num_proc=6,
        fn_kwargs={
            "tokenizer": tokenizer,
            "SYSTEM_MESSAGE": SYSTEM_MESSAGE,
            "PROMPT_TEMPLATE": PROMPT_TEMPLATE,
        },
        desc="Preprocessing dataset",
    )
    if dist.get_rank() == 0:
        # Rank 0 will enter the barrier so that other ranks can start preprocessing
        dist.barrier(device_ids=[torch.cuda.current_device()])
    dist.barrier(device_ids=[torch.cuda.current_device()])

    # Split dataset
    train_test_split = dataset.train_test_split(test_size=500, seed=42)
    train_dataset = train_test_split["train"]
    orig_train_dataset_size = len(train_dataset)
    test_dataset = train_test_split["test"]

    # Shard the training dataset
    train_dataset = train_dataset.shard(num_shards=dist.get_world_size(), index=dist.get_rank())

    logger.info(f"Train dataset size: {orig_train_dataset_size}; each rank will process {len(train_dataset)} samples")
    logger.info(f"Test dataset size: {len(test_dataset)}")

    ############################################
    # Initialize Models
    ############################################

    policy_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map=torch.cuda.current_device(),
    )
    reference_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map=torch.cuda.current_device(),
    )
    policy_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    see_memory_usage("Before initializing DeepSpeed engines", force=dist.get_rank() == 0)

    # Initialize DeepSpeed engines
    policy_model, *_ = deepspeed.initialize(
        model=policy_model,
        config=deepspeed_config,
        model_parameters=policy_model.parameters(),
    )
    reference_model, *_ = deepspeed.initialize(
        model=reference_model,
        config=ref_deepspeed_config,
    )

    reference_model.module.cpu()
    dist.barrier(device_ids=[torch.cuda.current_device()])

    ############################################
    # Initialize vLLM (Inference) engine
    ############################################

    see_memory_usage("Before initializing inference engine", force=dist.get_rank() == 0)

    if dist.get_rank() != 0:
        # Disable root vllm logger for non-main ranks
        vllm_logger = logging.getLogger("vllm")
        vllm_logger.setLevel(logging.ERROR)

    inference_engine = LLM(
        model=MODEL_NAME,
        skip_tokenizer_init=False,
        gpu_memory_utilization=0.3,
        enable_prefix_caching=True,
        swap_space=4,
        scheduling_policy="fcfs",
        dtype=torch.bfloat16,
        max_model_len=MAX_RESPONSE_TOKENS + 1024,
        enable_sleep_mode=True,
        device=f"cuda:{torch.cuda.current_device()}",
        tensor_parallel_size=1,
    )
    if args.algorithm == "vineppo":
        logits_processors = [fix_oov_logits_processor(inference_engine)]
    else:
        logits_processors = None

    see_memory_usage("After initializing inference engine", force=dist.get_rank() == 0)

    # Wandb for logging. Only rank 0 will initialize wandb
    if dist.get_rank() == 0:
        wandb.init(
            project="nano-aha-moment",
            name=RUN_NAME,
            resume="allow",
            config={
                "model_name": MODEL_NAME,
                "learning_rate": LEARNING_RATE,
                "num_iterations": NUM_ITERATIONS,
                "episodes_per_iteration": EPISODES_PER_ITERATION,
                "rollouts_per_episode": GENERATIONS_PER_SAMPLE,
                "kl_coefficient": KL_COEFFICIENT,
                "temperature": TEMPERATURE,
                "algorithm": args.algorithm,
                "vineppo_k": VINEPPO_K,
            },
        )

    sampler_rng = np.random.default_rng(seed=42)
    NUM_SAMPLES_PER_ITERATION = EPISODES_PER_ITERATION_PER_RANK // GENERATIONS_PER_SAMPLE

    # Load checkpoint if it exists
    begin_iter = 0
    ckpt_path, ckpt_iter = find_last_checkpoint(EXP_DIR)
    if ckpt_path is not None:
        logger.info(f"Resuming from checkpoint {ckpt_path} at iteration {ckpt_iter}")
        out = policy_model.load_checkpoint(ckpt_path / "deepspeed")
        if out is None:
            raise RuntimeError(f"Failed to load checkpoint {ckpt_path}")
        begin_iter = ckpt_iter + 1
        load_model_into_vllm(policy_model, inference_engine)

        logger.info(f"Skipping {ckpt_iter} rounds of samples")
        for _ in trange(ckpt_iter, disable=dist.get_rank() != 0):
            _ = sampler_rng.choice(len(train_dataset), size=NUM_SAMPLES_PER_ITERATION, replace=False)

    for iteration in trange(begin_iter, NUM_ITERATIONS):
        logger.info(f"Iteration {iteration}/{NUM_ITERATIONS}")

        metrics = {}

        #########################################################
        # Evaluation
        #########################################################

        eval_stats = None
        if iteration % 25 == 0 and iteration > 0 and dist.get_rank() == 0:  # Only rank 0 will evaluate:
            logger.info("Evaluating on eval set...")
            eval_episodes, eval_stats = evaluate_on_test_set(
                inference_engine=inference_engine,
                test_dataset=test_dataset,
                tokenizer=tokenizer,
                eos_token=EOS_TOKEN,
                eval_sampling_params=SamplingParams(
                    temperature=0.3,
                    max_tokens=MAX_RESPONSE_TOKENS,
                    n=1,
                    detokenize=False,
                    stop_token_ids=[EOS_TOKEN_ID],
                ),
                reward_func=lambda completion, sample: compute_reward(completion, sample, EOS_TOKEN),
            )
            eval_episode_table = dump_episodes(
                episodes=eval_episodes,
                episodes_stats=eval_stats,
                exp_dir=EXP_DIR,
                tokenizer=tokenizer,
                iteration=iteration,
                is_eval=True,
            )
            wandb.log({"eval/episodes": eval_episode_table, "iteration": iteration})
        dist.barrier(device_ids=[torch.cuda.current_device()])

        #########################################################
        # Generate Episodes
        #########################################################

        # Sample training batch
        indices = sampler_rng.choice(len(train_dataset), size=NUM_SAMPLES_PER_ITERATION, replace=False)
        samples = train_dataset.select(indices)

        gen_time = time.time()

        # Sample responses
        outputs = inference_engine.generate(
            prompt_token_ids=samples["input_ids"],
            sampling_params=SamplingParams(
                n=GENERATIONS_PER_SAMPLE,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                max_tokens=MAX_RESPONSE_TOKENS,
                detokenize=False,
                stop_token_ids=[EOS_TOKEN_ID],
                logits_processors=logits_processors,
            ),
        )
        all_generations = [list(g.token_ids) for out in outputs for g in out.outputs]
        all_finish_reasons = [g.finish_reason for out in outputs for g in out.outputs]

        logger.info(f"Generated {len(all_generations)} responses")
        logger.info(f"Time taken to generate {len(all_generations)} responses: {time.time() - gen_time} seconds")

        # Process responses and calculate rewards
        if args.algorithm == "grpo":
            episodes, episodes_stats = create_training_episodes(
                samples=samples,
                all_generations=all_generations,
                all_finish_reasons=all_finish_reasons,
                tokenizer=tokenizer,
                EOS_TOKEN=EOS_TOKEN,
                GENERATIONS_PER_SAMPLE=GENERATIONS_PER_SAMPLE,
            )
        elif args.algorithm == "vineppo":
            episodes, episodes_stats = create_vineppo_training_episodes(
                inference_engine=inference_engine,
                samples=samples,
                all_generations=all_generations,
                all_finish_reasons=all_finish_reasons,
                tokenizer=tokenizer,
                EOS_TOKEN_ID=EOS_TOKEN_ID,
                EOS_TOKEN=EOS_TOKEN,
                GENERATIONS_PER_SAMPLE=GENERATIONS_PER_SAMPLE,
                MAX_RESPONSE_TOKENS=MAX_RESPONSE_TOKENS,
                VINEPPO_K=VINEPPO_K,
                TEMPERATURE=TEMPERATURE,
                TOP_P=TOP_P,
                TOP_K=TOP_K,
            )
        else:
            raise ValueError(f"Invalid algorithm: {args.algorithm}")

        inference_engine.sleep(1)
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

        for k, v in episodes_stats.items():
            metrics.setdefault(k, []).extend(v)

        episode_table = dump_episodes(
            episodes=episodes,
            episodes_stats=episodes_stats,
            exp_dir=EXP_DIR,
            tokenizer=tokenizer,
            iteration=iteration,
            do_save=iteration % 10 == 0 or iteration == 0,
        )

        #########################################################
        # Training
        #########################################################

        # Prepare training batch
        model_inputs = prepare_model_inputs(
            query_token_ids=episodes["all_query_token_ids"],
            response_token_ids=episodes["all_response_token_ids"],
            advantages=episodes["all_advantages"],
            device=curr_cuda_device,
        )

        logger.info("Moving reference model to GPU")
        reference_model.module.to(curr_cuda_device)
        reference_model.eval()

        with torch.no_grad():
            ref_log_probs = []
            for i in trange(
                0,
                EPISODES_PER_ITERATION_PER_RANK,
                PER_DEVICE_BATCH_SIZE,
                desc="Computing reference logprobs",
                disable=dist.get_rank() != 0,
            ):
                batch = {k: v[i : i + PER_DEVICE_BATCH_SIZE] for k, v in model_inputs.items()}
                ref_log_probs.append(
                    compute_token_log_probs(
                        model=reference_model,
                        inputs=batch,
                        temperature=TEMPERATURE,
                    )
                )
            ref_log_probs = torch.cat(ref_log_probs)
            model_inputs["ref_log_probs"] = ref_log_probs
            del ref_log_probs

        # Free memory taken by reference model
        logger.info("Moving reference model back to CPU")
        reference_model.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

        # Calculate losses and update model
        policy_model.train()
        total_response_len = (model_inputs["labels"] != -100).sum()
        train_time = time.time()

        for i in trange(
            0,
            EPISODES_PER_ITERATION_PER_RANK,
            PER_DEVICE_BATCH_SIZE,
            desc="Gradient Accumulation",
            disable=dist.get_rank() != 0,
        ):
            batch = {k: v[i : i + PER_DEVICE_BATCH_SIZE] for k, v in model_inputs.items()}

            # Compute policy gradient loss
            loss, loss_metrics = compute_pg_loss(
                policy_model=policy_model,
                batch=batch,
                total_response_len=total_response_len,
                TEMPERATURE=TEMPERATURE,
                KL_COEFFICIENT=KL_COEFFICIENT,
            )

            # Track metrics
            metrics.setdefault("loss", []).append(loss.item())
            grad_norm = policy_model.get_global_grad_norm()
            if grad_norm is not None:
                grad_norm = grad_norm.item()
            metrics.setdefault("grad_norm", []).append(grad_norm)
            for k, v in loss_metrics.items():
                metrics.setdefault(k, []).append(v.item() if isinstance(v, torch.Tensor) else v)

            # Backpropagation and optimization step
            # scale_wrt_gas=False because we are already normalizing by total_response_len
            policy_model.backward(loss, scale_wrt_gas=False)
            del loss, loss_metrics

            policy_model.step()

        logger.info(f"Time taken to train: {time.time() - train_time} seconds")

        #########################################################
        # Update inference engine weights
        #########################################################

        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

        inference_engine.wake_up()
        load_model_into_vllm(policy_model, inference_engine)

        #########################################################
        # Log metrics
        #########################################################

        if dist.get_rank() == 0:
            train_metrics = {k: np.mean(v) for k, v in metrics.items() if None not in v}
            train_metrics["learning_rate"] = policy_model.get_lr()[0]
            logs = {
                "iteration": iteration,
                f"episodes/iter_{iteration:06d}": episode_table,
                **{f"train/{k}": v for k, v in train_metrics.items()},
            }
            if eval_stats is not None:
                logs.update({f"eval/{k}": np.mean(v) for k, v in eval_stats.items()})
            wandb.log(logs)

            selected_keys = [
                "train/kl_penalty",
                "train/rewards",
                "train/reward_metrics/format_reward",
                "train/reward_metrics/equation_reward",
                "train/response_lengths",
                "eval/rewards",
                "eval/reward_metrics/format_reward",
                "eval/reward_metrics/equation_reward",
            ]
            selected_metrics = {k: float(logs[k]) for k in selected_keys if k in logs}
            logger.info(f"KEY METRICS: {selected_metrics}")

        if iteration % 50 == 0 and iteration != 0:
            ckpt_dir = EXP_DIR / "checkpoints" / f"ckpt_{iteration:06d}"

            logger.info("Saving HF model")
            if dist.get_rank() == 0:
                policy_model.module.save_pretrained(str(ckpt_dir / "hf_model"))
                tokenizer.save_pretrained(str(ckpt_dir / "hf_model"))
            dist.barrier(device_ids=[torch.cuda.current_device()])

            logger.info("Saving DeepSpeed checkpoint")
            policy_model.save_checkpoint(str(ckpt_dir / "deepspeed"))

            if dist.get_rank() == 0:
                clean_up_checkpoints(
                    exp_dir=EXP_DIR,
                    keep_every_n_steps=50,
                    exclude=[ckpt_dir],
                )
            dist.barrier(device_ids=[torch.cuda.current_device()])

    dist.destroy_process_group()


if __name__ == "__main__":
    args = arg_parser.parse_args()

    n_gpus = torch.cuda.device_count()
    if args.nproc > n_gpus:
        raise ValueError(f"Requested {args.nproc} processes, but only {n_gpus} GPUs are available.")

    if args.nproc == 1:
        main(rank=0)
    else:
        torch.multiprocessing.spawn(main, nprocs=args.nproc)
