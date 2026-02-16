from datetime import timedelta
import json
import os
import shutil
import socket
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import wandb
from datasets import Dataset
from deepspeed import DeepSpeedEngine
from transformers import AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams

DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer."
DEFAULT_PROMPT_TEMPLATE = "Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer>(1 + 2) / (3 * 5)</answer>."


def create_prompt(
    numbers: List[int],
    target: int,
    tokenizer: AutoTokenizer,
    system_message: str = DEFAULT_SYSTEM_MESSAGE,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
) -> str:
    prefix = [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": prompt_template.format(numbers=numbers, target=target),
        },
        {"role": "assistant", "content": "Let me solve this step by step.\n<think>"},
    ]
    return tokenizer.apply_chat_template(prefix, tokenize=False, continue_final_message=True)


def prepare_model_inputs(
    query_token_ids: List[List[int]],
    response_token_ids: List[List[int]],
    advantages: List[List[float]],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Prepare padded model inputs with attention masks, labels, and advantages.
    Args:
        query_token_ids: List of query token ids
        response_token_ids: List of response token ids
        advantages: List of lists of advantage values, matching response_token_ids structure
        device: Device to move the tensors to
    Returns:
        Dict with input_ids, attention_mask, labels, and advantages

    Example:
        >>> query_token_ids = [[1, 2, 3], [4, 5]]
        >>> response_token_ids = [[6, 7], [8]]
        >>> advantages = [[0.5, 0.8], [0.3]]
        >>> outputs = prepare_model_inputs(query_token_ids, response_token_ids, advantages, "cuda")
        >>> outputs
        {
            'input_ids': tensor([
                [1, 2, 3, 6, 7],
                [4, 5, 8, 0, 0]
            ]),
            'attention_mask': tensor([
                [1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0]
            ]),
            'labels': tensor([
                [-100, -100, -100, 6, 7],
                [-100, -100, 8, -100, -100]
            ]),
            'advantages': tensor([
                [0.0, 0.0, 0.0, 0.5, 0.5],
                [0.0, 0.0, 0.0, 0.9, 0.0]
            ])
        }
    """
    max_seq_len = max(len(q) + len(r) for q, r in zip(query_token_ids, response_token_ids))
    inputs = {"input_ids": [], "attention_mask": [], "labels": [], "advantages": [], "labels_mask": []}

    pad_token_id = 0  # Doesn't matter, will be masked
    ignore_index = -100

    for query, response, advantage in zip(query_token_ids, response_token_ids, advantages):
        combined_ids = query + response
        seq_len = len(combined_ids)

        # Create padded sequences
        input_ids = combined_ids + [pad_token_id] * (max_seq_len - seq_len)
        attention_mask = [1] * seq_len + [0] * (max_seq_len - seq_len)
        labels = [ignore_index] * len(query) + response + [ignore_index] * (max_seq_len - seq_len)
        labels_mask = [0] * len(query) + [1] * len(response) + [0] * (max_seq_len - seq_len)
        advantages_seq = [0.0] * len(query) + advantage + [0.0] * (max_seq_len - seq_len)

        assert len(input_ids) == max_seq_len
        assert len(attention_mask) == max_seq_len
        assert len(labels) == max_seq_len
        assert len(advantages_seq) == max_seq_len
        assert len(labels_mask) == max_seq_len

        inputs["input_ids"].append(input_ids)
        inputs["attention_mask"].append(attention_mask)
        inputs["labels"].append(labels)
        inputs["advantages"].append(advantages_seq)
        inputs["labels_mask"].append(labels_mask)

    # Convert to tensors
    return {
        k: torch.tensor(v, dtype=torch.long if k != "advantages" else torch.float, device=device)
        for k, v in inputs.items()
    }


@torch.compile(dynamic=True)
def log_softmax_and_gather(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Copied from https://github.com/allenai/open-instruct/blob/main/open_instruct/model_utils.py#L425

    torch compiled version of the common `log_softmax -> gather` operation.

    The compiled version of this opration avoids the (significant) memory overhead of
    allocating a new (batch_size, seq_len, vocab_size) tensor to store the logprobs.

    Args:
        logits: Tensor of shape (batch_size, seq_len, vocab_size) containing the logits
        index: Tensor of shape (batch_size, seq_len) containing the indices to gather

    Returns:
        Tensor of shape (batch_size, seq_len) containing the log probabilities for the
        specified indices

    See https://github.com/allenai/open-instruct/pull/584
    """
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)


def compute_token_log_probs(
    model: Union[DeepSpeedEngine, PreTrainedModel],
    inputs: Dict[str, torch.Tensor],
    temperature: float,
) -> torch.Tensor:
    """
    Compute log probabilities for each token in the sequence, masked for valid labels only.

    This function:
    1. Runs the model forward pass
    2. Applies temperature scaling to logits
    3. Shifts the sequences for causal language modeling
    4. Computes log probabilities for the actual tokens that appeared in the sequence
    5. Masks the log probabilities to only include valid labels (non -100 positions)

    Args:
        model: The language model (either DeepSpeed-wrapped or regular HuggingFace model)
        inputs: Dictionary containing:
            - input_ids: Tensor of token ids [batch_size, seq_len]
            - attention_mask: Tensor of attention mask [batch_size, seq_len]
            - labels: Tensor of target labels [batch_size, seq_len] with -100 for ignored positions
        temperature: Temperature for scaling the logits before softmax

    Returns:
        torch.Tensor: Log probabilities tensor of shape [batch_size, seq_len-1], where:
            - Each value is the log probability of the actual token that appeared
            - Values are masked to 0.0 for positions where labels were -100
            - The sequence length is reduced by 1 due to the causal shift

    Example:
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> inputs = {
        ...     "input_ids": torch.tensor([[1, 2, 3]]),
        ...     "attention_mask": torch.tensor([[1, 1, 1]]),
        ...     "labels": torch.tensor([[-100, 2, 3]])
        ... }
        >>> log_probs = compute_token_log_probs(model, inputs, temperature=1.0)
        >>> log_probs.shape
        torch.Size([1, 2])  # batch_size=1, seq_len-1=2
        >>> # First position is 0 (masked), second position has actual log prob
    """
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        return_dict=True,
        use_cache=False,
    )

    logits = outputs.logits / temperature  # Shape: [batch_size, seq_len, vocab_size]
    shift_logits = logits[..., :-1, :]  # Shape: [batch_size, seq_len-1, vocab_size]
    shift_labels = inputs["labels"][..., 1:]  # Shape: [batch_size, seq_len-1]
    shift_labels_mask = inputs["labels_mask"][..., 1:]  # Shape: [batch_size, seq_len-1]

    # Create mask for valid labels
    shift_labels[~(shift_labels_mask.bool())] = 0  # Shape: [batch_size, seq_len-1]

    # Calculate log probabilities
    log_probs = log_softmax_and_gather(shift_logits, shift_labels)  # Shape: [batch_size, seq_len-1]
    log_probs = log_probs * shift_labels_mask  # Shape: [batch_size, seq_len-1]

    return log_probs


def find_free_port():
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def evaluate_on_test_set(
    inference_engine: LLM,
    test_dataset: Dataset,
    tokenizer: AutoTokenizer,
    eos_token: str,
    eval_sampling_params: SamplingParams,
    reward_func: Callable[[str, Dict[str, Any]], Tuple[float, Dict[str, float]]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Evaluate the model on a test dataset by generating responses and computing rewards.

    Args:
        inference_engine: The sglang Engine instance used for text generation
        test_dataset: Dataset containing test samples
        tokenizer: Tokenizer for decoding generated token IDs
        eos_token: End of sequence token string
        eval_sampling_params: Dictionary of parameters for controlling the generation process
        reward_func: Function that computes rewards for generated responses. Takes a response
            string and sample dict as input, returns a tuple of (overall_reward, reward_components)

    Returns:
        Dictionary containing evaluation statistics:
            - response_lengths: List of token counts for each generated response
            - rewards: List of overall reward values for each response
            - non_stop_rate: List of booleans indicating if generation ended for non-stop reason
            - reward_metrics/*: Lists of individual reward component values, prefixed with
              "reward_metrics/"
        episodes: Dictionary containing:
            - all_query_token_ids: List of query token IDs for each episode
            - all_response_token_ids: List of response token IDs for each episode

    Example:
        >>> episodes, episodes_stats = evaluate_on_test_set(
        ...     inference_engine=engine,
        ...     test_dataset=dataset,
        ...     tokenizer=tokenizer,
        ...     eos_token="</s>",
        ...     eval_sampling_params={"temperature": 0.7, "max_tokens": 100},
        ...     reward_func=compute_rewards
        ... )
        >>> print(f"Average reward: {episodes_stats['rewards']:.3f}")
    """
    generations = inference_engine.generate(
        prompt_token_ids=test_dataset["input_ids"], sampling_params=eval_sampling_params
    )

    metrics = {
        "response_lengths": [],
        "rewards": [],
        "non_stop_rate": [],
    }

    all_query_token_ids = []
    all_responses_token_ids = []

    for i, sample in enumerate(test_dataset):
        query_token_ids = sample["input_ids"]
        response_token_ids = generations[i].outputs[0].token_ids
        finish_reason = generations[i].outputs[0].finish_reason

        response = tokenizer.decode(response_token_ids, skip_special_tokens=False)
        reward, reward_components = reward_func(response, sample)

        all_query_token_ids.append(query_token_ids)
        all_responses_token_ids.append(response_token_ids)

        metrics["rewards"].append(reward)
        metrics["non_stop_rate"].append(finish_reason != "stop")
        metrics["response_lengths"].append(len(response_token_ids))
        for k, v in reward_components.items():
            metrics.setdefault(f"reward_metrics/{k}", []).append(v)

    episodes = {
        "all_query_token_ids": all_query_token_ids,
        "all_response_token_ids": all_responses_token_ids,
    }

    return episodes, metrics


def dump_episodes(
    episodes: Dict[str, Any],
    episodes_stats: Dict[str, Any],
    exp_dir: Path,
    tokenizer: AutoTokenizer,
    iteration: int,
    is_eval: bool = False,
    do_save: bool = True,
) -> wandb.Table:
    query_token_ids = episodes["all_query_token_ids"]
    response_token_ids = episodes["all_response_token_ids"]
    rewards = episodes_stats["rewards"]
    response_lengths = episodes_stats["response_lengths"]

    query_texts = tokenizer.batch_decode(
        query_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    response_texts = tokenizer.batch_decode(
        response_token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    if not is_eval and rank == 0:
        print(f"########## Example 1 (Reward: {rewards[0]}, Response Length: {response_lengths[0]})")
        print(f"#### Query:\n`{query_texts[0]}`")
        print(f"#### Response:\n`{response_texts[0]}`\n\n")

        print(f"########## Example 2 (Reward: {rewards[1]}, Response Length: {response_lengths[1]})")
        print(f"#### Query:\n`{query_texts[1]}`")
        print(f"#### Response:\n`{response_texts[1]}`\n\n")

    if is_eval:
        episodes_dir = exp_dir / "eval_episodes"
    else:
        episodes_dir = exp_dir / "episodes"
    if dist.is_initialized():
        episodes_dir = episodes_dir / f"rank_{rank:02d}"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    # Create wandb table
    table = wandb.Table(columns=["query", "response", "reward", "response_length"])
    for i in range(len(query_texts)):
        table.add_data(query_texts[i], response_texts[i], rewards[i], response_lengths[i])

    if not do_save:
        return table

    with open(episodes_dir / f"eps_{iteration:06d}.json", "w") as f:
        json.dump(
            [
                {
                    "query": query_texts[i],
                    "response": response_texts[i],
                    "reward": rewards[i],
                }
                for i in range(len(query_texts))
            ],
            f,
        )

    return table


def find_last_checkpoint(exp_dir: Path) -> Tuple[Optional[Path], Optional[int]]:
    checkpoint_dir = exp_dir / "checkpoints"
    checkpoints = list(checkpoint_dir.glob("ckpt_*"))
    # Filter out directories that don't have a deepspeed subdirectory
    checkpoints = [ckpt for ckpt in checkpoints if (ckpt / "deepspeed").exists()]
    if not checkpoints:
        return None, None
    ckpt_path = max(checkpoints, key=lambda x: int(x.stem.split("_")[-1]))
    ckpt_iter = int(ckpt_path.stem.split("_")[-1])
    return ckpt_path, ckpt_iter


def load_model_into_vllm(model: Union[DeepSpeedEngine, PreTrainedModel], llm: LLM) -> None:
    """
    Load weights from a HuggingFace model (either wrapped in DeepSpeed or not) into a vLLM inference engine.

    This function transfers the weights from a training model to a vLLM inference engine,
    allowing for efficient inference using the updated model weights.

    Args:
        model (Union[DeepSpeedEngine, PreTrainedModel]): The source model to copy weights from.
            Can be either a DeepSpeed-wrapped model or a regular HuggingFace PreTrainedModel.
        vllm (LLM): The target vLLM inference engine to load the weights into.
            Must be already initialized and ready to accept new weights.

    Returns:
        None
    """
    state_dict = model.module.state_dict() if isinstance(model, DeepSpeedEngine) else model.state_dict()
    llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(state_dict.items())


def initialize_training_process_group(rank: int, world_size: int):
    """
    Initialize the PyTorch distributed process group for multi-GPU training using NCCL backend.

    This function sets up the distributed training environment by:
    1. Setting the CUDA device for the current process
    2. Initializing the process group with NCCL backend
    3. Creating a barrier to ensure all processes are synchronized

    Args:
        rank (int): The rank of the current process (0 to world_size-1)
        world_size (int): Total number of processes participating in the distributed training

    Note:
        - The function uses a free port on localhost for process group initialization
        - A timeout of 1800 seconds (30 minutes) is set for process group initialization
    """
    master_addr = "localhost"
    master_training_port = 8237

    # os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    # os.environ["WORLD_SIZE"] = str(world_size)

    torch.cuda.set_device(rank)

    if rank == 0:
        print(
            f"{'#' * 80}\n" f"# Initializing the training NCCL PG with\n" f"# world_size={world_size} \n" f"{'#' * 80}"
        )

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_training_port}",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(hours=1),
    )
    dist.barrier(device_ids=[rank])
    print(
        f"Rank{rank}: training NCCL PG initialized. "
        f"(world_size={world_size}, local_rank={rank}, gpu_id={torch.cuda.current_device()})"
    )


def clean_up_checkpoints(
    exp_dir: Path, keep_every_n_steps: Optional[int] = None, exclude: Optional[List[Path]] = None
) -> None:
    """
    Clean up checkpoint directories by removing unnecessary files and directories.

    This function manages checkpoint storage by:
    1. Keeping only essential model files (hf_model) in checkpoints that are multiples of keep_every_n_steps
    2. Removing all other checkpoints that are not in the exclude list
    3. Preserving checkpoints that are in the exclude list regardless of their iteration number

    Args:
        exp_dir (Path): The experiment directory containing the checkpoints
        keep_every_n_steps (Optional[int]): If specified, keeps checkpoints that are multiples of this number.
            For these checkpoints, only the hf_model directory is preserved.
        exclude (Optional[List[Path]]): List of checkpoint paths to exclude from cleanup.
            These checkpoints will be preserved regardless of their iteration number.

    Example:
        >>> clean_up_checkpoints(
        ...     exp_dir=Path("experiments/run1"),
        ...     keep_every_n_steps=1000,
        ...     exclude=[Path("experiments/run1/checkpoints/ckpt_5000")]
        ... )
        # This will:
        # - Keep checkpoints 1000, 2000, 3000, etc. (only hf_model directory)
        # - Keep checkpoint 5000 completely (all files)
        # - Remove all other checkpoints
    """
    if exclude is None:
        exclude = []

    checkpoint_dir = exp_dir / "checkpoints"
    for ckpt in checkpoint_dir.glob("ckpt_*"):
        if keep_every_n_steps is None or ckpt in exclude:
            continue

        ckpt_iter = int(ckpt.stem.split("_")[-1])
        if ckpt_iter % keep_every_n_steps == 0:
            # Remove non-hf_model files and dirs
            removed_files_and_dirs = []
            for file in ckpt.iterdir():
                if file.name not in ["hf_model"]:
                    try:
                        removed_files_and_dirs.append(file.name)
                        if file.is_dir():
                            shutil.rmtree(file, ignore_errors=True)
                    except Exception as e:
                        print(f"Error removing {file}: {e}")
            if len(removed_files_and_dirs) > 0:
                print(f"Removed non-hf_model files and dirs: of checkpoint {ckpt.name}")

            continue

        print(f"Removing checkpoint {ckpt}")
        shutil.rmtree(ckpt)


def fix_oov_logits_processor(inference_engine: LLM):
    # https://github.com/issues/recent?issue=vllm-project%7Cvllm%7C13175
    # Qwen and some other models come with a few hundred extra out-of-vocab tokens that can be used for
    # fine-tuning in case new special domain-specific tokens are required.

    # Sampling the OOV token will trigger an error:
    # ValueError: Token id 151791 is out of vocabulary
    # So we mask them using process_token
    # fix_oov # remove asap when this is fixed in vllm, it is dirty and even logit processors are not supported in engine v1 of vllm

    tokenizer_vocab_size = len(inference_engine.get_tokenizer().get_vocab())

    def fix_oov(token_ids, logits):
        logits[tokenizer_vocab_size:] = -float("inf")
        return logits

    return fix_oov


def close_to_zero(tensor: torch.Tensor, mask: torch.Tensor, threshold: float = 1e-8) -> torch.Tensor:
    """
    Computes the number of values in the tensor that are close to zero.
    """
    close_to_zero_mask = torch.abs(tensor) < threshold
    num_close_to_zero = (close_to_zero_mask * mask).sum()
    return num_close_to_zero
