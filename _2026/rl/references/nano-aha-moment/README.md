# nanoAhaMoment: Single File "RL for LLM" Library
> Amirhossein Kazemnejad*, Milad Aghajohari*, Alessandro Sordoni, Aaron Courville, Siva Reddy

Implementation of DeepSeek R1-zero style training with:

- Single 80G GPU (and also multi-GPU)
- No RL Library 
- 3B Base Model (and also 7B models with multi-GPU)
- Full Parameter Tuning 
- Efficient (Competetive performance to verl but much simpler)
- Up to 32K context size for 3B model with multi-GPU (or 16K context size for 7B model)

## News
- **June 2025**: Added multi-GPU support for faster training and 7B models
- **June 2025**: Added VinePPO episode generation (experimental)

Inspired by [TinyZero](https://github.com/Jiayi-Pan/TinyZero) and [Mini-R1](https://www.philschmid.de/mini-deepseek-r1), but designed to be much **simpler**, **cleaner**, and **faster**, with every line of code visible and understandable.

## Karpathy-style Detailed Lecture on YouTube

- [nanoAhaMoment: RL for LLM from Scratch with 1 GPU - Part 1](https://youtu.be/ZMO5tv30ri8)
- [nanoAhaMoment: RL for LLM from Scratch with 1 GPU - Part 2](https://youtu.be/dxhCyhc_bcQ)

## File Descriptions
- `nano_r1.ipynb` is the interactive single file jupyter notebook with tutorial.
- `nano_r1_script.py` is also just the `nano_r1.ipynb` but for convenience of running with python and multi-GPU support.
- `notebooks/checkpoint_playground.ipynb` is a notebook for comparing different model checkpoints (including our trained model) and playing with them.
- [🤗 McGill-NLP/nano-aha-moment-3b](https://huggingface.co/McGill-NLP/nano-aha-moment-3b): The HF model trained using the above script (~60\% Accuracy on CountDown Task)

## Setup Instructions

1. **Clone the repository**  
   ```bash
   git clone https://github.com/McGill-NLP/nano-aha-moment.git
   ```

2. **Install dependencies**  
   First, make sure cuda 12.4 is installed.
   
   Install PyTorch:
   ```bash
   pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
   ```
   
   Install the rest of the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   **Alternative Installation with uv (Optional)**  
   ```bash
   uv sync
   uv sync --extra compile  # Install flash-attention
   ```

3. **Run the training script**  
   Open `nano_r1.ipynb` or `nano_r1_script.py` and start training.

   > If using uv, you can run with either `uv run nano_r1_script.py` or activate the env with `source .venv/bin/activate` and run with `python nano_r1_script.py`

## Multi-GPU Training
Here is the command to run the training script with 4 GPUs:
```bash
python nano_r1_script.py --nproc 4  # Use 4 GPUs
```

## Batch Sizes for different context lengths

| Context Length | 3B Model (per_device_batch_size) | 7B Model (per_device_batch_size) |
|---------------|----------------------------------|----------------------------------|
| 1024            | 32                               | 16                               |
| 2048            | 16                               | 8                               |
| 4K            | 8                               | 4                               |
| 8K            | 4                               | 2                                |
| 16K           | 2                                | 1                                |
| 32K           | 1                                | N/A                              |

> Note: These batch sizes are optimized for 4xA100 80GB GPUs. For other GPU types, you may need to adjust the batch sizes accordingly.

## Todos
- [ ] Full evaluation suite
- [x] Multi-GPU support (Added June 2025)

## Acknowledgement
We gratefully acknowledge the support of Lambda for providing compute resources through their research compute grant.
<p align="left">
  <img src="https://lambda.ai/hubfs/lambda%20logo%202.svg" alt="Lambda AI" width="200">
</p>

## Citation
If you use this codebase in your research, please cite us using:

```bibtex
@misc{Kazemnejad2025:NanoAhaMoment,
  author       = {Amirhossein Kazemnejad and Milad Aghajohari and Alessandro Sordoni and Aaron Courville and Siva Reddy},
  title        = {Nano Aha! Moment: Single File "RL for LLM" Library},
  year         = {2025},
  howpublished = {\url{https://github.com/McGill-NLP/nano-aha-moment}},
  note         = {GitHub repository}
}
```
