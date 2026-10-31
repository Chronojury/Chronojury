"""SWER defaults. Command-line arguments override the corresponding values.

Keep API keys and private endpoint addresses in environment variables.
"""
import os


# Sent explicitly to Qwen-compatible model servers. Individual requests can override it.
QWEN_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}


# Defaults for chronojury-reason. The model can also be set by CHRONOJURY_MODEL.
REASONING_CONFIG = {
    "model": "your_model_name",
    "temperature": 0.2,
    "max_completion_tokens": 512,
    "max_workers": 5,
    "prefetch_cases": 2,
    "sleep": 0,
    "output_dir": "Results/test",
}

# Preserve the existing defaults for direct Python calls to BatchDebateGenerator.
BATCH_CONFIG = {
    "model": "your_model_name",
    "temperature": 0.2,
    "max_completion_tokens": 4096,
    "max_workers": 5,
    "prefetch_cases": 2,
    "sleep": 0.5,
    "output_dir": "Results",
}

# Each task keeps its own sampling settings. Do not add token limits to tasks
# that previously let the model server determine the response length.
STAGE_REQUEST_CONFIG = {
    "segmentation": {"temperature": 0.2, "max_tokens": 512},
    "selection": {"temperature": 0.2, "timeout": 30},
    "perception": {"temperature": 0.2, "timeout": 60},
    "multimodal_verdict": {"timeout": 60},
}

SUMMARY_CONFIG = {
    "temperature": 0.2,
    "timeout": 30,
    "max_workers": 20,
    "data_dir": "dataset",
    "segment_dir": "segment",
    "output_dir": "results_stage_summaries_9B",
}

MAX_VIDEO_FRAMES = 8
MODEL_CONTEXT_LENGTHS = {
    "your_model_name": int(os.environ.get("QWEN_MAX_MODEL_LEN", "131072")),
}
DEFAULT_CONTEXT_LENGTH = 128000

# Keep six jurors as the default; robustness experiments can override the count.
FREE_ROUNDS = 3
NUM_AGENTS = int(os.environ.get("CHRONOJURY_NUM_AGENTS", "6"))
if NUM_AGENTS < 2:
    raise ValueError("CHRONOJURY_NUM_AGENTS must be at least 2")
DEBATE_AGENTS = [f"Agent_{i}" for i in range(NUM_AGENTS)]

# Existing network files take precedence over the degree used for new networks.
FOLLOWING_NETWORK_OUT_DEGREE = 2
FOLLOWING_NETWORKS_DIR = 'following_networks' if NUM_AGENTS == 6 else f'following_networks_{NUM_AGENTS}'
SAVE_DIR = "Results/train"
