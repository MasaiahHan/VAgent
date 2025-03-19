# Copyright 2023 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from fastchat.conversation import get_conv_template
from tqdm import tqdm
import random
from transformers import AutoTokenizer, pipeline

from rewardbench import (
    REWARD_MODEL_CONFIG,
    check_tokenizer_chat_template,
    load_eval_dataset,
    save_to_hub,
    torch_dtype_mapping,
)
from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
from rewardbench.utils import calculate_scores_per_section
from reward_agent.agent import RewardAgent
from reward_agent.planner import Planner
from reward_agent.build_model import APIModel, LocalAPIModel
from reward_agent.judger import Judger
from datasets import load_dataset

# Enable TensorFloat32 (TF32) tensor cores on Ampere GPUs for matrix multiplications (faster than FP32)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# get token from HF_TOKEN env variable, but if it doesn't exist pass none
HF_TOKEN = os.getenv("HF_TOKEN", None)
# this is necessary to automatically log in when running this script in docker/batch beaker jobs
if HF_TOKEN is not None:
    from huggingface_hub._login import _login

    _login(token=HF_TOKEN, add_to_git_credential=False)

import torch
import random
import numpy as np

def set_seed(seed: int):
    """
    设置随机种子，以确保结果可以复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 确保 cudnn 的行为是确定性的
    torch.backends.cudnn.benchmark = False  # 禁用 cudnn 的自适应算法


def get_args():
    """
    Parse arguments strings model and chat_template
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="path to model")
    args = parser.parse_args()
    args.torch_dtype = torch_dtype_mapping(args.torch_dtype)
    return args


def main():
    args = get_args()
    set_seed(args.seed)
    ###############
    # Setup logging
    ###############
    accelerator = Accelerator()

    # load chat template
    chat_template = args.chat_template
    conv = get_conv_template(chat_template)


    # if not datatype in config (default), check args
    if torch_dtype is None:
        # if datatype is bfloat16, then manually turn off quantizaiton (done with bitsandbytes)
        if args.torch_dtype == torch.bfloat16:
            quantized = False
            logger.info("Disabling quantization for bfloat16 datatype")
        torch_dtype = args.torch_dtype



    ############################
    # Load reward model pipeline
    ############################
    BATCH_SIZE = args.batch_size
    reward_model = load_reward_model(model, config)
    # reward_pipe = pipeline_builder(
    #     "text-classification",
    #     model=model,
    #     tokenizer=tokenizer,
    # )

    # load planner GPT
    TOKEN = os.environ["OPENAI_API_KEY"]
    base_url = os.environ["OPENAI_BASE_URL"]
    planner_model =  APIModel(base_url, llm_backbone, TOKEN)
    planner = Planner(planner_model)
    
    # load judger
    judger_model = APIModel(base_url, llm_backbone, TOKEN)
    judger = Judger(judger_model)

    # load tools
    tools = {}

    # load agent, combine all
    reward_agent = RewardAgent(planner, judger, reward_model, tools)

    # load difference proposal
    difference_model = DifferenceModel(diff_model)

    ############################
    # Run inference 
    ############################
    # prepare for inference
    reward_pipe = accelerator.prepare(reward_pipe)

    #inference using agent tools
    dummy_judge_res, scores = reward_agent.dummy_judge_different_types(instruction, response_chosen, response_rejected)

            




if __name__ == "__main__":
    main()
