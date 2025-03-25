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
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
# import torch
import transformers

from tqdm import tqdm
import random
from transformers import AutoTokenizer, pipeline
from reward_agent.tools import ObjectDetect, TextExtractor, AttributeChecker

from reward_agent.agent import RewardAgent
from reward_agent.planner import Planner
from reward_agent.build_model import APIModel
from reward_agent.judger import Judger
from reward_agent.difference_model import DifferenceModel
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
    # args = get_args()
    # set_seed(args.seed)
    ###############
    # Setup logging
    ###############

    # load chat template
    image = '/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mtcv/tianyanghan/code/regional_prompt_flux/Regional-Prompting-FLUX/output.jpg'
    query = 'hello'
    response_chosen, response_rejected = '1', '1'

    # # if not datatype in config (default), check args
    # if torch_dtype is None:
    #     # if datatype is bfloat16, then manually turn off quantizaiton (done with bitsandbytes)
    #     if args.torch_dtype == torch.bfloat16:
    #         quantized = False
    #         logger.info("Disabling quantization for bfloat16 datatype")
    #     torch_dtype = args.torch_dtype



    ############################
    # Load reward model pipeline
    ############################

    model_name = 'gpt4o-mini'
    url = "https://aigc.sankuai.com/v1/openai/native"
    key = '1722579627421638682'
    # load planner GPT
    planner =  Planner(model_name, url, key)

    # load difference model
    difference_model = DifferenceModel(model_name, url, key)
    
    # load judger
    judger = Judger(model_name, url, key)

    # load tools
    attribute_checker = AttributeChecker(model_name, url, key)
    text_extractor = TextExtractor()
    object_detector = ObjectDetect(1)
    tools = {
        'attribute_checker':attribute_checker ,
        'text_extractor': text_extractor, 
        'object_detector': object_detector
    }

    # load agent, combine all
    reward_agent = RewardAgent(difference_model, planner, judger, tools)    

    ############################
    # Run inference 
    ############################
    #inference using agent tools
    dummy_judge_res = reward_agent.dummy_judge_different_types(image, query, response_chosen, response_rejected)
    print(f'The winner is {dummy_judge_res}')

            




if __name__ == "__main__":
    main()
