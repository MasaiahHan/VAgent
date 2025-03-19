
from .tools import ConstraintAnalyzer, evaluate_if_reward, evaluate_if_reward_multi
from .tools import FactChecker
from .tools import GoogleSerperAPIWrapper
from .tools import process_judgment, process_judgment_multi
from .rm import rm
import asyncio
import logging
import random
from collections import Counter
from itertools import combinations
from difference_model import DifferenceModel


class RewardAgent:
    def __init__(self, planner, judger, judger_type, reward_model, tokenizer, tools):
        self.planner = planner
        self.judger = judger
        self.judger_type = judger_type
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.tools = tools

        self.tool_call_count = Counter()


    def dummy_judge_different_types(self, image, instruction, response_chosen, response_rejected, **kwargs):

        # difference propose
        differences = difference_model.propose_diff(response_chosen, response_rejected)

        dummy_result = []

        # first plan, tool_selected
        for difference in differences:
            dummy_plan_result = self.planner.plan(instruction, difference)

            # using tools
            for tool in dummy_plan_result['tools']:
                tool_result = self.tools[tool].forward(image, dummy_plan_result['inputs_text'])
                dummy_result.append(tool_result)
        #judge winnder
        dummy_judge_res, scores = self.judger.dummy_judge(instruction, dummy_result)

        return dummy_judge_res, scores

