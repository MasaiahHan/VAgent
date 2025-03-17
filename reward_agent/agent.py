
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



class RewardAgent:
    def __init__(self, planner, judger, judger_type, reward_model, tokenizer, tools):
        self.planner = planner
        self.judger = judger
        self.judger_type = judger_type
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.tools = tools

        self.tool_call_count = Counter()


    def dummy_judge_different_types(self, instruction, response_chosen, response_rejected, **kwargs):
        # first plan
        dummy_plan_result = self.planner.plan(instruction)

        # tools
        # tools.forward()

        # reward model
        dummy_result = rm.dummy_get_reward(instruction, response_chosen, response_rejected, **kwargs)

        #judge winnder
        dummy_judge_res, scores = self.judger.dummy_judge(instruction, dummy_result)

        return dummy_judge_res, scores

