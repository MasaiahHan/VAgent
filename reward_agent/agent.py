
class RewardAgent:
    def __init__(self,difference_model, planner, judger, tools):
        self.planner = planner
        self.judger = judger
        self.tools = tools
        self.difference_model = difference_model



    def dummy_judge_different_types(self, image, instruction, response_chosen, response_rejected, **kwargs):
        dummy_result = []

        # difference propose
        differences = self.difference_model.generate(response_chosen, response_rejected) # List[String]

        

        # first plan, tool_selected
        for difference in differences:
            dummy_plan_result = self.planner.generate(image, instruction, response_chosen, response_rejected, difference)

            # using tools
            for tool in dummy_plan_result['tools']:
                tool_result = self.tools[tool].generate(image, dummy_plan_result['inputs_text'])
                dummy_result.append({'input_question':dummy_plan_result['inputs_text'], 'answer': tool_result})
        #judge winnder
        winner = self.judger.generate(instruction, dummy_result,  response_chosen, response_rejected)

        return winner

