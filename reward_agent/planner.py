
class Planner:
    def __init__(self, model):
        """
        Initialize the Planner with a given model.

        Args:
            model: An instance of a text generation model that supports planning.
        """
        self.model = model

    
    def plan(self, instruction, difference):
        """
        Determine whether the given instruction and response require a constraint check or a factuality check.

        Args:
            instruction (str): The instruction provided.
            response (str): The generated response.

        Returns:
            dict: A dictionary with keys 'constraint_check' and 'factuality_check', each mapping to a boolean.
        """
        plan_system_prompt = f"""
        select which tool to use for verifying the differences. The descriptions of the tools have to be fed into the selector

        [Instruction]
        {instruction}

        [Checks]
        {difference}

        """
        messages = [
            {"role": "user", "content": plan_system_prompt}
        ]

        # Use the model to generate a decision
        dummy_output = self.model.generate_chat(messages)

        return dummy_output
