


template = """

You are an expert in extracting the differences between two responses. You need to consider the following types of differences: \n
1) Object existence: one response contains some object that the other does not contain \n
2) Object attribute: some objects are contained in both responses, but their attributes are different. E.g., color, size, position, texture ... \n
3）Object count: the count of some objects differ in the two responses. \n
You need to list out the differences in the two responses one by one into a list of strings in the following format: \n
["<existence>dog</existence>", "<existence>skateboard</existence>", "<attribute>the color of the boy's shirt</attribute>", "<attribute>the texture of the table</attribute>", "<count>people</count>"] \n
I will now provide you with the two reponses: \n
"""


class DifferenceModel:
    def __init__(self, model):
        """
        Initialize the difference proposal with a given model.

        Args:
            model
        """
        self.model = model

    
    def propose_diff(self, response1, response2):
        """
        Determine the difference between two responses.

        Args:
            response1 (str): The instruction provided.
            response2 (str): The generated response.

        Returns:
            
        """
        system_prompt = f"""
        {template}

        Response1:
        {response1}\n
        
        Response2:
        {response2}

        """
        messages = [
            {"role": "user", "content": plan_system_prompt}
        ]

        # Use the model to generate a decision
        dummy_output = self.model.generate_chat(messages)

        return dummy_output