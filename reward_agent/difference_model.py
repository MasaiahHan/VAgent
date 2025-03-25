from .build_model import APIModel
import openai
class DifferenceModel(APIModel):
    def generate(self, response1, response2):
        """
        Determine the difference between two responses.

        Args:
            response1 (str): The instruction provided.
            response2 (str): The generated response.

        Returns:
            List[String]
        """

        template = """
            You are an expert in extracting the differences between two responses. You need to consider the following types of differences: \n
            1) Object existence: one response contains some object that the other does not contain \n
            2) Object attribute: some objects are contained in both responses, but their attributes are different. E.g., color, size, position, texture ... \n
            3）Object count: the count of some objects differ in the two responses. \n
            You need to list out the differences in the two responses one by one into a list of strings in the following format: \n
            ["<existence>dog</existence>", "<existence>skateboard</existence>", "<attribute>the color of the boy's shirt</attribute>", "<attribute>the texture of the table</attribute>", "<count>people</count>"] \n
            I will now provide you with the two reponses: \n
            """

        system_prompt = f"""
        {template}

        Response1:
        {response1}\n
        
        Response2:
        {response2}

        """

        try:
            chat_completion =openai.ChatCompletion.create(
                            model=self.model_name,
                            messages=[
                                {"role": "user", "content":  [
                                    {"type": "text", "text": {system_prompt}},
                                    ]}
                            ]
                        )
            dummy_output = chat_completion.choices[0].message.content
            difference_list = eval(dummy_output)

        except Exception as e:
            print(e)
            difference_list = []
        return difference_list

    
