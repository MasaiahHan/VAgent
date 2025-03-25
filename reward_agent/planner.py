from .build_model import APIModel
import openai
class Planner(APIModel):    
    def generate(self, image, instruction, response_chosen, response_rejected, difference):
        """
        Determine whether the given instruction and response require a constraint check or a factuality check.

        Returns:
            dict
        """

        tools_description = """
        [attribute_checker]
        A tool that checks certain attribute in the image.
        inputs:
        image: str - The path to the image file.
        question: the question regarding the attribute of the image.
        output:
        string - the number of object occurances
        examples:
        result=counter(image="path/to/image.png", question="what is the color of the shirt. the boy is wearing?") # querying for the color of the shirt the boy is wearing

        [text_extractor]
        A tool that extracts the texts with their locations in the image.
        inputs:
        image: str - The path to the image file.
        output:
        list of dicts - the texts associated with their locations
        examples:
        result=counter(image="path/to/image.png") # extracting the texts

        [object_detector]
        A tool that counts the number of certain object in the image.
        inputs:
        image: str - The path to the image file.
        label: object label to count.
        output:
        int - the number of object occurances
        examples:
        result=counter(image="path/to/image.png", label="baseball") # count the number of baseballs in the image
        
        """

        plan_system_prompt = f"""
        You are responsible for selecting the appropriate tool to verify the property of an image. 
        You will be given the property to verify, e.g., existence, color, count , etc.
        You will be provided with a set of tools with their descriptions. You need to select the appropriate tool, and also determine the values of the input parameters to the tool.
        tools: {tool_description}
        property: {difference}
        Answer following the format: [{{"tool": "tool_name", "inputs":the values of the input parameters to the tool }},{{}}]
        """

        try:
            chat_completion = openai.ChatCompletion.create(
                            model=self.model_name,
                            messages=[
                                {"role": "user", "content":  [
                                    {"type": "text", "text": {plan_system_prompt}},
                                    ]}
                            ]
                        )
            dummy_output = chat_completion.choices[0].message.content

            # need modified, convert dummy_output to the following List[dict] format 

            #
            dummy_output = [
                {'tool': 'attribute_checker', 'input-text':'test' },
                {'tool': 'text_extractor', 'input-text':'test' },
                {'tool': 'object_detector', 'input-text':'test' },
            ]

        except Exception as e:
            print(e)
            dummy_output = []
        return dummy_output


    
