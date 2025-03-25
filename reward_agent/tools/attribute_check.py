# -*- coding: utf-8 -*-
from ..build_model import APIModel
import base64
import openai

class AttributeChecker(APIModel):

    def generate(self, image_path, prompt): 
        """
        returns: string
        """

        attribute_check_prompt_template = "you are an expert in answering questions related to images. …  Summarize your answer in the following format: The answer is: xxx"

        try:
            base64_image = self.encode_image(image_path)
            response = openai.ChatCompletion.create(
                            model=self.model_name,
                            messages=[
                                {"role": "system", "content": attribute_check_prompt_template},
                                {"role": "user", "content":  [
                                    {"type": "text", "text": {prompt}},
                                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]}
                            ]
                        )
            check_result = response.choices[0].message.content
        except Exception as e:
            print(e)
            check_result = ''
        return check_result



