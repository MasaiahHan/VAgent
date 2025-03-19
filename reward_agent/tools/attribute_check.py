# -*- coding: utf-8 -*-
import openai
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")



class GPTAgent():
    def __init__(self, api_key , api_base ):
        openai.api_key = api_key
        openai.api_base = api_base

    
    def forward(self, image, prompt): 
        try:
            base64_image = encode_image(image)
            response = openai.ChatCompletion.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": "you are an expert in answering questions related to images. …  Summarize your answer in the following format: The answer is: xxx"},
                                {"role": "user", "content":  [
                                    {"type": "text", "text": {prompt}},
                                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]}
                            ]
                        )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error occured when connecting to GPT-4o-mini: {e}")
            return "Error occured when connecting to GPT-4o-mini."  


