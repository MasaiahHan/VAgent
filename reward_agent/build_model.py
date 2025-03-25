import openai
import requests
import base64

class APIModel:
    def __init__(self, model_name, base_url, api_key):
        openai.api_key = api_key
        openai.api_base = base_url
        self.model_name = model_name

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")



    def generate(self,  message_template, max_tokens=1024, temperature=0.0):
        try:
            chat_completion = openai.ChatCompletion.create(
                            model="gpt-4o-mini",
                            messages=message_template
                        )
            response = chat_completion.choices[0].message.content
        except Exception as e:
            print(e)
            response = "NA"
        return response


    
    def safety_check(self, query):
        try:
            chat_completion = self.client.moderations.create(
                model=self.model_name,
                input=query
            )
            import pdb; pdb.set_trace()
            return dict(chat_completion.results[0].categories), chat_completion.results[0].flagged
        except Exception as e:
            print(e)
            response = {"NA": 1}
        return response






