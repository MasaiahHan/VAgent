import easyocr

class TextExtractor:
    def __init__(self,  text_only=True ):
        self.reaader = easyocr.Reader(['ch_sim', 'en'])
        self.text_only = text_only
    
    def forward(self, image_path):
        if self.text_only:
            result = self.reader.readtext(image_path, detail = 0)
        else:
            result = self.reader.readtext(image_path)
        return result


