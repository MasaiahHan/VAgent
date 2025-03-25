import easyocr

class TextExtractor:
    def __init__(self,  text_only=False ):
        self.reaader = easyocr.Reader(['ch_sim', 'en'])
        self.text_only = text_only
    
    def generate(self, image_path, prompt):
        """
        returns: List[]

            example: each item represents a bounding box, the text detected and confident level, respectively.
                [([[189, 75], [469, 75], [469, 165], [189, 165]], '愚园路', 0.3754989504814148),
                ([[86, 80], [134, 80], [134, 128], [86, 128]], '西', 0.40452659130096436),
                ([[517, 81], [565, 81], [565, 123], [517, 123]], '东', 0.9989598989486694),
                ([[78, 126], [136, 126], [136, 156], [78, 156]], '315', 0.8125889301300049),
                ([[514, 126], [574, 126], [574, 156], [514, 156]], '309', 0.4971577227115631),
                ([[226, 170], [414, 170], [414, 220], [226, 220]], 'Yuyuan Rd.', 0.8261902332305908),
                ([[79, 173], [125, 173], [125, 213], [79, 213]], 'W', 0.9848111271858215),
                ([[529, 173], [569, 173], [569, 213], [529, 213]], 'E', 0.8405593633651733)]

        """
        if self.text_only:
            result = self.reader.readtext(image_path, detail = 0)
        else:
            result = self.reader.readtext(image_path)
        return result


