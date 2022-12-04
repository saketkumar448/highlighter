'''
Preprocessing of input text
'''

import re
import string


# Loading configuration
import sys
sys.path.append("../config/")
from config import Config
cfg = Config


class Preprocessor:
    '''
    Preprocessing input text 
    '''

    def clean_text(self, text) -> str:
        '''
        Preprocessing input text

        Arguments: text(str), input text for highlighter

        Returns: str, preprocessed text
        '''

        text = text.replace('¶', '').replace('\n', '')
    
        # Putting spaces before and after punctuations
        for punct in string.punctuation:

            text = text.replace(punct, f" {punct} ")

        # Removing multiple spaces
        text = re.sub(' +', ' ', text)

        # Removing leading and trailing spaces
        text = text.strip()

        return text