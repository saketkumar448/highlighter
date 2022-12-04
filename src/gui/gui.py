# GUI for higlighter

import re
import sys
import json
import datetime
import streamlit as st
from annotated_text import annotated_text


sys.path.append("../inferencer/")
from inferencer import Inferencer


@st.cache(allow_output_mutation=True)
def load_inferencer():
    '''
    Avoids loading of inferencer object again and again
    '''
    inferencer = Inferencer()
    return inferencer


inferencer = load_inferencer()


def main():

    # creating heading "Highlight"
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Highlight</h2>
    </div>
    """
    st.sidebar.markdown(html_temp, unsafe_allow_html=True)

    # Input area for text
    text = st.sidebar.text_area(label = 'Text', value="Enter Text",
                                height = 20)
    
    # "Highlight" button
    if st.sidebar.button("Highlight"):

        # Highlighting the text
        highlighted_tokens = inferencer.highlight(text)[0]

        # Manipulating higlighted tokens for annotated_text
        for idx, ele in enumerate(highlighted_tokens):
            if ele[1] == 1:
                highlighted_tokens[idx] = (ele[0], '', '#afa')
            else:
                highlighted_tokens[idx] = (' '+ele[0])

        # Printing the highlighted text
        annotated_text(*highlighted_tokens)


if __name__ == '__main__':
    main()