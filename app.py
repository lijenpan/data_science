import streamlit as st

from streamlit_utils import display_app_header, display_side_panel_header, space_header
from topic_models.streamlit_topic_models import topic_model
from streamlit_nlp import sentiment_analysis, entity_extraction, text_summarization_spacy, text_summarization_nltk,\
    aspect_based_sentiment_analysis


def main():
    # Main panel title
    display_app_header(main_txt='Natural Language Processing Web Application', sub_txt="")

    display_side_panel_header(txt='Step 1:')
    data_input_mthd = st.sidebar.radio("Select Data Input Method", ('Copy-Paste Text', 'Upload a CSV File'))

    display_side_panel_header(txt='Step 2:')
    option = st.sidebar.radio("Select a NLP Service", ("Sentiment Analysis", "Entity Extraction", "Text Summarization",
                                                       "Aspect Based Sentiment Analysis", "Topic Modeling"))
    text = ""
    if option != "Topic Modeling":
        if data_input_mthd == "Copy-Paste Text":
            st.subheader("Enter the text you would like to analyze.")
            text = st.text_area("Enter text")
        elif data_input_mthd == "Upload a CSV File":
            user_file = st.file_uploader("Choose a file", type=["txt", "csv"], accept_multiple_files=False, key="file_uploader")
            if user_file:
                text = user_file.read().decode("utf-8")
                # Clear file buffer
                user_file.seek(0)

    # You can swap out functions here with your own implementation.
    if option == 'Sentiment Analysis':
        space_header()
        sentiment_analysis(text)
    elif option == 'Entity Extraction':
        space_header()
        entity_extraction(text)
    elif option == 'Text Summarization':
        space_header()
        text_summarization_spacy(text)
        text_summarization_nltk(text)
    elif option == "Aspect Based Sentiment Analysis":
        space_header()
        aspect1 = st.text_input("Word 1")
        aspect2 = st.text_input("Word 2")
        aspect_based_sentiment_analysis(text, aspect1, aspect2)
    elif option == "Topic Modeling":
        topic_model(data_input_mthd)


if __name__ == "__main__":
    main()
