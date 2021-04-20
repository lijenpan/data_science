from collections import Counter
from heapq import nlargest

import numpy as np
import streamlit as st
from streamlit import components
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from textblob import TextBlob
from nltk.tokenize import sent_tokenize
import aspect_based_sentiment_analysis as absa

nlp = spacy.load("en_core_web_sm")


def ent_recognizer(ent_dict, type_ent):
    return [ent for ent in ent_dict if ent_dict[ent] == type_ent]


def sentiment_analysis(text):
    # Create graph for sentiment across each sentence in the text input
    if text:
        sents = sent_tokenize(text)
        entire_text = TextBlob(text)
        sent_scores = []
        for sent in sents:
            text = TextBlob(sent)
            score = text.sentiment[0]
            sent_scores.append(score)

        # Plot line chart
        st.line_chart(sent_scores)

        # Polarity and Subjectively of the entire text
        sent_total = entire_text.sentiment
        st.write("The sentiment of the overall text below.")
        st.write(sent_total)


def entity_extraction(text):
    if text:
        entities = []
        entity_labels = []
        doc = nlp(text)
        for ent in doc.ents:
            entities.append(ent.text)
            entity_labels.append(ent.label_)
        ent_dict = dict(zip(entities, entity_labels))
        ent_org = ent_recognizer(ent_dict, "ORG")
        ent_cardinal = ent_recognizer(ent_dict, "CARDINAL")
        ent_person = ent_recognizer(ent_dict, "PERSON")
        ent_date = ent_recognizer(ent_dict, "DATE")
        ent_gpe = ent_recognizer(ent_dict, "GPE")

        st.write("Organization Entities:\t" + str(ent_org))
        st.write("Cardinal Entities:\t" + str(ent_cardinal))
        st.write("Person Entities:\t" + str(ent_person))
        st.write("Date Entities:\t" + str(ent_date))
        st.write("GPE Entities:\t" + str(ent_gpe))


def text_summarization(text):
    # Text summarization is no longer available in gensim 4.x.
    # So let's try spacy's implementation.
    if text:
        doc = nlp(text)
        keyword = []
        stopwords = list(STOP_WORDS)
        pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
        for token in doc:
            if token.text in stopwords or token.text in punctuation:
                continue
            if token.pos_ in pos_tag:
                keyword.append(token.text)

        freq_word = Counter(keyword)
        max_freq = Counter(keyword).most_common(1)[0][1]
        for word in freq_word.keys():
            freq_word[word] = (freq_word[word] / max_freq)

        sent_strength = {}
        for sent in doc.sents:
            for word in sent:
                if word.text in freq_word.keys():
                    if sent in sent_strength.keys():
                        sent_strength[sent] += freq_word[word.text]
                    else:
                        sent_strength[sent] = freq_word[word.text]

        summarized_sentences = nlargest(3, sent_strength, key=sent_strength.get)
        final_sentences = [w.text for w in summarized_sentences]
        summary = ' '.join(final_sentences)

        st.subheader("Summary")
        st.write(summary)


def aspect_based_sentiment_analysis(text, aspect1, aspect2):
    if text and aspect1 and aspect2:
        recognizer = absa.aux_models.BasicPatternRecognizer()
        absa_nlp = absa.load(pattern_recognizer=recognizer)
        a1, a2 = absa_nlp(text, aspects=[aspect1, aspect2]).examples
        st.subheader("Summary")
        a1_rounded_scores = np.round(a1.scores, decimals=3)
        a2_rounded_scores = np.round(a2.scores, decimals=3)
        st.write(f'{str(a1.sentiment)} for "{a1.aspect}"')
        st.write(f'Scores (neutral/negative/positive): {a1_rounded_scores}')
        if a1.review.patterns is not None:
            components.v1.html(absa.plots.display_html(a1.review.patterns)._repr_html_())
        st.write(f'{str(a2.sentiment)} for "{a2.aspect}"')
        st.write(f'Scores (neutral/negative/positive): {a2_rounded_scores}')
        if a2.review.patterns is not None:
            components.v1.html(absa.plots.display_html(a2.review.patterns)._repr_html_())
    else:
        st.write("Please enter aspect based sentiment analysis parameters.")


def main():
    # Heading for the app
    st.title("Natural Language Processing Web Application")
    st.subheader("What type of NLP service would you like to use?")

    # Picking what NLP task you want to do
    option = st.selectbox("NLP Service", ("Sentiment Analysis", "Entity Extraction", "Text Summarization",
                                          "Aspect Based Sentiment Analysis", "Topic Modeling"))

    # Textbox for user input
    st.subheader("Enter the text you would like to analyze.")
    text = st.text_area("Enter text")
    user_files = st.file_uploader("Choose a file", type=["txt", "csv"], accept_multiple_files=True, key="file_uploader")
    for file in user_files:
        text = file.read().decode("utf-8")
        file.seek(0)

    if option == "Aspect Based Sentiment Analysis":
        aspect1 = st.text_input("First aspect")
        aspect2 = st.text_input("Second aspect")

    # Display results of the NLP task
    st.header("Results")

    if option == 'Sentiment Analysis':
        sentiment_analysis(text)
    elif option == 'Entity Extraction':
        entity_extraction(text)
    elif option == 'Text Summarization':
        text_summarization(text)
    elif option == "Aspect Based Sentiment Analysis":
        aspect_based_sentiment_analysis(text, aspect1, aspect2)


if __name__ == "__main__":
    main()
