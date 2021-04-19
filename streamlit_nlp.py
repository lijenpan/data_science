from collections import Counter
from heapq import nlargest

import streamlit as st
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from textblob import TextBlob
from nltk.tokenize import sent_tokenize
import aspect_based_sentiment_analysis as absa

nlp = spacy.load("en_core_web_sm")

# Heading for the app
st.title("Natural Language Processing Web Application")
st.subheader("What type of NLP service would you like to use?")

# Picking what NLP task you want to do
option = st.selectbox("NLP Service", ("Sentiment Analysis", "Entity Extraction", "Text Summarization",
                                      "Aspect Based Sentiment Analysis", "Topic Modeling"))

# Textbox for user input
st.subheader("Enter the text you would like to analyze.")
text = st.text_input("Enter text")

if option == "Aspect Based Sentiment Analysis":
    aspect1 = st.text_input("First aspect")
    aspect2 = st.text_input("Second aspect")

# Display results of the NLP task
st.header("Results")


def ent_recognizer(ent_dict, type_ent):
    return [ent for ent in ent_dict if ent_dict[ent] == type_ent]


if option == 'Sentiment Analysis':
    # Create graph for sentiment across each sentence in the text input
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
elif option == 'Entity Extraction':
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
elif option == 'Text Summarization':
    # Text summarization is no longer available in gensim 4.x.
    # So let's try spacy's implementation.
    doc = nlp(text)
    keyword = []
    stopwords = list(STOP_WORDS)
    pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
    for token in doc:
        if (token.text in stopwords or token.text in punctuation):
            continue
        if (token.pos_ in pos_tag):
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
elif option == "Aspect Based Sentiment Analysis":
    nlp = absa.load()
    a1, a2 = nlp(text, aspects=[aspect1, aspect2])
    st.subheader("Summary")
    st.write(aspect1 + " Sentiment:\t" + str(a1.sentiment))
    st.write(aspect2 + " Sentiment:\t" + str(a2.sentiment))
