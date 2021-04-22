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
import nltk
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


def text_summarization_spacy(text):
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

        st.subheader("Summary by spaCy")
        st.write(summary)
        st.write(f'{round((1 - len(summary) / len(text)) * 100, 2)}% shorter than the original text.')


def text_summarization_nltk(text):
    # Removing Square Brackets and Extra Spaces
    # formatted_text = re.sub(r'[[0-9]*]', ' ', text)
    # formatted_text = re.sub(r's+', ' ', formatted_text)
    # # Removing special characters and digits
    # formatted_text = re.sub('[^a-zA-Z]', ' ', formatted_text)
    # formatted_text = re.sub(r's+', ' ', formatted_text)

    sentence_list = sent_tokenize(text)

    stopwords = nltk.corpus.stopwords.words('english')
    word_frequencies = {}
    for word in nltk.word_tokenize(text):
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    maximum_frequency = max(word_frequencies.values())

    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    summary_sentences = nlargest(7, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)

    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / maximum_frequency)

    st.subheader("Summary by NLTK")
    st.write(summary)
    st.write(f'{round((1 - len(summary) / len(text)) * 100, 2)}% shorter than the original text.')


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
