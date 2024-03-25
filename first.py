import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
# nltk.download('stopwords')
stopwords_english = stopwords.words('english')

def read_article(file_name):
    file = open(file_name, "r")
    filedata = file.readlines()
    print("File Data:", filedata)
    article = filedata[0].split(". ")
    sentences = []
    for sentence in article:
        sentences.append(sentence.split())
    print("Sentences:", sentences)
    return sentences

def sentence_similarity(sent1, sent2, stopwords=stopwords_english):
    sent1 = [w.lower() for w in sent1 if w not in stopwords]
    sent2 = [w.lower() for w in sent2 if w not in stopwords]
    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    for w in sent1:
        vector1[all_words.index(w)] += 1
    for w in sent2:
        vector2[all_words.index(w)] += 1
    return 1 - cosine_distance(vector1, vector2)

def gen_sim_matrix(sentences, stopwords):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stopwords)
    return similarity_matrix

def generate_summary(file_name, top_n=5):
    sentences = read_article(file_name)
    if not sentences:
        return
    sentence_similarity_matrix = gen_sim_matrix(sentences, stopwords_english)
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)
    ranked_sentences = sorted(((scores[i], ' '.join(s)) for i, s in enumerate(sentences)), reverse=True)
    summary_sentences = [s[1] for s in ranked_sentences[:top_n]]
    summary = ' '.join(summary_sentences)
    print(summary)

# We can adjust the file name and top_n value as per our need and requirements
generate_summary("file.txt", 2)