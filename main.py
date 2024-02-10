import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import spacy

from database import save_vectors
from plot import plot_vectors

nlp = spacy.load('ja_ginza')

# ユーザーからの長いテキストを入力として受け取る
TextInput = input("Enter a long text: ")

# テキストを文に分割
doc = nlp(TextInput)
sentences = [sent.text for sent in doc.sents]
for sentence in sentences:
    print(sentence)

# 各文をベクトルに変換
vectors = np.array([nlp(sentence).vector for sentence in sentences])
data_to_save = [{"text": sentence, "vector": vector.tolist()} for sentence, vector in zip(sentences, vectors)]
save_vectors(data_to_save)

# コサイン類似度を計算
similarity_matrix = cosine_similarity(vectors)
print(similarity_matrix)

plot_vectors()