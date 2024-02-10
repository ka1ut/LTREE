import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import spacy

from database import save_vectors, setLabel
from plot import plot_vectors

TextInput = input("Enter a long text: ")

nlp = spacy.load('ja_ginza')

# テキストを文に分割
doc = nlp(TextInput)
sentences = [sent.text for sent in doc.sents]
print("")
print("---------sentences---------")
for sentence in sentences:
    print(sentence)
print("---------------------------")

data_to_save = []
Label = setLabel()
for sentence in sentences:
    doc = nlp(sentence)
    # テキストを単語に分割
    for token in doc:
        # 単語のベクトルを取得し、リストに追加
        data_to_save.append({"text": token.text, "WordClass": token.pos_, "vector": token.vector.tolist(), "label": f"{Label}"})

# 各単語とそのラベルを表示
print("")
print("-----------word------------")
for word in data_to_save:
    print(f"text: {word['text']}, label: {word['label']}")
print("---------------------------")

save_vectors(data_to_save)

# コサイン類似度を計算
vectors = [item['vector'] for item in data_to_save]
similarity_matrix = cosine_similarity(vectors)
print("")
print("-----similarity_matrix-----")
print(similarity_matrix)
print("---------------------------")

# plot_vectors()