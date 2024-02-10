import sqlite3
import json
import numpy as np

def setLabel():
    conn = sqlite3.connect('./data/text_vectors.db')
    c = conn.cursor()
    try:
        c.execute("SELECT MAX(label) FROM text_vectors")
        # 最大のラベル値を取得
        last_label_result = c.fetchone()[0]
        label = last_label_result + 1
        conn.close()
    except:
        conn.close()
        return 1
    return label


def save_vectors(data_to_save):
    conn = sqlite3.connect('./data/text_vectors.db')
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS text_vectors
                 (text TEXT, WordClass TEXT, vector TEXT, label INTEGER)''')

    for item in data_to_save:
        vector_str = json.dumps(item['vector'])
        c.execute("INSERT INTO text_vectors (text, WordClass, vector, label) VALUES (?, ?, ?, ?)", 
                  (item['text'], item["WordClass"], vector_str, item['label']))
        
    conn.commit()
    conn.close()

def load_vectors():
    conn = sqlite3.connect('./data/text_vectors.db')
    c = conn.cursor()

    c.execute("SELECT text, WordClass, vector, label FROM text_vectors")
    data = c.fetchall()

    texts = [item[0] for item in data]
    WordClass = [item[1] for item in data]
    vectors = np.array([json.loads(item[2]) for item in data])
    labels = [item[3] for item in data]

    conn.close()
    return texts, WordClass, vectors, labels
