import sqlite3
import json
import numpy as np

def save_vectors(data_to_save):
    # データベース接続を開く（データベースファイルがない場合は新たに作成される）
    conn = sqlite3.connect('./data/text_vectors.db')
    c = conn.cursor()

    # テーブルの作成（既に存在する場合はスキップ）
    c.execute('''CREATE TABLE IF NOT EXISTS text_vectors
                 (text TEXT, vector TEXT)''')

    # データを挿入
    for item in data_to_save:
        # ベクトルをJSON形式の文字列に変換して保存
        vector_str = json.dumps(item['vector'])
        c.execute("INSERT INTO text_vectors (text, vector) VALUES (?, ?)", (item['text'], vector_str))

    # 変更をコミットし、接続を閉じる
    conn.commit()
    conn.close()

def load_vectors():
    # データベース接続を開く
    conn = sqlite3.connect('./data/text_vectors.db')
    c = conn.cursor()

    # データを読み込む
    c.execute("SELECT text, vector FROM text_vectors")
    data = c.fetchall()

    # データを処理
    texts = [item[0] for item in data]
    vectors = np.array([json.loads(item[1]) for item in data])

    # データベース接続を閉じる
    conn.close()
    
    return texts, vectors
