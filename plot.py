from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from database import load_vectors

def plot_vectors():
    # load_vectors()関数からテキストとベクトルを読み込む
    texts, vectors = load_vectors()

    # PCAを適用して2次元に削減
    pca = PCA(n_components=2)
    transformed_data = pca.fit_transform(vectors)

    # プロット時に日本語フォントを指定
    font_prop = FontProperties(fname='./font/gothic.ttf')  # 適切なフォントパスを指定

    # プロットを生成
    plt.figure(figsize=(8, 6))
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1])
    for i, txt in enumerate(texts):
        plt.annotate(txt, (transformed_data[i, 0], transformed_data[i, 1]),
                    fontsize=9, fontproperties=font_prop)
    plt.title("PCA of Text Vectors", fontproperties=font_prop)
    plt.xlabel("Component 1", fontproperties=font_prop)
    plt.ylabel("Component 2", fontproperties=font_prop)
    plt.grid(True)
    plt.show()
