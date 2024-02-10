from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from database import load_vectors

def plot_vectors():
    texts, WordClass, vectors, labels = load_vectors()

    print("------------data------------")
    for text, WordClass, label in zip(texts, WordClass, labels):
        print(f"Text: {text}, Class: {WordClass}, Label: {label}")
    print("---------------------------")


    # PCAを適用して2次元に削減
    pca = PCA(n_components=2)
    transformed_data = pca.fit_transform(vectors)

    font_prop = FontProperties(fname='./font/gothic.ttf')

    # プロットを生成
    plt.figure(figsize=(8, 6))
    unique_labels = set(labels)
    for label in unique_labels:
        idx = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(transformed_data[idx, 0], transformed_data[idx, 1], label=label)
    for i, txt in enumerate(texts):
        plt.annotate(txt, (transformed_data[i, 0], transformed_data[i, 1]),
                    fontsize=9, fontproperties=font_prop)
    plt.title("PCA of Text Vectors", fontproperties=font_prop)
    plt.xlabel("Component 1", fontproperties=font_prop)
    plt.ylabel("Component 2", fontproperties=font_prop)
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_vectors()