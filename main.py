import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from gensim.models import Word2Vec
from collections import Counter
from tqdm import tqdm
import os


def build_vocab_and_cooccurrence_matrix(corpus):
    """
    从给定的语料中构建词汇表和稀疏共现矩阵。
    """
    vocab = Counter()
    
    for sentence in tqdm(corpus, desc="Building vocabulary..."):
        vocab.update(sentence)
    
    vocab = {word: i for i, word in enumerate(vocab)}
    
    rows, cols, data = [], [], []
    
    for sentence in tqdm(corpus, desc="Building Matrix..."):
        for i, token in enumerate(sentence):
            # 相邻5个token被认为是共同出现
            token_window = sentence[max(i - 5, 0):i] + sentence[i + 1:i + 6]
            for cooccur_word in token_window:
                if cooccur_word in vocab:
                    rows.append(vocab[token])
                    cols.append(vocab[cooccur_word])
                    data.append(1)
    
    cooccurrence_matrix = coo_matrix((data, (rows, cols)), shape=(len(vocab), len(vocab)), dtype=float)
    
    return vocab, cooccurrence_matrix


def cosine_similarity(vec1, vec2):
    """
    计算两个向量之间的余弦相似度。
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def load_corpus(filepath):
    """
    加载语料库
    """
    corpus = []
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            words = line.strip().split()
            corpus.append(words)
    return corpus


def train_word2vec(corpus):
    """
    使用Word2Vec训练词向量
    """
    model = Word2Vec(sentences=corpus, vector_size=50, window=2, min_count=1, workers=4, sg=1)
    return model


if __name__ == '__main__':
    corpus_filepath = "data/training.txt"
    corpus = load_corpus(corpus_filepath)
    
    vocab, cooccurrence_matrix = build_vocab_and_cooccurrence_matrix(corpus)
    print(cooccurrence_matrix.shape)
    # SVD
    K = 50
    u, s, vt = svds(cooccurrence_matrix, k=K)
    svd_word_vectors = u @ np.diag(s)
    # SGNS
    sgns_model = train_word2vec(corpus)
    
    # 非零奇异值的数量
    non_zero_singular_values = np.sum(s > 0)
    
    # 选取的奇异值之和与全部奇异值之和
    sum_selected_singular_values = np.sum(s)
    
    print(f"非零奇异值数量: {non_zero_singular_values}, \n"
          f"选取的奇异值数量: {K}, \n"
          f"选取的奇异值之和: {sum_selected_singular_values}, \n")
    
    # Prepare output directory
    os.makedirs("output", exist_ok=True)
    
    # Processing sim test and writing output
    with open("data/pku_sim_test.txt", "r", encoding="utf-8") as infile, open("output/2021213368.txt", "w",
                                                                              encoding="utf-8") as outfile:
        for line in infile:
            word1, word2 = line.strip().split('\t')
            
            # SVD Similarity
            if word1 in vocab and word2 in vocab:
                vec1 = svd_word_vectors[vocab[word1]]
                vec2 = svd_word_vectors[vocab[word2]]
                sim_svd = cosine_similarity(vec1, vec2)
            else:
                sim_svd = 0
            
            # SGNS Similarity
            if word1 in sgns_model.wv and word2 in sgns_model.wv:
                sim_sgns = 1 - sgns_model.wv.distance(word1, word2)
            else:
                sim_sgns = 0
            
            outfile.write(f"{line.strip()}\t{sim_svd}\t{sim_sgns}\n")

