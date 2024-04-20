import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def initialize_vocab(vocab_size, vector_size):
    # 初始化词向量
    word_vectors = np.random.uniform(-0.8, 0.8, (vocab_size, vector_size))
    return word_vectors


def train_skipgram(word_vectors, word_indices, context_indices, lr=0.01, epochs=100, negative_samples=5):
    """
    训练Skip-Gram模型
    :param word_vectors: 初始化的词向量
    :param word_indices: 中心词索引数组
    :param context_indices: 上下文词索引数组
    :param lr: 学习率
    :param epochs: 训练轮数
    :param negative_samples: 每个正样本的负样本数量
    """
    vocab_size, vector_size = word_vectors.shape
    for epoch in range(epochs):
        total_loss = 0
        for word_idx, context_idx in zip(word_indices, context_indices):
            # 获取中心词和上下文词的向量
            v_w = word_vectors[word_idx]
            v_c = word_vectors[context_idx]
            
            # 正样本的损失和梯度更新
            score = np.dot(v_w, v_c)
            predicted = sigmoid(score)
            pos_loss = -np.log(predicted)
            total_loss += pos_loss
            
            # 正样本的梯度
            grad_pos = (predicted - 1) * v_c
            word_vectors[word_idx] -= lr * grad_pos
            word_vectors[context_idx] -= lr * (predicted - 1) * v_w
            
            # 负样本的处理
            for _ in range(negative_samples):
                neg_sample = np.random.randint(0, vocab_size)
                v_n = word_vectors[neg_sample]
                score_neg = np.dot(v_w, v_n)
                predicted_neg = sigmoid(score_neg)
                neg_loss = -np.log(1 - predicted_neg)
                total_loss += neg_loss
                
                # 负样本的梯度
                word_vectors[word_idx] -= lr * predicted_neg * v_n
                word_vectors[neg_sample] -= lr * predicted_neg * v_w
        
        print(f"Epoch {epoch + 1}, Loss: {total_loss}")


# 示例用法
vocab_size = 10  # 假设我们有一个由10个词构成的词汇表
vector_size = 5  # 每个词的向量维度设置为5

# 初始化词向量
word_vectors = initialize_vocab(vocab_size, vector_size)

# 示例中心词和上下文词索引
word_indices = np.array([0, 1, 2, 3])  # 假设这是中心词索引
context_indices = np.array([1, 2, 3, 4])  # 对应的上下文词索引

# 训练模型
train_skipgram(word_vectors, word_indices, context_indices, lr=0.01, epochs=1000, negative_samples=2)
