from collections import defaultdict

def load_data(file_path):
    """加载训练/测试数据"""
    sentences = []
    tags = []
    current_sentence = []
    current_tags = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                word, tag = line.split('\t')
                tag = tag.split('-')[0]  # 提取标签前缀（如"S-CWS"转为"S"）
                current_sentence.append(word)
                current_tags.append(tag)
            else:
                if current_sentence:
                    sentences.append(current_sentence)
                    tags.append(current_tags)
                    current_sentence = []
                    current_tags = []
        # 处理文件末尾没有空行的情况
        if current_sentence:
            sentences.append(current_sentence)
            tags.append(current_tags)
    return sentences, tags

def build_vocab(sentences):
    """构建词汇表"""
    vocab = set()
    for sentence in sentences:
        for word in sentence:
            vocab.add(word)
    return list(vocab)