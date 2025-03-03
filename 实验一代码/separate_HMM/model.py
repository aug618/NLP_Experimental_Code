import pickle
from collections import defaultdict
from math import log
from data_utils import build_vocab

def build_states():
    """定义HMM状态集合"""
    return ['B', 'I', 'E', 'S']

def init_params(train_sentences, train_tags, states, vocab):
    """初始化HMM参数（使用加一平滑）"""
    # 初始化计数器
    state_counts = defaultdict(int)
    start_state_counts = defaultdict(int)
    transition_counts = defaultdict(lambda: defaultdict(int))
    emission_counts = defaultdict(lambda: defaultdict(int))

    # 加一平滑
    for state in states:
        state_counts[state] += 1
        for word in vocab:
            emission_counts[state][word] += 1
        for next_state in states:
            transition_counts[state][next_state] += 1

    # 统计实际数据
    for sentence, tag_seq in zip(train_sentences, train_tags):
        if not sentence:
            continue
        start_state = tag_seq[0]
        start_state_counts[start_state] += 1
        for prev_state, curr_state in zip(tag_seq[:-1], tag_seq[1:]):
            transition_counts[prev_state][curr_state] += 1
        for word, state in zip(sentence, tag_seq):
            emission_counts[state][word] += 1

    # 计算初始概率（对数空间）
    initial_probs = {}
    total_start = sum(start_state_counts.values())
    for state in states:
        initial_probs[state] = log((start_state_counts[state] + 1e-10) / (total_start + 1e-10))

    # 转移概率（对数空间）
    transition_probs = defaultdict(dict)
    for prev_state in states:
        total = sum(transition_counts[prev_state].values())
        for curr_state in states:
            transition_probs[prev_state][curr_state] = log(
                (transition_counts[prev_state][curr_state] + 1e-10) / (total + 1e-10)
            )

    # 发射概率（对数空间）
    emission_probs = defaultdict(dict)
    for state in states:
        total = sum(emission_counts[state].values())
        for word in vocab:
            emission_probs[state][word] = log(
                (emission_counts[state][word] + 1e-10) / (total + 1e-10)
            )
        emission_probs[state]['UNK'] = log(1e-10 / (total + 1e-10))  # 未登录词处理

    return initial_probs, transition_probs, emission_probs

def save_model(model_path, initial_probs, transition_probs, emission_probs, vocab):
    """保存模型到文件"""
    model_data = {
        'initial_probs': initial_probs,
        'transition_probs': dict(transition_probs),
        'emission_probs': dict(emission_probs),
        'vocab': vocab
    }
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

def load_model(model_path):
    """从文件加载模型"""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # 恢复defaultdict结构
    transition_probs = defaultdict(dict)
    for prev_state, probs in model_data['transition_probs'].items():
        transition_probs[prev_state].update(probs)
    
    emission_probs = defaultdict(dict)
    for state, probs in model_data['emission_probs'].items():
        emission_probs[state].update(probs)
    
    return (
        model_data['initial_probs'],
        transition_probs,
        emission_probs,
        model_data['vocab']
    )