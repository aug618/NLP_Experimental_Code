import numpy as np

def viterbi(sentence, states, initial_probs, transition_probs, emission_probs):
    """维特比算法解码"""
    T = len(sentence)
    state_index = {s: i for i, s in enumerate(states)}
    
    # 初始化对数概率表
    viterbi_table = np.full((len(states), T), -np.inf)
    backpointer = np.zeros((len(states), T), dtype=int)
    
    # 处理第一个词
    for s_idx, state in enumerate(states):
        word = sentence[0]
        emission = emission_probs[state].get(word, emission_probs[state]['UNK'])
        viterbi_table[s_idx, 0] = initial_probs[state] + emission
    
    # 填充表格
    for t in range(1, T):
        for s_idx, curr_state in enumerate(states):
            max_log_prob = -np.inf
            best_prev_idx = 0
            for prev_idx, prev_state in enumerate(states):
                log_prob = viterbi_table[prev_idx, t-1] + \
                           transition_probs[prev_state].get(curr_state, -np.inf) + \
                           emission_probs[curr_state].get(sentence[t], emission_probs[curr_state]['UNK'])
                if log_prob > max_log_prob:
                    max_log_prob = log_prob
                    best_prev_idx = prev_idx
            viterbi_table[s_idx, t] = max_log_prob
            backpointer[s_idx, t] = best_prev_idx
    
    # 回溯路径
    best_path = []
    best_last_idx = np.argmax(viterbi_table[:, -1])
    best_path.append(states[best_last_idx])
    
    for t in range(T-1, 0, -1):
        best_last_idx = backpointer[best_last_idx, t]
        best_path.insert(0, states[best_last_idx])
    
    return best_path

def tags_to_segments(sentence, tags):
    """将标签序列转换为分词结果"""
    segments = []
    current_segment = []
    for word, tag in zip(sentence, tags):
        if tag == 'B':
            if current_segment:
                segments.append(''.join(current_segment))
            current_segment = [word]
        elif tag == 'I':
            current_segment.append(word)
        elif tag == 'E':
            current_segment.append(word)
            segments.append(''.join(current_segment))
            current_segment = []
        elif tag == 'S':
            if current_segment:
                segments.append(''.join(current_segment))
            segments.append(word)
            current_segment = []
    # 处理剩余部分
    if current_segment:
        segments.append(''.join(current_segment))
    return segments