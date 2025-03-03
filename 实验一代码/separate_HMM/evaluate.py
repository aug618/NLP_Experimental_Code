from decode import viterbi
from decode import tags_to_segments
def evaluate_model(sentences, true_tags, states, initial_probs, transition_probs, emission_probs):
    """模型评估（精确率、召回率、F1）"""
    correct = 0
    total_pred = 0
    total_true = 0
    
    for sent, true in zip(sentences, true_tags):
        pred = viterbi(sent, states, initial_probs, transition_probs, emission_probs)
        true_segs = tags_to_segments(sent, true)
        pred_segs = tags_to_segments(sent, pred)
        
        # 转换为字符位置比较
        true_set = set()
        pos = 0
        for seg in true_segs:
            true_set.add((pos, pos+len(seg)))
            pos += len(seg)
        
        pred_set = set()
        pos = 0
        for seg in pred_segs:
            pred_set.add((pos, pos+len(seg)))
            pos += len(seg)
        
        total_true += len(true_set)
        total_pred += len(pred_set)
        correct += len(true_set & pred_set)
    
    precision = correct / total_pred if total_pred > 0 else 0
    recall = correct / total_true if total_true > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1