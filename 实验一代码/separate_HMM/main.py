import argparse
from data_utils import load_data, build_vocab
from model import build_states, init_params, save_model, load_model
from decode import viterbi, tags_to_segments
from evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser(description='HMM中文分词')
    parser.add_argument('--train', type=str, help='训练数据路径')
    parser.add_argument('--test', type=str, help='测试数据路径')
    parser.add_argument('--model', type=str, default='hmm_model.pkl', help='模型保存路径')
    args = parser.parse_args()

    # 训练模式
    if args.train:
        print("训练模式...")
        train_sentences, train_tags = load_data(args.train)
        vocab = build_vocab(train_sentences)
        states = build_states()
        initial_probs, transition_probs, emission_probs = init_params(
            train_sentences, train_tags, states, vocab
        )
        save_model(args.model, initial_probs, transition_probs, emission_probs, vocab)
        print(f"模型已保存至 {args.model}")

    # 测试模式
    if args.test:
        print("\n测试模式...")
        initial_probs, transition_probs, emission_probs, vocab = load_model(args.model)
        states = build_states()
        test_sentences, test_tags = load_data(args.test)
        
        # 替换未登录词
        processed_test = []
        for sent in test_sentences:
            processed_test.append([word if word in vocab else 'UNK' for word in sent])
        
        precision, recall, f1 = evaluate_model(
            processed_test, test_tags, states, initial_probs, transition_probs, emission_probs
        )
        print(f"测试结果 - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # 交互式分词演示
    if not args.train and not args.test:
        print("\n交互模式（示例）...")
        initial_probs, transition_probs, emission_probs, vocab = load_model(args.model)
        states = build_states()
        
        while True:
            sentence = input("\n输入句子（输入q退出）: ").strip()
            if sentence.lower() == 'q':
                break
            # 处理未登录词
            processed_sent = [word if word in vocab else 'UNK' for word in sentence]
            tags = viterbi(processed_sent, states, initial_probs, transition_probs, emission_probs)
            segments = tags_to_segments(sentence, tags)  # 使用原始字符
            print("分词结果:", '/ '.join(segments))

if __name__ == '__main__':
    main()