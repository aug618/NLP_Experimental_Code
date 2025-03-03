import torch
from model import BiLSTM_CRF

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载保存的模型和字典
checkpoint = torch.load('best_model.pth', weights_only="True",map_location=device)
word2idx = checkpoint['word2idx']
tag2idx = checkpoint['tag2idx']
idx2tag = {v: k for k, v in tag2idx.items()}

# 初始化模型
vocab_size = len(word2idx)
tagset_size = len(tag2idx)
model = BiLSTM_CRF(vocab_size, tagset_size)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

def predict(text, model, word2idx, idx2tag, max_len=128):
    words = list(text)
    input_ids = [word2idx.get(w, word2idx['<UNK>']) for w in words]
    input_mask = [1] * len(input_ids)
    
    padding_length = max_len - len(input_ids)
    input_ids += [word2idx['<PAD>']] * padding_length
    input_mask += [0] * padding_length
    
    with torch.no_grad():
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        mask_tensor = torch.tensor([input_mask], dtype=torch.bool).to(device)
        emissions = model(input_tensor, mask_tensor)
        predictions = model.decode(emissions, mask_tensor)[0]
    
    valid_predictions = predictions[:len(words)]
    return [(word, idx2tag[tag]) for word, tag in zip(words, valid_predictions)]

if __name__ == "__main__":
    print("模型已加载，输入文本进行命名实体识别（输入 'exit' 退出）:")
    while True:
        text = input("请输入文本: ")
        if text.lower() == 'exit':
            break
        result = predict(text, model, word2idx, idx2tag)
        print("预测结果：")
        for word, tag in result:
            print(f"{word}\t{tag}")
        print()  # 空行分隔