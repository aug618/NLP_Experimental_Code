import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm
from model import BiLSTM_CRF  # 模型定义在model.py中

class NERDataset(Dataset):
    def __init__(self, file_path, word2idx, tag2idx, max_len=128):
        self.sentences = []
        self.labels = []
        self.max_len = max_len
        
        with open(file_path, 'r', encoding='utf-8') as f:
            words, tags = [], []
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) == 2:
                        words.append(parts[0])
                        tags.append(parts[1])
                else:
                    if words:
                        self.sentences.append(words)
                        self.labels.append(tags)
                        words, tags = [], []
        
        self.word2idx = word2idx
        self.tag2idx = tag2idx

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx][:self.max_len]
        label = self.labels[idx][:self.max_len]
        
        input_ids = [self.word2idx.get(w, self.word2idx['<UNK>']) for w in sentence]
        label_ids = [self.tag2idx[t] for t in label]
        
        input_mask = [1] * len(input_ids)
        padding_length = self.max_len - len(input_ids)
        
        input_ids += [self.word2idx['<PAD>']] * padding_length
        input_mask += [0] * padding_length
        label_ids += [self.tag2idx['O']] * padding_length
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(input_mask, dtype=torch.bool),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }

def build_vocab(files):
    words = set()
    tags = set()
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) == 2:
                        words.add(parts[0])
                        tags.add(parts[1])
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    tag2idx = {'O': 0}
    for w in words:
        word2idx[w] = len(word2idx)
    for t in sorted(tags):
        if t not in tag2idx:
            tag2idx[t] = len(tag2idx)
    return word2idx, tag2idx

def train_model(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        emissions = model(input_ids, attention_mask)
        loss = -model.crf(emissions, labels, mask=attention_mask, reduction='mean')
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_model(model, valid_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in valid_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            emissions = model(input_ids, attention_mask)
            predictions = model.decode(emissions, attention_mask.bool())
            
            for i in range(len(predictions)):
                valid_length = attention_mask[i].sum().item()
                y_true.extend(labels[i][:valid_length].cpu().numpy())
                y_pred.extend(predictions[i][:valid_length])
    
    report = classification_report(y_true, y_pred, output_dict=True)
    return report

if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MAX_LEN = 128
    BATCH_SIZE = 32
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 256
    LEARNING_RATE = 0.0005
    EPOCHS = 20

    files = ['data/youku/train.txt', 'data/youku/dev.txt', 'data/youku/test.txt']
    word2idx, tag2idx = build_vocab(files)
    idx2tag = {v: k for k, v in tag2idx.items()}

    train_dataset = NERDataset('data/youku/train.txt', word2idx, tag2idx, MAX_LEN)
    valid_dataset = NERDataset('data/youku/dev.txt', word2idx, tag2idx, MAX_LEN)
    test_dataset = NERDataset('data/youku/test.txt', word2idx, tag2idx, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = BiLSTM_CRF(
        vocab_size=len(word2idx),
        tagset_size=len(tag2idx),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_f1 = 0.0
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        train_loss = train_model(model, train_loader, optimizer, DEVICE)
        print(f"Train Loss: {train_loss:.4f}")
        
        valid_report = evaluate_model(model, valid_loader, DEVICE)
        print(f"Validation F1-Score: {valid_report['weighted avg']['f1-score']:.4f}")
        
        if valid_report['weighted avg']['f1-score'] > best_f1:
            best_f1 = valid_report['weighted avg']['f1-score']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'word2idx': word2idx,
                'tag2idx': tag2idx
            }, 'best_model.pth')
            print("Saved best model")

    checkpoint = torch.load('best_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_report = evaluate_model(model, test_loader, DEVICE)
    print("Test Report:")
    print(f"Precision: {test_report['weighted avg']['precision']:.4f}")
    print(f"Recall: {test_report['weighted avg']['recall']:.4f}")
    print(f"F1-Score: {test_report['weighted avg']['f1-score']:.4f}")