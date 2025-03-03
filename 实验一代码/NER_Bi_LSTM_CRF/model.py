import torch.nn as nn
from torchcrf import CRF

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim // 2, 
                             num_layers=2, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, inputs, masks):
        embeddings = self.embedding(inputs)
        lstm_out, _ = self.bilstm(embeddings)
        emissions = self.hidden2tag(lstm_out)
        return emissions

    def decode(self, emissions, masks):
        return self.crf.decode(emissions, masks)