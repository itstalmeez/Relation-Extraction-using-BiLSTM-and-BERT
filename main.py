  pip install torch transformers torchtext
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel as TransformersBertModel, BertTokenizer
from transformers import BertModel, BertTokenizer
# Load data from JSON files
import json

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data
train_data = load_data("/content/train_full.json")
test_data = load_data("/content/test_full.json")

# Tokenize and preprocess data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_data(data):
    sentences = [item['sentence'] for item in data]

    # Convert string labels to a list of integers using a mapping
    relation_mapping = {
        "Component-Whole(e2,e1)": 0,
        "Other": 1,
        "Instrument-Agency(e2,e1)": 2,
        "Member-Collection(e1,e2)": 3,
        "Cause-Effect(e2,e1)": 4  # Add any new relations here
    }

    # Handle unknown relation types
    relations = [relation_mapping.get(item['relation'], -1) for item in data]

    # Tokenize sentences and convert to PyTorch tensors
    tokenized_sentences = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=128)
    tokenized_sentences = {key: value.squeeze(dim=1) for key, value in tokenized_sentences.items()}  # Squeeze to remove extra dimension

    # Convert relations to PyTorch tensor
    labels = torch.tensor(relations)

    return tokenized_sentences, labels

train_inputs, train_labels = preprocess_data(train_data)
test_inputs, test_labels = preprocess_data(test_data)
# BiLSTM Model
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        return out
      # MyBertModel
class MyBertModel(nn.Module):
    def __init__(self, num_classes):
        super(MyBertModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooler_output = outputs['pooler_output']
        out = self.fc(pooler_output)
        return out


# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BiLSTM model training
input_size = len(tokenizer.vocab)
hidden_size = 128
num_layers = 2
num_classes = len(set(train_labels.numpy()))
bi_lstm_model = BiLSTMModel(input_size, hidden_size, num_layers, num_classes).to(device)
# MyBertModel training
num_classes = len(set(train_labels.numpy()))
bert_model = MyBertModel(num_classes).to(device)
# DataLoaders
batch_size = 32
train_dataset = torch.utils.data.TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# Training loop for BiLSTM
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(bi_lstm_model.parameters(), lr=0.001)

epochs = 5
for epoch in range(epochs):
    for inputs, attention_mask, labels in train_loader:
        inputs, attention_mask, labels = inputs.to(device), attention_mask.to(device), labels.to(device)

        # Filter out batches with unknown labels (-1)
        mask = (labels != -1)
        inputs, attention_mask, labels = inputs[mask], attention_mask[mask], labels[mask]

        if inputs.size(0) == 0:
            # Skip batch if all labels are unknown
            continue

        optimizer.zero_grad()

        outputs = bi_lstm_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
unknown_label = -1

# Tokenize and preprocess data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_data(data):
    sentences = [item['sentence'] for item in data]

    # Convert string labels to a list of integers using a mapping
    relation_mapping = {
        "Component-Whole(e2,e1)": 0,
        "Other": 1,
        "Instrument-Agency(e2,e1)": 2,
        "Member-Collection(e1,e2)": 3,
    }

    # Assign a specific value for unknown labels
    relations = [relation_mapping.get(item['relation'], unknown_label) for item in data]

    # Tokenize sentences and convert to PyTorch tensors
    tokenized_sentences = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=128)
    tokenized_sentences = {key: value.squeeze(dim=1) for key, value in tokenized_sentences.items()}  # Squeeze to remove extra dimension

    # Convert relations to PyTorch tensor
    labels = torch.tensor(relations)

    return tokenized_sentences, labels

train_inputs, train_labels = preprocess_data(train_data)
test_inputs, test_labels = preprocess_data(test_data)
bi_lstm_model = BiLSTMModel(input_size, hidden_size, num_layers, num_classes).to(device)

# Training loop for BiLSTM
criterion = nn.CrossEntropyLoss(ignore_index=unknown_label)  # Set ignore_index to handle -1 labels
optimizer = torch.optim.Adam(bi_lstm_model.parameters(), lr=0.001)

epochs = 5
for epoch in range(epochs):
    for inputs, attention_mask, labels in train_loader:
        inputs, attention_mask, labels = inputs.to(device), attention_mask.to(device), labels.to(device)

        # Filter out batches with unknown labels
        mask = (labels != unknown_label)
        inputs, attention_mask, labels = inputs[mask], attention_mask[mask], labels[mask]

        if inputs.size(0) == 0:
            # Skip batch if all labels are unknown
            continue

        optimizer.zero_grad()

        outputs = bi_lstm_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
