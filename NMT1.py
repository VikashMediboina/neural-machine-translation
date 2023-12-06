import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import pickle
from nltk.translate.bleu_score import corpus_bleu
import random
from pyvi import ViTokenizer, ViPosTagger
from nltk.translate.bleu_score import sentence_bleu
import spacy
import torchtext
import sys
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def remove_punctuation_from_tokenized_data(tokenized_data):
    return [[token for token in sentence if token not in [',', '.']] for sentence in tokenized_data]
# vocab_en='data/en_vocab.pkl'
# vocab_de='data/vi_vocab.pkl'

vocab_en='english_vocab.pkl'
vocab_de='german_vocab.pkl'


with open(vocab_en, 'rb') as f:
    eng_vocab = pickle.load(f)

# Load viman tokenized sentences from the pickle file
with open(vocab_de, 'rb') as f:
    vi_vocab = pickle.load(f)

if '<unk>' not in eng_vocab:
    eng_vocab.insert_token('<unk>', 0)  # Adjust the index if needed

if '<unk>' not in vi_vocab:
    vi_vocab.insert_token('<unk>', 0)  # Adjust the index if needed

# train_en_tokenize_file_name='data/train_en_toknized.pkl'
# train_vi_tokenize_file_name='data/train_vi_toknized.pkl'
# val_en_tokenize_file_name='data/test_en_toknized.pkl'
# val_vi_tokenize_file_name='data/test_vi_toknized.pkl'


train_en_tokenize_file_name='english_tokenized.pkl'
train_vi_tokenize_file_name='german_tokenized.pkl'
val_en_tokenize_file_name='val_english_tokenized.pkl'
val_vi_tokenize_file_name='val_german_tokenized.pkl'

def read_all_data_from_pickle(file_name):
    data = []
    try:
        with open(file_name, 'rb') as f:
            # while True:
                # try:
                    data.extend(pickle.load(f))
                # except EOFError:
                #     break
    except FileNotFoundError:
        print(f"File not found: {file_name}")
    return data

def preProcess( english_tokenized_file_name,vi_tokenized_file_name,eng_vocab,ger_vocab):
    english_tokenized=read_all_data_from_pickle(english_tokenized_file_name)
    vi_tokenized=read_all_data_from_pickle(vi_tokenized_file_name)

    english_tokenized = remove_punctuation_from_tokenized_data(english_tokenized)
    vi_tokenized = remove_punctuation_from_tokenized_data(vi_tokenized)
    # Convert words to indices
    english_indices = [torch.tensor([eng_vocab[word] if word in eng_vocab else eng_vocab['<unk>'] for word in sentence], dtype=torch.long) for sentence in english_tokenized]
    vi_indices = [torch.tensor([ger_vocab[word] if word in ger_vocab else eng_vocab['<unk>'] for word in sentence], dtype=torch.long) for sentence in vi_tokenized]

    # Pad sequences to the same length
    max_len = max(max(len(seq) for seq in english_indices), max(len(seq) for seq in vi_indices))
    english_padded = pad_sequence([torch.cat([seq, torch.zeros(max_len - len(seq))], dim=0) for seq in english_indices], batch_first=True)
    vi_padded = pad_sequence([torch.cat([seq, torch.zeros(max_len - len(seq))], dim=0) for seq in vi_indices], batch_first=True)

    return english_padded,vi_padded, max_len
train_english_padded,train_vi_padded,max_len = preProcess(train_en_tokenize_file_name,train_vi_tokenize_file_name,eng_vocab,vi_vocab)
val_english_padded,val_vi_padded,max_len = preProcess(val_en_tokenize_file_name,val_vi_tokenize_file_name,eng_vocab,vi_vocab)
train_english_padded = train_english_padded.to(device)
train_vi_padded = train_vi_padded.to(device)
val_english_padded = val_english_padded.to(device)
val_vi_padded = val_vi_padded.to(device)
# Define the dataset
class TranslationDataset(Dataset):
    def __init__(self, english_data, vi_data):
        self.english_data = english_data
        self.vi_data = vi_data

    def __len__(self):
        return len(self.english_data)

    def __getitem__(self, idx):
        return self.english_data[idx], self.vi_data[idx]


# Create DataLoader
train_dataset = TranslationDataset(train_english_padded[:10], train_vi_padded[:10])
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Create DataLoader
val_dataset = TranslationDataset(val_english_padded[:10], val_vi_padded[:10])
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout):
        super(EncoderLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout)

    def forward(self, x):
        # x shape: (seq_length, N)

        embedding = self.dropout(self.embedding(x.long()))
        # embedding shape: (seq_length, N, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding)
        return hidden, cell

class DecoderLSTM(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, dropout):
        super(DecoderLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.output_size=output_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        # x shape: (N) but we want (1, N)
        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x.long()))
        # embedding shape: (1, N, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        # outputs shape: (1, N, hidden_size)

        predictions = self.fc(outputs)
        # predictions shape: (1, N, length_of_vocab)

        predictions = predictions.squeeze(0)
        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_forcing_ratio = 0.5):
        # source shape: (src_len, N)
        # target shape: (trg_len, N)

        trg_len, N = target.shape
        trg_vocab_size = self.decoder.output_size

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, N, trg_vocab_size).to(device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(source)

        # first input to the decoder is the <sos> tokens
        x = target[0]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(x, hidden, cell)

            outputs[t] = output

            # decide if we will use teacher forcing or not
            best_guess = output.argmax(1)
            x = target[t] if random.random() < teacher_forcing_ratio else best_guess
        
        return outputs
# Training loop
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    
    epoch_loss = 0
    for i, (src, trg) in enumerate(iterator):
        
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()
        
        output = model(src, trg)
       
        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]
        
        output_dim = output.shape[-1]
       
        
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1).long()
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)
def evaluate(model, iterator, criterion,ger_vocab):
    model.eval()
    
    epoch_loss = 0
    predictions, targets = [], []
    with torch.no_grad():
        for _, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)

            output = model(src, trg, 0)  # turn off teacher forcing
            

            # Convert output to token indices
            output_indices = output.argmax(2)  # Choose the word with highest probability
            output_sentences = tensor_to_sentence(output_indices[:, 1:], ger_vocab)
            predictions.extend(output_sentences)

            # Convert trg to token strings, skipping <sos> token
            trg_sentences = tensor_to_sentence(trg[:, 1:], ger_vocab)
            targets.extend([[sent] for sent in trg_sentences])  # Wrap each sentence in another list


            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1).long()

            loss = criterion(output, trg)
            epoch_loss += loss.item()
    bleu=corpus_bleu( targets,predictions)
    return epoch_loss / len(iterator) , bleu
def tensor_to_sentence(tensor, vocab, pad_index=0, eos_index=None, sos_index=None):
    itos = vocab.get_itos()
    sentences = []
    
    for i in range(tensor.size(0)):  # Loop over each item in the batch
        sentence = []
        for idx in tensor[i]:
            if idx == pad_index or idx == eos_index or idx == sos_index:
                continue  # Skip pad, eos, and sos tokens
            sentence.append(itos[int(idx.item())])
        sentences.append(sentence)
    return sentences
def training():
    # Training settings
    # Initialize models
    INPUT_DIM = len(eng_vocab)  # Assuming eng_vocab is your English vocabulary
    OUTPUT_DIM = len(vi_vocab)  # Assuming ger_vocab is your German vocabulary
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 1024
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    learning_rate=0.001
    N_EPOCHS = 50
    CLIP = 1
    start_epoch=0
    
    enc = EncoderLSTM(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT).to(device)
    dec = DecoderLSTM(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, OUTPUT_DIM, N_LAYERS, DEC_DROPOUT).to(device)
    model = Seq2Seq(enc, dec).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0) 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')
    for epoch in range(start_epoch,N_EPOCHS):
        train_loss = train(model, train_dataloader, optimizer, criterion, CLIP)
        valid_loss, val_bleu = evaluate(model, val_dataloader, criterion,vi_vocab)
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            # Save the model
            torch.save(model.state_dict(), 'data/eng_to_vi_translation_model.pth')
            print("Saved Best Model")
        print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Val. Loss: {valid_loss:.3f}, Valid Bleu: {val_bleu:.5f}')


    # Save the model
    torch.save(model.state_dict(), 'data/eng_to_vi_translation_model_end.pth')


def load_model(model_path):
    INPUT_DIM = len(eng_vocab)  # Assuming eng_vocab is your English vocabulary
    OUTPUT_DIM = len(vi_vocab)  # Assuming ger_vocab is your German vocabulary
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 1024
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    learning_rate=0.001
    enc = EncoderLSTM(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT).to(device)
    dec = DecoderLSTM(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, OUTPUT_DIM, N_LAYERS, DEC_DROPOUT).to(device)
    model = Seq2Seq(enc, dec).to(device)
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)
    return model

def evaluate(model, iterator, vi_vocab):
    model.eval()
    
    epoch_loss = 0
    predictions, targets = [], []
    with torch.no_grad():
        for _, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)

            output = model(src, trg, 0)  # Turn off teacher forcing

            # Convert output to token indices
            output_indices = output.argmax(2)  # Choose the word with the highest probability

            # Skip the first output of the sequence, which is <sos> token
            output_sentences = tensor_to_sentence(output_indices[:, 1:], vi_vocab)
            predictions.extend([' '.join(sent) for sent in output_sentences])

            # Convert target to token strings, skipping <sos> token
            trg_sentences = tensor_to_sentence(trg[:, 1:], vi_vocab)
            targets.extend([[' '.join(sent)] for sent in trg_sentences])  # Wrap each sentence in another list
            print("Original German Sentence:", targets[0][0])
            print("Translated Sentence:", predictions[0])
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)


    bleu_score = corpus_bleu(targets, predictions)
    print("BLEU Score:", bleu_score)



def main():
    if len(sys.argv) != 2:
        print("Usage: python NMT.py [train/test]")
        return

    if sys.argv[1] == "train":
        training()
    elif sys.argv[1] == "test":
        test_en_tokenize_file_name='test_english_tokenized.pkl'
        test_vi_tokenize_file_name='test_german_tokenized.pkl'
        test_english_padded,test_vi_padded,max_len = preProcess(test_en_tokenize_file_name,test_vi_tokenize_file_name,eng_vocab,vi_vocab)
        test_english_padded = test_english_padded.to(device)
        test_vi_padded = test_vi_padded.to(device)
        test_dataset = TranslationDataset(test_english_padded[:1], test_vi_padded[:1])
        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
        test_model = load_model('model/eng_to_ger_translation_model.pth')
        evaluate(test_model, test_dataloader, vi_vocab)
    else:
        print("Invalid command")

# Main script execution
if __name__ == "__main__":
    main()