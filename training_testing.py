import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import random

import config


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

    def forward(self, x):
        # x shape: (seq_length, N) where N is batch size

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (seq_length, N, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding)
        # outputs shape: (seq_length, N, hidden_size)

        return hidden, cell


class Decoder(nn.Module):
    def __init__(
        self, input_size, embedding_size, hidden_size, output_size, num_layers, p
    ):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        # x shape: (N) where N is for batch size, we want it to be (1, N), seq_length
        # is 1 here because we are sending in a single word and not a sentence
        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (1, N, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        # outputs shape: (1, N, hidden_size)

        predictions = self.fc(outputs)

        # predictions shape: (1, N, length_target_vocabulary) to send it to
        # loss function we want it to be (N, length_target_vocabulary) so we're
        # just gonna remove the first dim
        predictions = predictions.squeeze(0)

        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = config.VOCAB_SIZE

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)

        hidden, cell = self.encoder(source)

        # Grab the first input to the Decoder which will be <SOS> token
        x = target[0]

        for t in range(1, target_len):
            # Use previous hidden, cell as context from encoder at start
            output, hidden, cell = self.decoder(x, hidden, cell)

            # Store next output prediction
            outputs[t] = output

            # Get the best word the Decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)

            # With probability of teacher_force_ratio we take the actual next word
            # otherwise we take the word that the Decoder predicted it to be.
            # Teacher Forcing is used so that the model gets used to seeing
            # similar inputs at training and testing time, if teacher forcing is 1
            # then inputs at test time might be completely different than what the
            # network is used to. This was a long comment.
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs


class Model(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class, hidden_dim):
        super(Model, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_class)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (last_hidden_state, _) = self.lstm(embedded)
        linear_input = last_hidden_state[-1]
        out = self.fc(linear_input)

        return out


def train(train_len, optimizer, model, criterion, epoch_loss, device, dataloaders, semi_supervised=0):
    # Train the model

    train_loss = torch.zeros(1, dtype=torch.float).to(device)
    train_acc = torch.zeros(1, dtype=torch.float).to(device)

    data = dataloaders[config.FILE_TRAINING]

    for sample in data:
        optimizer.zero_grad()

        if semi_supervised == config.SEMI_SUPERVISED_PHASE_1:
            y, x = sample['eleventh word'].to(device), sample['10 words'].to(device)
        else:
            y, x = sample['category'].to(device), sample['10 words'].to(device)

        output = model(x, y)

        output = output[1:].reshape(-1, output.shape[2])
        y = y[1:].reshape(-1)

        loss = criterion(output, y)
        train_loss += loss.item()
        loss.backward()

        # Clip to avoid exploding gradient issues, makes sure grads are
        # within a healthy range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        epoch_loss.append(loss.item())
        optimizer.step()
        train_acc += (output.argmax(1) == y).sum().item()

    return train_loss / train_len, train_acc / train_len


def test(dataloader, data_len, batch_size, criterion, model, device, epoch_loss=None,
         semi_supervised=config.SUPERVISED):
    # Test the model

    val_loss = torch.zeros(1, dtype=torch.float).to(device)
    val_acc = torch.zeros(1, dtype=torch.float).to(device)

    for sample in dataloader:

        if semi_supervised == config.SEMI_SUPERVISED_PHASE_1:
            y, x = sample['eleventh word'].to(device), sample['10 words'].to(device)
        else:
            y, x = sample['category'].to(device), sample['10 words'].to(device)

        with torch.no_grad():
            output = model(x)
            loss = criterion(output, y)
            val_loss += loss.item()
            val_acc += (output.argmax(1) == y).sum().item()

            if epoch_loss is not None:
                epoch_loss.append(loss.item())

    return val_loss / data_len, val_acc / data_len


def my_plot(epochs, loss_train, loss_val, semi_supervised=0):
    # Save/show graph of training/validation loss

    plt.plot(epochs, loss_train, label='Training')
    plt.plot(epochs, loss_val, label='Validation')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    if semi_supervised == 0:  # Supervised
        plt.savefig('outputs/graphs/supervised_training/loss_graph.png')
    elif semi_supervised == 1:  # Semi-supervised phase 1
        plt.savefig('outputs/graphs/semi_supervised_training/phase_one/loss_graph.png')
    else:  # Semi-supervised phase 2
        plt.savefig('outputs/graphs/semi_supervised_training/phase_two/loss_graph.png')

    plt.show()


def run(device, dataset_sizes, dataloaders, num_classes, semi_supervised, num_epochs, model=None):
    plt.ion()  # interactive mode

    if model is None:
        encoder_net = Encoder(
            config.VOCAB_SIZE, config.EMBED_DIM, config.HIDDEN_DIM, config.NUM_LAYERS, config.ENC_DROPOUT
        ).to(device)

        if semi_supervised == config.SEMI_SUPERVISED_PHASE_1:
            decoder_net = Decoder(
                config.VOCAB_SIZE,
                config.EMBED_DIM,
                config.HIDDEN_DIM,
                config.VOCAB_SIZE,
                config.NUM_LAYERS,
                config.DEC_DROPOUT,
            ).to(device)
        else:
            decoder_net = Decoder(
                config.VOCAB_SIZE,
                config.EMBED_DIM,
                config.HIDDEN_DIM,
                config.SUPERVISED_NUM_CLASSES,
                config.NUM_LAYERS,
                config.DEC_DROPOUT,
            ).to(device)

        model = Seq2Seq(encoder_net, decoder_net, device).to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    train_len = dataset_sizes[config.FILE_TRAINING]

    loss_vals_train = []
    loss_vals_val = []

    if not semi_supervised:
        print("Supervised training:\n")
    elif semi_supervised == 1:
        print("\nSemi-supervised phase 1 training:\n")
    else:
        print("\nSemi-supervised phase 2 training:\n")

    for epoch in range(num_epochs):
        start_time = time.time()

        epoch_loss_train = []
        epoch_loss_val = []

        train_loss, train_acc = train(train_len, optimizer, model, criterion, epoch_loss_train, device,
                                      dataloaders, semi_supervised=semi_supervised)

        loss_vals_train.append(sum(epoch_loss_train) / len(epoch_loss_train))

        valid_loss, valid_acc = test(dataloaders[config.FILE_VALIDATION], dataset_sizes[config.FILE_VALIDATION],
                                     config.BATCH_SIZE,
                                     criterion, model, device, epoch_loss_val,
                                     semi_supervised=semi_supervised)

        loss_vals_val.append(sum(epoch_loss_val) / len(epoch_loss_val))

        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60

        print('Epoch: %d' % (epoch + 1), " | time in %d minutes, %d seconds" % (mins, secs))
        print(f'\tLoss: {train_loss.item():.4f}(train)\t|\tAcc: {train_acc.item() * 100:.1f}%(train)')
        print(f'\tLoss: {valid_loss.item():.4f}(valid)\t|\tAcc: {valid_acc.item() * 100:.1f}%(valid)')

        print('Checking the results of test dataset...')

        test_loss, test_acc = test(dataloaders[config.FILE_TESTING], dataset_sizes[config.FILE_TESTING],
                                   config.BATCH_SIZE,
                                   criterion, model, device,
                                   semi_supervised=semi_supervised)

        print(f'\tLoss: {test_loss.item():.4f}(test)\t|\tAcc: {test_acc.item() * 100:.1f}%(test)')

    my_plot(np.linspace(1, num_epochs, num_epochs).astype(int), loss_vals_train,
            loss_vals_val, semi_supervised)

    # Model from semi-supervised phase 1 is trained further in semi-supervised phase 2
    if semi_supervised == config.SEMI_SUPERVISED_PHASE_1:
        return model
