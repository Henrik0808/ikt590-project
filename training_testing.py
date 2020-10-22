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

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True, dropout=p)

    def forward(self, x):
        # x shape: (seq_length, batch_size)

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (seq_length, batch_size, embedding_size)

        outputs, (hidden, cell) = self.lstm(embedding)
        # outputs shape: (seq_length, batch_size, hidden_size)

        return hidden, cell


class Decoder(nn.Module):
    def __init__(
        self, input_size, embedding_size, hidden_size, output_size, num_layers, p
    ):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(p)

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell, device):
        # x shape: (batch_size), needs to be (1, batch_size).
        # seq_length is 1 here because a single word is sent in and not a sentence
        x = torch.tensor([x for _ in range(hidden.shape[1])], dtype=torch.int64).to(device)
        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (1, batch_size, embedding_size)

        outputs, (hidden, cell) = self.lstm(embedding, (hidden, cell))
        # outputs shape: (1, batch_size, hidden_size)

        predictions = self.fc(outputs)

        # predictions shape: (1, batch_size, length_target_vocabulary). To send it to the
        # loss function it needs to be (batch_size, length_target_vocabulary) so the first
        # dim is going to be erased
        predictions = predictions.squeeze(0)

        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.device = device

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, semi_supervised, teacher_force_ratio=config.TEACHER_FORCE_RATIO):
        batch_size = source.shape[0]
        target_len = config.TARGET_LEN

        if semi_supervised == config.SEMI_SUPERVISED_PHASE_1:
            target_vocab_size = config.VOCAB_SIZE
        else:
            target_vocab_size = config.SUPERVISED_NUM_CLASSES

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)

        hidden, cell = self.encoder(source)

        # Get the first input to the decoder which is the sos token
        x = config.SOS_TOKEN

        for t in range(0, target_len):
            # Use previous hidden, cell as context from encoder at start
            output, hidden, cell = self.decoder(x, hidden, cell, self.device)

            # Store next output prediction
            outputs[t] = output

            # Get the best word the decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)

            # With probability of teacher_force_ratio the actual next word is used
            # otherwise the word that the decoder predicted it to be is used.
            # Teacher Forcing is utilized so that the model gets accustomed to seeing
            # similar inputs at training and testing time. If teacher forcing is 1
            # then inputs at test time could be very different from what the
            # network is accustomed to
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs


def reshape_output_and_y(output, y):
    # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
    # doesn't take input in that form. For example if MNIST is used we want to have
    # output to be: (batch_size, 10) and targets just (batch_size). Here we can view it in a familiar
    # way that we have output_words * batch_size that we want to send into
    # our cost function, so some reshaping needs to be done
    output = output.reshape(-1, output.shape[2])
    y = y.reshape(-1)

    return output, y


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

        output = model(x, y, semi_supervised)

        output, y = reshape_output_and_y(output, y)

        loss = criterion(output, y)
        train_loss += loss.item()
        loss.backward()

        # Clip to avoid exploding gradient problems, makes sure grads are
        # within an okay range
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
            output = model(x, y, semi_supervised)

            output, y = reshape_output_and_y(output, y)

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

        decoder_net = Decoder(
            config.VOCAB_SIZE,
            config.EMBED_DIM,
            config.HIDDEN_DIM,
            config.VOCAB_SIZE if semi_supervised == config.SEMI_SUPERVISED_PHASE_1
            else config.SUPERVISED_NUM_CLASSES,
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
