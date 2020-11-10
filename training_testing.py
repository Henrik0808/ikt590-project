import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import random
import os

import config


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers, bidirectional=True)

        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        # x shape: (seq_length, batch_size)

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (seq_length, batch_size, embedding_size)

        encoder_states, hidden = self.gru(embedding)
        # encoder_states shape: (seq_length, batch_size, hidden_size * 2)

        # Use forward, backward hidden through a linear layer
        # so that it can be input to the decoder which is not bidirectional
        # Also using index slicing ([idx:idx+1]) to keep the dimension
        # hidden shape: (2, batch_size, hidden_size)
        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))

        return encoder_states, hidden


class Decoder(nn.Module):
    def __init__(
            self, input_size, embedding_size, hidden_size, output_size, num_layers, p
    ):
        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(hidden_size * 2 + embedding_size, hidden_size, num_layers)

        self.energy = nn.Linear(hidden_size * 3, 1)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

    def forward(self, x, encoder_states, hidden, device):
        if isinstance(x, int):
            x = torch.tensor([x for _ in range(hidden.shape[1])], dtype=torch.int64).to(device)

        # x shape: (batch_size), needs to be (1, batch_size).
        # seq_length is 1 here because a single word is sent in and not a sentence
        x = x.unsqueeze(0)
        # x: (1, batch_size)

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (1, batch_size, embedding_size)

        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length, 1, 1)
        # h_reshaped: (seq_length, batch_size, hidden_size)

        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim=2)))
        # energy: (seq_length, batch_size, 1)

        attention = self.softmax(energy)

        # attention: (seq_length, batch_size, 1), snk
        # encoder_states: (seq_length, batch_size, hidden_size*2), snl
        # we want context_vector: (1, batch_size, hidden_size*2), i.e knl
        context_vector = torch.einsum("snk,snl->knl", attention, encoder_states)

        gru_input = torch.cat((context_vector, embedding), dim=2)
        # gru_input: (1, batch_size, hidden_size*2 + embedding_size)

        outputs, hidden = self.gru(gru_input, hidden)
        # outputs shape: (1, batch_size, hidden_size)

        predictions = self.fc(outputs).squeeze(0)
        # predictions: (batch_size, output_size)

        return predictions, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.device = device

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, semi_supervised, teacher_force_ratio=config.TEACHER_FORCE_RATIO):
        batch_size = source.shape[1]

        if semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS:
            target_len = config.N_FEATURES
        else:
            target_len = config.TARGET_LEN

        if semi_supervised == config.SEMI_SUPERVISED_PHASE_1_ELEVENTH_WORD:
            target_vocab_size = config.VOCAB_SIZE
        elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS:
            target_vocab_size = config.N_FEATURES
        else:
            target_vocab_size = config.SUPERVISED_NUM_CLASSES

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)

        encoder_states, hidden = self.encoder(source)

        # Get the first input to the decoder which is the sos token
        if self.decoder.embedding.num_embeddings == 1:
            x = config.SOS_TOKEN_ELEVENTH
        else:
            x = config.SOS_TOKEN_SHUFFLED

        for t in range(0, target_len):
            # Use previous hidden, cell as context from encoder at start
            output, hidden = self.decoder(x, encoder_states, hidden, self.device)

            # Store next output prediction
            outputs[t] = output

            # Get the best word the decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)

            if semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS:
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


def get_x_y(semi_supervised, batch, device):
    if semi_supervised == config.SEMI_SUPERVISED_PHASE_1_ELEVENTH_WORD:
        x, y = batch['10 words'].to(device), batch['eleventh word'].to(device)
    elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS:
        x, y = batch['10 words shuffled'].to(device), batch['10 words shuffled to sentence indexes'].to(device)
        y = y.transpose(1, 0)
    else:
        x, y = batch['10 words'].to(device), batch['category'].to(device)

    # Swapping dimensions of x to make it's shape correct for the GRU without batch_first=True,
    # which expects an input of shape (seq_length, batch_size, hidden_size)
    x = x.transpose(1, 0)
    # x shape: (seq_length, batch_size)

    return x, y


def train(train_len, optimizer, model, criterion, device, dataloaders, semi_supervised=0):
    # Train the model

    total_loss = torch.zeros(1, dtype=torch.float).to(device)
    train_acc = torch.zeros(1, dtype=torch.float).to(device)

    data = dataloaders[config.FILE_TRAINING]

    # Get number of batches
    n_batches = len(data)

    for batch in data:
        optimizer.zero_grad()

        x, y = get_x_y(semi_supervised, batch, device)

        output = model(x, y, semi_supervised)

        output, y = reshape_output_and_y(output, y)

        batch_loss = criterion(output, y)
        total_loss += batch_loss.item()
        batch_loss.backward()

        # Clip to avoid exploding gradient problems, makes sure grads are
        # within an okay range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()
        train_acc += (output.argmax(1) == y).sum().item()

    return total_loss / n_batches, train_acc / train_len


def test(dataloader, data_len, batch_size, criterion, model, device,
         semi_supervised=config.SUPERVISED):
    # Test the model

    total_loss = torch.zeros(1, dtype=torch.float).to(device)
    val_acc = torch.zeros(1, dtype=torch.float).to(device)

    # Get number of batches
    n_batches = len(dataloader)

    for batch in dataloader:

        x, y = get_x_y(semi_supervised, batch, device)

        with torch.no_grad():
            output = model(x, y, semi_supervised)

            output, y = reshape_output_and_y(output, y)

            batch_loss = criterion(output, y)
            total_loss += batch_loss.item()
            val_acc += (output.argmax(1) == y).sum().item()

    return total_loss / n_batches, val_acc / data_len


def my_plot(epochs, loss_train, loss_val, semi_supervised=0):
    # Save/show graph of training/validation loss

    plt.plot(epochs, loss_train, label='Training')
    plt.plot(epochs, loss_val, label='Validation')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    if semi_supervised == config.SUPERVISED:  # Supervised
        plt.savefig('outputs/graphs/supervised_training/loss_graph.png')
    elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_ELEVENTH_WORD:  # Semi-supervised phase 1 eleventh word
        plt.savefig('outputs/graphs/semi_supervised_training/phase_one/eleventh_word/loss_graph.png')
    elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS:  # Semi-supervised phase 1 shuffled words
        plt.savefig('outputs/graphs/semi_supervised_training/phase_one/shuffled_words/loss_graph.png')
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
            config.N_FEATURES + 1 if semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS
            else 1,
            config.EMBED_DIM,
            config.HIDDEN_DIM,
            config.VOCAB_SIZE if semi_supervised == config.SEMI_SUPERVISED_PHASE_1_ELEVENTH_WORD
            else (config.N_FEATURES if semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS
                  else config.SUPERVISED_NUM_CLASSES),
            config.NUM_LAYERS,
            config.DEC_DROPOUT,
        ).to(device)

        model = Seq2Seq(encoder_net, decoder_net, device).to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    if semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS:
        train_len = dataset_sizes[config.FILE_TRAINING] * config.N_FEATURES
    else:
        train_len = dataset_sizes[config.FILE_TRAINING]

    if semi_supervised == config.SUPERVISED:
        print("Supervised training:\n")
    elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_ELEVENTH_WORD:
        print("\nSemi-supervised phase 1 eleventh word training:\n")
    elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS:
        print("\nSemi-supervised phase 1 shuffled words training:\n")
    else:
        print("\nSemi-supervised phase 2 training:\n")

    if config.LOAD_MODEL:
        # Load checkpoint, if one exists
        for filename in os.listdir(config.CHECKPOINTS_DIR):
            root, ext = os.path.splitext(filename)

            if root.startswith(f'checkpoint_{semi_supervised}') and ext == '.tar':
                data = torch.load(config.CHECKPOINTS_DIR + filename)
                print(f'Using checkpoint for {semi_supervised}')
                model.load_state_dict(data)
                return model

        print(f'No checkpoint for {semi_supervised}')

    training_loss = []
    validation_loss = []

    best_loss, best_acc, best_model = float('inf'), 0, None
    best_epoch_loss, best_epoch_acc = 0, 0

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_acc = train(train_len, optimizer, model, criterion, device,
                                      dataloaders, semi_supervised=semi_supervised)

        training_loss.append(train_loss)

        valid_loss, valid_acc = test(dataloaders[config.FILE_VALIDATION],
                                     dataset_sizes[config.FILE_VALIDATION] * config.N_FEATURES
                                     if semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS
                                     else dataset_sizes[config.FILE_VALIDATION],
                                     config.BATCH_SIZE,
                                     criterion, model, device,
                                     semi_supervised=semi_supervised)

        validation_loss.append(valid_loss)
        
        # Save model as the best model if the loss is lower
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch_loss = epoch + 1
            best_model = model

        if valid_acc > best_acc:
            best_acc = valid_acc
            best_epoch_acc = epoch + 1

        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60

        print('Epoch: %d' % (epoch + 1), " | time in %d minutes, %d seconds" % (mins, secs))
        print(f'\tLoss: {train_loss.item():.4f}(train)\t|\tAcc: {train_acc.item() * 100:.1f}%(train)')
        print(f'\tLoss: {valid_loss.item():.4f}(valid)\t|\tAcc: {valid_acc.item() * 100:.1f}%(valid)')

        print('Checking the results of test dataset...')

        test_loss, test_acc = test(dataloaders[config.FILE_TESTING],
                                   dataset_sizes[config.FILE_TESTING] * config.N_FEATURES
                                   if semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS
                                   else dataset_sizes[config.FILE_TESTING],
                                   config.BATCH_SIZE,
                                   criterion, model, device,
                                   semi_supervised=semi_supervised)

        print(f'\tLoss: {test_loss.item():.4f}(test)\t|\tAcc: {test_acc.item() * 100:.1f}%(test)')

    print(f'\nBest validation loss: {best_loss.item():.4f} (Epoch: {best_epoch_loss})'
          f'\nBest validation accuracy: {best_acc.item() * 100:.1f}% (Epoch: {best_epoch_acc})')

    if config.SAVE_MODEL:
        # Save best model to disk
        torch.save(best_model.state_dict(),
                   f'outputs/checkpoints/checkpoint_{semi_supervised}_{best_epoch_loss}-{best_loss.item():.4f}.tar')

    my_plot(np.linspace(1, num_epochs, num_epochs).astype(int), training_loss,
            validation_loss, semi_supervised)

    # Best model from semi-supervised phase 1 is trained further in semi-supervised phase 2
    if semi_supervised == config.SEMI_SUPERVISED_PHASE_1_ELEVENTH_WORD or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS:
        return best_model
