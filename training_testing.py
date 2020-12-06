import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import random
import os

import config


class SimpleModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class, n_features, hidden_dim):
        super(SimpleModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(n_features * embed_dim, hidden_dim)  # TODO: Change number of out_features?
        self.fc = nn.Linear(hidden_dim, num_class)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.long()  # self.embedding(x) expects x of type LongTensor

        x = x.transpose(1, 0)

        embedded = self.embedding(x)
        out = embedded.view(embedded.size(0), -1)
        out = self.relu(self.fc1(out))
        out = self.fc(out)

        return out


class SimpleGRUModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class, hidden_dim):
        super(SimpleGRUModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.gru = nn.GRU(embed_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_class)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.long()  # self.embedding(x) expects x of type LongTensor

        embedded = self.relu(self.embedding(x))
        _, hidden = self.gru(embedded)
        linear_input = hidden[-1]
        out = self.fc(linear_input)

        return out


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers, bidirectional=True)

        self.hidden = nn.Linear(hidden_size * 2, hidden_size)
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
        hidden = self.hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))

        return encoder_states, hidden


class Classifier(nn.Module):
    def __init__(
            self, encoder, num_classes
    ):
        super(Classifier, self).__init__()

        self.encoder = encoder
        self.fc1 = nn.Linear(config.HIDDEN_DIM, num_classes)
        self.relu = nn.ReLU()
        # self.fc = nn.Linear(1024, config.SUPERVISED_NUM_CLASSES)

    def forward(self, x):
        _, hidden = self.encoder(x)
        hidden = hidden.squeeze(0)
        hidden = self.relu(hidden)
        # outputs = self.relu(self.fc1(hidden))
        predictions = self.fc1(hidden)

        return predictions


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
        # predictions: (batch_size, target_vocab_length)

        return predictions, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.device = device

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, semi_supervised, teacher_force_ratio=config.TEACHER_FORCE_RATIO):
        batch_size = source.shape[1]

        target_len = config.N_FEATURES
        target_vocab_size = config.VOCAB_SIZE

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)

        encoder_states, hidden = self.encoder(source)

        # Get the first input to the decoder which is the sos token
        x = config.SOS_TOKEN_VOCAB

        for t in range(target_len):
            # Use previous hidden as context from encoder at start
            output, hidden = self.decoder(x, encoder_states, hidden, self.device)

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


def get_x_y(semi_supervised, batch, device):
    if semi_supervised == config.SEMI_SUPERVISED_PHASE_1_ELEVENTH_WORD:
        x, y = batch['10 words'].to(device), batch['eleventh word'].to(device)
    elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS:
        x, y = batch['10 words shuffled'].to(device), batch['10 words'].to(device)
        y = y.transpose(1, 0)
    elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD:
        x, y = batch['10 words masked word'].to(device), batch['masked word'].to(device)
    elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER:
        x, y = batch['10 words'].to(device), batch['10 words'].to(device)
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

        if semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS or \
                semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER:

            output = model(x, y, semi_supervised)
            output, y = reshape_output_and_y(output, y)
        else:
            output = model(x)

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
            if semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS or \
                    semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER:

                output = model(x, y, semi_supervised)
                output, y = reshape_output_and_y(output, y)
            else:
                output = model(x)

            batch_loss = criterion(output, y)
            total_loss += batch_loss.item()
            val_acc += (output.argmax(1) == y).sum().item()

    return total_loss / n_batches, val_acc / data_len


def my_plot(epochs, loss_train, loss_val, model_num, semi_supervised=0):
    # Save/show graph of training/validation loss

    plt.plot(epochs, loss_train, label='Training')
    plt.plot(epochs, loss_val, label='Validation')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.savefig('outputs/graphs/' + str(model_num) + '_' + str(semi_supervised) + '_loss_graph.png')

    plt.show()


def run(device, dataset_sizes, dataloaders, num_classes, semi_supervised, num_epochs, model_num,
        initialized_model=None):
    plt.ion()  # interactive mode

    if initialized_model is None:
        if semi_supervised == config.SUPERVISED:
            # Supervised training

            if model_num == 0:
                print("\nSimple model pure supervised:\n")

                model = SimpleModel(config.VOCAB_SIZE, config.EMBED_DIM, config.SUPERVISED_NUM_CLASSES,
                                    config.N_FEATURES, config.HIDDEN_DIM).to(device)
            elif model_num == 1:
                print("\nSimple GRU-model pure supervised:\n")

                model = SimpleGRUModel(config.VOCAB_SIZE, config.EMBED_DIM, config.SUPERVISED_NUM_CLASSES,
                                       config.HIDDEN_DIM).to(device)

            else:
                print("\nSeq2seq model pure supervised:\n")

                encoder = Encoder(
                    config.VOCAB_SIZE, config.EMBED_DIM, config.HIDDEN_DIM, config.NUM_LAYERS, config.ENC_DROPOUT
                ).to(device)

                model = Classifier(encoder, config.SUPERVISED_NUM_CLASSES).to(device)
        else:
            # Semi-supervised phase 1 training

            if model_num == 0:

                if semi_supervised == config.SEMI_SUPERVISED_PHASE_1_ELEVENTH_WORD:
                    print("\nSimple model semi-supervised phase 1 11th word:\n")

                    model = SimpleModel(config.VOCAB_SIZE, config.EMBED_DIM, config.VOCAB_SIZE,
                                        config.N_FEATURES, config.HIDDEN_DIM).to(device)
                elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD:
                    print("\nSimple model semi-supervised phase 1 masked word:\n")

                    model = SimpleModel(config.VOCAB_SIZE, config.EMBED_DIM, config.VOCAB_SIZE,
                                        config.N_FEATURES, config.HIDDEN_DIM).to(device)
                elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS:
                    print("\nSorry, currently not supported: Simple model semi-supervised phase 1 shuffled")
                    return None
                else:
                    print("\nSorry, currently not supported: Simple model semi-supervised phase 1 autoencoder")
                    return None

            elif model_num == 1:

                if semi_supervised == config.SEMI_SUPERVISED_PHASE_1_ELEVENTH_WORD:
                    print("\nSimple GRU-model semi-supervised phase 1 11th word:\n")

                    model = SimpleGRUModel(config.VOCAB_SIZE, config.EMBED_DIM, config.VOCAB_SIZE,
                                        config.HIDDEN_DIM).to(device)
                elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD:
                    print("\nSimple GRU-model semi-supervised phase 1 masked word:\n")

                    model = SimpleGRUModel(config.VOCAB_SIZE, config.EMBED_DIM, config.VOCAB_SIZE,
                                           config.HIDDEN_DIM).to(device)
                elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS:
                    print("\nSorry, currently not supported: Simple GRU-model semi-supervised phase 1 shuffled")
                    return None
                else:
                    print("\nSorry, currently not supported: Simple GRU-model semi-supervised phase 1 autoencoder")
                    return None

            else:
                # model_num is 2 (seq2seq model)

                encoder = Encoder(
                    config.VOCAB_SIZE, config.EMBED_DIM, config.HIDDEN_DIM, config.NUM_LAYERS, config.ENC_DROPOUT
                ).to(device)

                if semi_supervised == config.SEMI_SUPERVISED_PHASE_1_ELEVENTH_WORD:
                    # Find 11th word task
                    print("\nSeq2seq model semi-supervised phase 1 11th word:\n")

                    model = Classifier(encoder, config.VOCAB_SIZE).to(device)
                elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD:
                    # Identify masked word task
                    print("\nSeq2seq model semi-supervised phase 1 masked word:\n")

                    model = Classifier(encoder, config.VOCAB_SIZE).to(device)
                else:
                    # Reordering shuffled sentence task, or auto-encoder task

                    if semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER:
                        print("\nSeq2seq model semi-supervised phase 1 autoencoder:\n")
                    else:
                        print("\nSeq2seq model semi-supervised phase 1 shuffled:\n")

                    decoder = Decoder(
                        config.VOCAB_SIZE,
                        config.EMBED_DIM,
                        config.HIDDEN_DIM,
                        config.VOCAB_SIZE,
                        config.NUM_LAYERS,
                        config.DEC_DROPOUT,
                    ).to(device)

                    model = Seq2Seq(encoder, decoder, device).to(device)
    else:
        # Semi-supervised phase 2 training

        if model_num == 2:
            print("\nSemi-supervised phase 2 seq2seq training:\n")

            model = Classifier(initialized_model, config.SUPERVISED_NUM_CLASSES).to(device)
        elif model_num == 1:
            print("\nSemi-supervised phase 2 simple GRU-model training:\n")

            model = initialized_model
        else:
            # model_num is 0 (SimpleModel)
            print("\nSemi-supervised phase 2 simple model training:\n")

            model = initialized_model

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad is True],
                                 lr=config.LEARNING_RATE)

    if semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER:
        train_len = dataset_sizes[config.FILE_TRAINING] * config.N_FEATURES
    else:
        train_len = dataset_sizes[config.FILE_TRAINING]

    if config.LOAD_MODEL:
        # Load checkpoint, if one exists
        for filename in os.listdir(config.CHECKPOINTS_DIR):
            root, ext = os.path.splitext(filename)

            if root.startswith(f'checkpoint_{model_num}_{semi_supervised}') and ext == '.tar':
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
                                     if semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS or
                                        semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER
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
                                   if semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS or  # todo!
                                      semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER
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
                   f'outputs/checkpoints/checkpoint_{model_num}_{semi_supervised}_{best_epoch_loss}-{best_loss.item():.4f}.tar')

    my_plot(np.linspace(1, num_epochs, num_epochs).astype(int), training_loss,
            validation_loss, model_num, semi_supervised)

    # Best model from semi-supervised phase 1 is trained further in semi-supervised phase 2
    if semi_supervised == config.SEMI_SUPERVISED_PHASE_1_ELEVENTH_WORD or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER:
        return best_model
