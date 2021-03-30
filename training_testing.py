import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

import config
from main import experiment_to_filename, experiment_to_filename_phase1


class SimpleModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class, n_features, hidden_dim):
        super(SimpleModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(n_features * embed_dim, hidden_dim)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, config.NUM_CLASSES)

        if not config.SEMI_SUPERVISED == config.SUPERVISED:
            self.linears = nn.ModuleList(
                [nn.Linear(hidden_dim, config.VOCAB_SIZE) for _ in range(config.TARGET_LEN)])

        if config.SEMI_SUPERVISED == config.PHASE_1:
            self.fc.requires_grad_(False)
        elif config.SEMI_SUPERVISED == config.PHASE_2:
            self.linears.requires_grad_(False)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(config.ENC_DROPOUT)

    def forward(self, x, x_lengths=None):
        embedded = self.dropout(self.embedding(x))
        out = embedded.view(embedded.size(0), -1)
        out = self.fc1(out)
        out = self.sigmoid(out)
        out = self.hidden(out)
        out = self.sigmoid(out)

        if config.SEMI_SUPERVISED == config.PHASE_1:
            outputs = []

            for idx, l in enumerate(self.linears):
                outputs.append(l(out))

            return outputs
        else:
            out = self.fc(out)

            return out


class SimpleGRUModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class, hidden_dim):
        super(SimpleGRUModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.gru = nn.GRU(embed_dim, hidden_dim)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, config.NUM_CLASSES)

        if not config.SEMI_SUPERVISED == config.SUPERVISED:
            self.linears = nn.ModuleList(
                [nn.Linear(hidden_dim, config.VOCAB_SIZE) for i in range(config.TARGET_LEN)])

        if config.SEMI_SUPERVISED == config.PHASE_1:
            self.fc.requires_grad_(False)
        elif config.SEMI_SUPERVISED == config.PHASE_2:
            self.linears.requires_grad_(False)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(config.ENC_DROPOUT)

    def forward(self, x, x_lengths):
        x = x.long()  # self.embedding(x) expects x of type LongTensor

        embedded = self.dropout(self.embedding(x))
        embedded_packed = pack_padded_sequence(embedded, x_lengths)

        _, hidden = self.gru(embedded_packed)

        hidden = self.hidden(hidden[-1])

        linear_input = self.relu(hidden)

        if config.SEMI_SUPERVISED == config.PHASE_1:
            outputs = []

            for idx, l in enumerate(self.linears):
                outputs.append(l(linear_input))

            return outputs
        else:
            out = self.fc(linear_input)

            return out


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers, bidirectional=True)

        self.hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(p)

    def forward(self, x, x_lengths):
        # x shape: (seq_length, batch_size)

        x = x.long()

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (seq_length, batch_size, embedding_size)

        embedding_packed = pack_padded_sequence(embedding, x_lengths)

        encoder_states, hidden = self.gru(embedding_packed)
        # encoder_states shape: (seq_length, batch_size, hidden_size * 2)

        # Unpack padding
        encoder_states, _ = torch.nn.utils.rnn.pad_packed_sequence(encoder_states)

        # Use forward, backward hidden through a linear layer
        # so that it can be input to the decoder which is not bidirectional
        # Also using index slicing ([idx:idx+1]) to keep the dimension
        # hidden shape: (2, batch_size, hidden_size)
        hidden = self.hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2)).to(config.DEVICE)

        return encoder_states, hidden


class Classifier(nn.Module):
    def __init__(
            self, encoder, num_classes
    ):
        super(Classifier, self).__init__()

        self.encoder = encoder
        self.fc = nn.Linear(config.HIDDEN_DIM, config.NUM_CLASSES)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # self.fc1 = nn.Linear(1024, config.SUPERVISED_NUM_CLASSES)

    def forward(self, x, x_lengths=None):
        _, hidden = self.encoder(x, x_lengths)
        hidden = hidden.squeeze(0)
        hidden = self.sigmoid(hidden)
        # outputs = self.relu(self.fc1(hidden))
        predictions = self.fc(hidden)

        return predictions


class Decoder(nn.Module):
    def __init__(
            self, input_size, embedding_size, hidden_size, output_size, num_layers, p
    ):
        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.gru = nn.GRU(hidden_size * 2 + embedding_size, hidden_size, num_layers)

        self.energy = nn.Linear(hidden_size * 3, 1)
        self.fc = nn.Linear(hidden_size, config.VOCAB_SIZE)
        self.dropout = nn.Dropout(p)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

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

        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim=2).to(config.DEVICE)))
        # energy: (seq_length, batch_size, 1)

        attention = self.softmax(energy)

        # attention: (seq_length, batch_size, 1), snk
        # encoder_states: (seq_length, batch_size, hidden_size*2), snl
        # we want context_vector: (1, batch_size, hidden_size*2), i.e knl
        context_vector = torch.einsum("snk,snl->knl", attention, encoder_states).to(config.DEVICE)

        gru_input = torch.cat((context_vector, embedding), dim=2).to(config.DEVICE)
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

    def forward(self, source, target, semi_supervised, x_lengths, teacher_force_ratio=config.TEACHER_FORCE_RATIO):
        batch_size = source.shape[1]

        if config.SEMI_SUPERVISED == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_20NEWS or \
                config.SEMI_SUPERVISED == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_20NEWS:
            target_len = config.N_FEATURES
            num_classes = config.VOCAB_SIZE
        elif config.SEMI_SUPERVISED == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_CLINC150 or \
                config.SEMI_SUPERVISED == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_CLINC150:
            target_len = x_lengths[0]
            num_classes = config.VOCAB_SIZE
        elif config.SEMI_SUPERVISED == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_20NEWS or \
                config.SEMI_SUPERVISED == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_CLINC150:
            target_len = config.TARGET_LEN_MASKED  # Because 2 masked words to predict
            num_classes = config.VOCAB_SIZE
        #elif config.SEMI_SUPERVISED == config.SEMI_SUPERVISED_PHASE_2_BANKING77 or \
                #config.SEMI_SUPERVISED == config.SUPERVISED_BANKING77:
            #target_len = 1
            #num_classes = config.NUM_CLASSES

        outputs = torch.zeros(target_len, batch_size, num_classes).to(self.device)

        encoder_states, hidden = self.encoder(source, x_lengths)

        # Get the first input to the decoder which is the sos token
        x = config.SOS_TOKEN_VOCAB

        for t in range(target_len):  # todo: if simpleModel target_len 10 else have eos or flexible target_len?
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
            # x = target[t] if random.random() < teacher_force_ratio else best_guess
            x = best_guess
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


def convert_ints_to_words(ints):
    sentence_list = []
    sentences = []

    for sentence in ints:
        for i in sentence:
            i = i.item()
            if int(i) == 0:
                continue
            word = config.TOKENIZER.index_word[int(i)]
            if word == 1:
                continue
            sentence_list.append(config.TOKENIZER.index_word[int(i)])
        sentences.append(sentence_list)
        sentence_list = []

    for idx, s in enumerate(sentences):
        sentences[idx] = ' '.join(s)

    return sentences


def get_x_y(semi_supervised, batch, device, model):
    x, x_lengths, y = batch
    x = x.to(device)
    x_lengths = x_lengths.to(device)
    y = y.long()
    y = y.to(device)

    if semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_20NEWS or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_CLINC150 or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_20NEWS or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_CLINC150 or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_20NEWS or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_CLINC150:
        y = y.transpose(1, 0)

    if not isinstance(config.MODEL, SimpleModel):
        # Swapping dimensions of x to make it's shape correct for the GRU without batch_first=True,
        # which expects an input of shape (seq_length, batch_size, hidden_size)
        x = x.transpose(1, 0)
        # x shape: (seq_length, batch_size)

    return x, y, x_lengths


def train(train_len, optimizer, model, criterion, device, dataloaders, semi_supervised=0):
    # Train the model

    total_loss = torch.zeros(1, dtype=torch.float).to(device)
    train_acc = torch.zeros(1, dtype=torch.float).to(device)

    if semi_supervised == config.SUPERVISED_20NEWS or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_2_20NEWS:
        data = dataloaders[config.FILE_TRAINING]
    elif semi_supervised == config.SUPERVISED_BANKING77 or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_2_BANKING77:
        data = dataloaders[config.FILE_TRAINING_BANKING77]
    elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_ELEVENTH_WORD_20NEWS:
        data = dataloaders[config.FILE_TRAINING]
    elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_20NEWS:
        data = dataloaders[config.FILE_TRAINING]
    elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_20NEWS or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_20NEWS:
        data = dataloaders[config.FILE_TRAINING]
    elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_CLINC150 or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_CLINC150:
        data = dataloaders[config.FILE_TRAINING_CLINC150]
    elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_CLINC150:
        data = dataloaders[config.FILE_TRAINING_CLINC150]

    del dataloaders

    # Get number of batches
    n_batches = len(data)

    for batch in data:
        optimizer.zero_grad()

        x, y, x_lengths = get_x_y(semi_supervised, batch, device, model)

        if (semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_20NEWS or
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_BANKING77 or
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_CLINC150 or
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_20NEWS or
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_BANKING77 or
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_CLINC150 or
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_20NEWS or
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_CLINC150) \
                and not (isinstance(model, SimpleModel) or
                         isinstance(model, SimpleGRUModel)):

            out = model(x, y, semi_supervised, x_lengths)
            out, y = reshape_output_and_y(out, y)

            batch_loss = criterion(out, y)
            loss = batch_loss
            total_loss += loss.item()
            loss.backward()

            train_acc += (out.argmax(1) == y).sum().item()
        else:
            if semi_supervised == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_20NEWS or \
                    semi_supervised == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_CLINC150 or \
                    semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_20NEWS or \
                    semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_CLINC150:
                outputs = model(x, x_lengths)

                y = y.transpose(1, 0)

                loss = torch.zeros(1).to(device)

                if semi_supervised == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_20NEWS or \
                        semi_supervised == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_CLINC150:
                    target_len = config.TARGET_LEN_MASKED
                elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_20NEWS or \
                    semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_CLINC150:
                    target_len = x_lengths[0]

                for idx in range(target_len):
                    y_temp = torch.tensor([i[idx] for i in y]).to(device)
                    batch_loss = criterion(outputs[idx], y_temp)
                    loss += batch_loss
                    total_loss += batch_loss.item()
                    train_acc += (outputs[idx].argmax(1) == y_temp).sum().item()

                del idx
                del target_len
                del y_temp
                del outputs

                loss.backward()
            else:
                out = model(x, x_lengths)

                batch_loss = criterion(out, y)
                loss = batch_loss
                total_loss += loss.item()
                loss.backward()

                train_acc += (out.argmax(1) == y).sum().item()

        # Clip to avoid exploding gradient problems, makes sure grads are
        # within an okay range
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1).to(device)

        # Free unused variables from memory
        del batch_loss
        del loss
        del x
        del y

        optimizer.step()

        if semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_CLINC150 or \
                semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_CLINC150:
            train_len += x_lengths[0] * x_lengths.shape[0]

    # Adjust the learning rate
    # scheduler.step()

    return total_loss / n_batches, train_acc / train_len


def test(dataloaders, dataset_sizes, batch_size, criterion, model, device,
         semi_supervised=config.SUPERVISED_BANKING77):
    # Test the model

    # Get dataloader
    if (semi_supervised == config.SUPERVISED_BANKING77 or
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_ELEVENTH_WORD_BANKING77 or
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_BANKING77 or
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_BANKING77 or
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_BANKING77 or
            semi_supervised == config.SEMI_SUPERVISED_PHASE_2_BANKING77):
        dataloader = dataloaders[config.FILE_VALIDATION_BANKING77]
    elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_ELEVENTH_WORD_20NEWS or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_20NEWS or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_20NEWS or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_20NEWS:
        dataloader = dataloaders[config.FILE_VALIDATION]
    elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_CLINC150 or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_CLINC150 or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_CLINC150:
        dataloader = dataloaders[config.FILE_VALIDATION_CLINC150]
    else:
        dataloader = dataloaders[config.FILE_VALIDATION]

    # Get dataset size
    if semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_20NEWS or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_20NEWS:
        data_len = dataset_sizes[config.FILE_VALIDATION] * config.N_FEATURES
    elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_CLINC150 or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_CLINC150:
        data_len = 0
    elif semi_supervised == config.SUPERVISED_20NEWS or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_2_20NEWS or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_ELEVENTH_WORD_20NEWS:
        data_len = dataset_sizes[config.FILE_VALIDATION]
    elif semi_supervised == config.SUPERVISED_CLINC150 or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_2_CLINC150:
        data_len = dataset_sizes[config.FILE_VALIDATION_CLINC150]
    elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_20NEWS:
        data_len = dataset_sizes[config.FILE_VALIDATION] * config.TARGET_LEN_MASKED
    elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_CLINC150:
        data_len = dataset_sizes[config.FILE_VALIDATION_CLINC150] * config.TARGET_LEN_MASKED
    else:
        data_len = dataset_sizes[config.FILE_VALIDATION_BANKING77]

    total_loss = torch.zeros(1, dtype=torch.float).to(device)
    val_acc = torch.zeros(1, dtype=torch.float).to(device)

    # Get number of batches
    n_batches = len(dataloader)

    for batch in dataloader:
        x, y, x_lengths = get_x_y(semi_supervised, batch, device, model)

        with torch.no_grad():
            if (semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_20NEWS or
                semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_BANKING77 or
                semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_CLINC150 or
                semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_20NEWS or
                semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_BANKING77 or
                semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_CLINC150 or
                semi_supervised == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_20NEWS or
                semi_supervised == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_CLINC150) \
                    and not (isinstance(model, SimpleModel) or
                             isinstance(model, SimpleGRUModel)):

                out = model(x, y, semi_supervised, x_lengths)
                out, y = reshape_output_and_y(out, y)

                batch_loss = criterion(out, y)
                loss = batch_loss
                total_loss += loss.item()
                val_acc += (out.argmax(1) == y).sum().item()
            else:
                if semi_supervised == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_20NEWS or \
                        semi_supervised == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_CLINC150 or \
                        semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_20NEWS or \
                        semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_CLINC150:
                    outputs = model(x, x_lengths)

                    y = y.transpose(1, 0)

                    loss = torch.zeros(1).to(device)

                    if semi_supervised == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_20NEWS or \
                            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_CLINC150:
                        target_len = config.TARGET_LEN_MASKED
                    elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_20NEWS or \
                            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_CLINC150:
                        target_len = x_lengths[0]

                    for idx in range(target_len):
                        y_temp = torch.tensor([i[idx] for i in y]).to(device)
                        batch_loss = criterion(outputs[idx], y_temp)
                        loss += batch_loss
                        total_loss += loss.item()
                        val_acc += (outputs[idx].argmax(1) == y_temp).sum().item()
                else:
                    out = model(x, x_lengths)

                    batch_loss = criterion(out, y)
                    loss = batch_loss
                    total_loss += loss.item()
                    val_acc += (out.argmax(1) == y).sum().item()

            if semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_CLINC150 or \
                    semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_CLINC150:
                data_len += x_lengths[0] * x_lengths.shape[0]

    return total_loss / n_batches, val_acc / data_len  # todo: is data_len correct (3080 should be 3000 maybe)?


def my_plot(epochs, loss_train, loss_val, model_num, semi_supervised=0):
    if config._experiment:
        # Yoink the data to a csv file instead
        fn = experiment_to_filename_phase1(config._experiment) if config._experiment[
            'phase1'] else experiment_to_filename(config._experiment)
        with open('outputs/experiments/' + fn, 'w', encoding='utf-8') as f:
            f.write('epoch,loss_train,loss_val\n')
            for row in zip(epochs, loss_train, loss_val):
                f.write(str(','.join(str(float(x)) for x in row)) + '\n')
    else:
        # Save/show graph of training/validation loss

        plt.plot(epochs, loss_train, label='Training')
        plt.plot(epochs, loss_val, label='Validation')
        plt.legend(loc='upper right')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')

        plt.savefig('outputs/graphs/' + str(model_num) + '_' + str(semi_supervised) + '_loss_graph.png')

        plt.show()


def find_max_ndarray(ndarray):
    ndarray_len = [len(i) for i in ndarray]
    max_query_len = max(ndarray_len)

    return max_query_len


def find_max_query_len(dataloaders):
    queries = dataloaders[config.FILE_TRAINING_BANKING77].dataset.records_tokenized['query']
    max_query_len = find_max_ndarray(queries)
    # - 2 below because of removing 'sos' and '[MASK]' tokens when getting query from dataset
    config.MAX_QUERY_LEN = max_query_len - 2


def load_pretrained_model(model_num, semi_supervised, model=None, optimizer=None):
    # Load pretrained model, if one exists
    for filename in os.listdir(config.CHECKPOINTS_DIR):
        root, ext = os.path.splitext(filename)

        if root.startswith(f'{config.MODEL_MAP[model_num]}_{config.PREPROC_MAP[semi_supervised]}') and ext == '.tar':
            # If model is None, then user wants to skip training if saved checkpoint exists,
            # therefore return 0, signaling to skip model training
            if model is None:
                return 0

            data = torch.load(config.CHECKPOINTS_DIR + filename)

            model.load_state_dict(data['model_state_dict'], strict=False)

            # todo: correct?
            if not config.SEMI_SUPERVISED == config.PHASE_2:
                optimizer.load_state_dict(data['optimizer_state_dict'])

            num_epochs = data['num_epochs']

            return model, optimizer, num_epochs

    print(f'No checkpoint for {semi_supervised}')

    return None


def run(device, dataset_sizes, dataloaders, num_classes, semi_supervised, num_epochs, model_num):
    plt.ion()  # interactive mode

    config.SEMI_SUPERVISED = semi_supervised

    find_max_query_len(dataloaders)  # todo: find max query len for 20news also

    if config.PHASE_1 == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_20NEWS or \
        config.PHASE_1 == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_CLINC150:
        config.TARGET_LEN = config.MAX_QUERY_LEN
    elif config.PHASE_1 == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_20NEWS or \
        config.PHASE_1 == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_CLINC150:
        config.TARGET_LEN = config.TARGET_LEN_MASKED

    if semi_supervised == config.SUPERVISED_BANKING77 or semi_supervised == config.SUPERVISED_20NEWS:
        # Supervised training banking77 or 20news

        if model_num == 0:

            if semi_supervised == config.SUPERVISED_BANKING77:
                print("\nSimple model pure supervised banking77:\n")
            elif semi_supervised == config.SUPERVISED_20NEWS:
                print("\nSimple model pure supervised 20news:\n")

            model = SimpleModel(config.VOCAB_SIZE, config.EMBED_DIM, config.NUM_CLASSES,
                                    config.MAX_QUERY_LEN, config.HIDDEN_DIM).to(device)
        elif model_num == 1:
            if semi_supervised == config.SUPERVISED_BANKING77:
                print("\nSimple GRU-model pure supervised banking77:\n")
            elif semi_supervised == config.SUPERVISED_20NEWS:
                print("\nSimple GRU-model pure supervised 20news:\n")

            model = SimpleGRUModel(config.VOCAB_SIZE, config.EMBED_DIM, config.NUM_CLASSES,
                                   config.HIDDEN_DIM).to(device)
        else:
            if semi_supervised == config.SUPERVISED_BANKING77:
                print("\nSeq2seq model pure supervised banking77:\n")
            elif semi_supervised == config.SUPERVISED_20NEWS:
                print("\nSeq2seq model pure supervised 20news:\n")

            encoder = Encoder(
                config.VOCAB_SIZE, config.EMBED_DIM, config.HIDDEN_DIM, config.NUM_LAYERS, config.ENC_DROPOUT
            ).to(device)

            model = Classifier(encoder, config.NUM_CLASSES).to(device)

            '''
            decoder = Decoder(
                config.VOCAB_SIZE,
                config.EMBED_DIM,
                config.HIDDEN_DIM,
                config.NUM_CLASSES,
                config.NUM_LAYERS,
                config.DEC_DROPOUT,
            ).to(device)

            model = Seq2Seq(encoder, decoder, device).to(device)
            '''
    elif semi_supervised == config.PHASE_1:
        # Semi-supervised phase 1 training

        if model_num == 0:
            # Simple model

            if semi_supervised == config.SEMI_SUPERVISED_PHASE_1_ELEVENTH_WORD_20NEWS:
                print("\nSimple model semi-supervised phase 1 11th word 20news:\n")
            elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_ELEVENTH_WORD_BANKING77:
                print("\nSimple model semi-supervised phase 1 11th word banking77:\n")
                # todo
            elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_20NEWS:
                print("\nSimple model semi-supervised phase 1 masked word 20news:\n")
            elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_BANKING77:
                print("\nSimple model semi-supervised phase 1 masked word banking77:\n")
                # todo
            elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_CLINC150:
                print("\nSimple model semi-supervised phase 1 masked word clinc150:\n")
            elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_20NEWS:
                print("\nSorry, currently not supported: Simple model semi-supervised phase 1 shuffled 20news")
                return None
            elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_BANKING77:
                print("\nSorry, currently not supported: Simple model semi-supervised phase 1 shuffled banking77")
                return None
            elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_CLINC150:
                print("\nSorry, currently not supported: Simple model semi-supervised phase 1 shuffled clinc150")
                return None
            elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_20NEWS:
                print("\nSorry, currently not supported: Simple model semi-supervised phase 1 autoencoder 20news")
                return None
            elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_BANKING77:
                print(
                    "\nSorry, currently not supported: Simple model semi-supervised phase 1 autoencoder banking77")
                return None
            elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_CLINC150:
                print(
                    "\nSimple model semi-supervised phase 1 autoencoder clinc150")

            model = SimpleModel(config.VOCAB_SIZE, config.EMBED_DIM, config.VOCAB_SIZE,
                                       config.MAX_QUERY_LEN, config.HIDDEN_DIM).to(device)

        elif model_num == 1:
            # SimpleGRUModel

            if semi_supervised == config.SEMI_SUPERVISED_PHASE_1_ELEVENTH_WORD_20NEWS:
                print("\nSimple GRU-model semi-supervised phase 1 11th word:\n")
            elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_20NEWS:
                print("\nSimple GRU-model semi-supervised phase 1 masked word:\n")
            elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_CLINC150:
                print("\nSimple GRU-model semi-supervised phase 1 masked word:\n")
            elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_20NEWS:
                print("\nSorry, currently not supported: Simple GRU-model semi-supervised phase 1 shuffled")
                return None
            elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_20NEWS:
                print("\nSimple GRU-model semi-supervised phase 1 20news autoencoder")
            elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_CLINC150:
                print("\nSimple GRU-model semi-supervised phase 1 clinc150 autoencoder")

            model = SimpleGRUModel(config.VOCAB_SIZE, config.EMBED_DIM, config.VOCAB_SIZE,
                                   config.HIDDEN_DIM).to(device)
        else:
            # model_num is 2 (seq2seq model)

            encoder = Encoder(
                config.VOCAB_SIZE, config.EMBED_DIM, config.HIDDEN_DIM, config.NUM_LAYERS, config.ENC_DROPOUT
            ).to(device)

            if semi_supervised == config.SEMI_SUPERVISED_PHASE_1_ELEVENTH_WORD_20NEWS:
                # Find 11th word task
                print("\nSeq2seq model semi-supervised phase 1 11th word:\n")
                model = Classifier(encoder, config.VOCAB_SIZE).to(device)
            elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_20NEWS or \
                    semi_supervised == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_CLINC150:
                # Identify masked word task
                print("\nSeq2seq model semi-supervised phase 1 masked word:\n")
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
                # Reordering shuffled sentence task, or auto-encoder task

                if semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_20NEWS or \
                        semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_CLINC150:
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

            encoder = Encoder(
                config.VOCAB_SIZE, config.EMBED_DIM, config.HIDDEN_DIM, config.NUM_LAYERS, config.ENC_DROPOUT
            ).to(device)

            model = Classifier(encoder, config.NUM_CLASSES).to(device)

            '''
            decoder = Decoder(
                config.VOCAB_SIZE,
                config.EMBED_DIM,
                config.HIDDEN_DIM,
                config.NUM_CLASSES,
                config.NUM_LAYERS,
                config.DEC_DROPOUT,
            ).to(device)

            model = Seq2Seq(encoder, decoder, device).to(device)
            '''
        elif model_num == 1:
            print("\nSemi-supervised phase 2 simple GRU-model training:\n")

            model = SimpleGRUModel(config.VOCAB_SIZE, config.EMBED_DIM, config.NUM_CLASSES,
                                   config.HIDDEN_DIM).to(device)

        else:
            # model_num is 0 (SimpleModel)
            print("\nSemi-supervised phase 2 simple model training:\n")

            model = SimpleModel(config.VOCAB_SIZE, config.EMBED_DIM, config.NUM_CLASSES,
                                config.MAX_QUERY_LEN, config.HIDDEN_DIM).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    num_epochs_already_trained = 0

    if config.CONTINUE_TRAINING_MODEL:
        # Continue training model, if saved model checkpoint exists
        if load_pretrained_model(model_num, semi_supervised, model, optimizer) is not None:
            model, optimizer, num_epochs_already_trained = load_pretrained_model(model_num, semi_supervised, model, optimizer)
            print(f'Continuing training model {config.MODEL_MAP[model_num]} from epoch {num_epochs_already_trained}')
    else:
        # Skip model training if saved checkpoint for model exists
        if load_pretrained_model(model_num, semi_supervised) is not None:
            print(f'Skipping training model {config.MODEL_MAP[model_num]}, because saved checkpoint already exists')

            del model
            return

    if semi_supervised == config.PHASE_2:
        model, optimizer, _ = load_pretrained_model(model_num, config.PHASE_1, model, optimizer)

    config.MODEL = model

    if semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_CLINC150 or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_CLINC150 or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_BANKING77 or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_BANKING77 or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_CLINC150 or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_20NEWS:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss().to(device)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

    if semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_20NEWS or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_BANKING77 or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_20NEWS or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_BANKING77:
        train_len = dataset_sizes[config.FILE_TRAINING] * config.N_FEATURES
    elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_CLINC150:
        train_len = 0  # Calculate later
    elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_CLINC150:
        train_len = 0  # Calculate later
    elif semi_supervised == config.SUPERVISED_20NEWS:
        train_len = dataset_sizes[config.FILE_TRAINING]
    elif semi_supervised == config.SUPERVISED_BANKING77 or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_2_BANKING77:
        train_len = dataset_sizes[config.FILE_TRAINING_BANKING77]
    elif semi_supervised == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_CLINC150:
        train_len = dataset_sizes[config.FILE_TRAINING_CLINC150] * config.TARGET_LEN_MASKED
    else:
        train_len = dataset_sizes[
                        config.FILE_TRAINING] * config.TARGET_LEN_MASKED  # * 2 because 2 masked words to predict

    training_loss = []
    validation_loss = []

    best_loss, best_acc, best_model, best_optimizer = float('inf'), 0, None, None
    best_epoch_loss, best_epoch_acc = 0, 0

    tot_num_epochs = num_epochs + num_epochs_already_trained

    for epoch in range(num_epochs_already_trained, tot_num_epochs):
        start_time = time.time()

        model.train()

        train_loss, train_acc = train(train_len, optimizer, model, criterion, device,
                                      dataloaders, semi_supervised=semi_supervised)

        training_loss.append(train_loss)

        model.eval()

        valid_loss, valid_acc = test(dataloaders,
                                     dataset_sizes,
                                     config.BATCH_SIZE,
                                     criterion, model, device,
                                     semi_supervised=semi_supervised)

        validation_loss.append(valid_loss)

        # Save model as the best model if the loss is lower
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch_loss = epoch + 1
            best_model = model
            best_optimizer = optimizer

        if valid_acc > best_acc:
            best_acc = valid_acc
            best_epoch_acc = epoch + 1

        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60

        print('Epoch: %d' % (epoch + 1), " | time in %d minutes, %d seconds" % (mins, secs))
        print(f'\tLoss: {train_loss.item():.4f}(train)\t|\tAcc: {train_acc.item() * 100:.1f}%(train)')
        print(f'\tLoss: {valid_loss.item():.4f}(valid)\t|\tAcc: {valid_acc.item() * 100:.1f}%(valid)')
        ''' 
        print('Checking the results of test dataset...')

        
        test_loss, test_acc = test(dataloaders[config.FILE_TESTING],
                                   dataset_sizes[config.FILE_TESTING] * config.N_FEATURES
                                   if semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS or
                                      semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER
                                   else dataset_sizes[config.FILE_TESTING],
                                   config.BATCH_SIZE,
                                   criterion, model, device,
                                   semi_supervised=semi_supervised)

        print(f'\tLoss: {test_loss.item():.4f}(test)\t|\tAcc: {test_acc.item() * 100:.1f}%(test)')
        '''
    print(f'\nBest validation loss: {best_loss.item():.4f} (Epoch: {best_epoch_loss})'
          f'\nBest validation accuracy: {best_acc.item() * 100:.1f}% (Epoch: {best_epoch_acc})')

    if config.SAVE_MODEL:
        # Save best model to disk
        torch.save({'model_state_dict': best_model.state_dict(),
                    'optimizer_state_dict': best_optimizer.state_dict(),
                    'num_epochs': tot_num_epochs
                    },
                   f'outputs/checkpoints/{config.MODEL_MAP[model_num]}_{config.PREPROC_MAP[semi_supervised]}_{best_epoch_loss}-{best_loss.item():.4f}.tar')

    my_plot(np.linspace(num_epochs_already_trained + 1, tot_num_epochs, num_epochs).astype(int), training_loss,
            validation_loss, model_num, semi_supervised)

    '''
    # Best model from semi-supervised phase 1 is trained further in semi-supervised phase 2
    if semi_supervised == config.SEMI_SUPERVISED_PHASE_1_ELEVENTH_WORD_20NEWS or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_ELEVENTH_WORD_BANKING77 or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_20NEWS or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_BANKING77 or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_CLINC150 or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_20NEWS or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_BANKING77 or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_CLINC150 or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_20NEWS or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_BANKING77 or \
            semi_supervised == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_CLINC150:

        return best_model
    '''
