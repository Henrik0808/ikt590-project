import config
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class, n_features, hidden_dim):
        super(Model, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(n_features * embed_dim, hidden_dim)  # TODO: Change number of out_features?
        self.fc2 = nn.Linear(hidden_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()

    def forward(self, x):
        x = x.long()  # self.embedding(x) expects x of type LongTensor

        embedded = self.embedding(x)
        out = embedded.view(embedded.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out


def train(train_len, optimizer, model, criterion, scheduler, epoch_loss, device, dataloaders, semi_supervised=0):
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

        output = model(x)
        loss = criterion(output, y)
        train_loss += loss.item()
        loss.backward()
        epoch_loss.append(loss.item())
        optimizer.step()
        train_acc += (output.argmax(1) == y).sum().item()

    # Adjust the learning rate
    scheduler.step()

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
        model = Model(config.VOCAB_SIZE, config.EMBED_DIM, num_classes, config.N_FEATURES, config.
                      HIDDEN_DIM).to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

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

        train_loss, train_acc = train(train_len, optimizer, model, criterion, scheduler, epoch_loss_train, device,
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
