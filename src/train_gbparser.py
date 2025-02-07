import os
import sys

from torch.utils.data import TensorDataset, DataLoader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config


class Trainer:
    """Encapsulates training and evaluation logic."""

    def __init__(self, model, optimizer, criterion, batch_size):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.history = {"train_loss": [], "dev_loss": []}

    def train(self, words_train, tags_train, governors_train, words_dev, tags_dev, governors_dev, nb_epochs):
        train_dataset = TensorDataset(words_train, tags_train, governors_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        dev_dataset = TensorDataset(words_dev, tags_dev, governors_dev)
        dev_loader = DataLoader(dev_dataset, batch_size=self.batch_size, shuffle=False)

        for epoch in range(nb_epochs):
            self.model.train()
            avg_train_loss = self._run_epoch(train_loader, train=True)

            self.model.eval()
            avg_dev_loss = self._run_epoch(dev_loader, train=False)

            print(f"Epoch {epoch + 1}/{nb_epochs}, Training Loss: {avg_train_loss:.4f}, Dev Loss: {avg_dev_loss:.4f}")

        return self.history

    def _run_epoch(self, dataloader, train):
        total_loss = 0
        for batch in dataloader:
            words, tags, governors = batch

            self.optimizer.zero_grad()
            output = self.model(words, tags)
            loss = self._compute_loss(output, governors)
            if train:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_key = "train_loss" if train else "dev_loss"
        self.history[loss_key].append(avg_loss)

        return avg_loss

    def _compute_loss(self, output, target):
        output = output.view(-1, output.shape[-1])
        print(output)
        print(target)
        target = target.view(-1)
        mask = target != config.PAD_TOKEN_VAL
        return self.criterion(output[mask], target[mask])
