import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import src.config as cf


class Trainer:
    """Encapsulates training and evaluation logic."""

    def __init__(self, model, optimizer, criterion, batch_size, device=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.history = {"train_loss": [], "dev_loss": []}
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, words_train, tags_train, governors_train, label_train, words_dev, tags_dev, governors_dev, label_dev, nb_epochs):
        train_dataset = TensorDataset(words_train, tags_train, governors_train, label_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        dev_dataset = TensorDataset(words_dev, tags_dev, governors_dev, label_dev)
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

        # Use tqdm for progress tracking
        with tqdm(dataloader, unit="batch") as pbar:
            for batch in pbar:
                words, tags, governors, labels = batch

                words = words.to(self.device)
                tags = tags.to(self.device)
                governors = governors.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                s_arc, s_rel = self.model(words, tags, train)
                arc_loss = self._compute_loss(s_arc, governors)
                # rel_loss = self._compute_loss(s_rel, labels)

                loss = arc_loss # + rel_loss
                if train:
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()

                # Update tqdm progress bar with the current loss
                pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        loss_key = "train_loss" if train else "dev_loss"
        self.history[loss_key].append(avg_loss)

        return avg_loss

    def _compute_loss(self, output, target):
        """
        Computes loss while handling padding.

        output: (batch_size, seq_length, vocab_size) - Model predictions
        target: (batch_size, seq_length) - Ground truth head indices
        """
        target = target.to(self.device)

        batch_size, seq_length, vocab_size = output.shape

        # Reshape output to (batch_size * seq_length, vocab_size)
        output = output.view(-1, vocab_size)

        # Reshape target to (batch_size * seq_length)
        target = target.view(-1)

        # Apply mask to ignore PAD_TOKEN_VAL
        mask = target != cf.PAD_TOKEN_VAL

        valid_output = output[mask]
        valid_target = target[mask]

        return self.criterion(valid_output, valid_target)
