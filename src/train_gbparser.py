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

    def train(self, words_train, tags_train, governors_train, label_train, words_dev, tags_dev, governors_dev,
              label_dev, nb_epochs):
        train_dataset = TensorDataset(words_train, tags_train, governors_train, label_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        dev_dataset = TensorDataset(words_dev, tags_dev, governors_dev, label_dev)
        dev_loader = DataLoader(dev_dataset, batch_size=self.batch_size, shuffle=False)

        for epoch in range(nb_epochs):
            self.model.train()
            avg_train_loss, arc_loss_train, rel_loss_train = self._run_epoch(train_loader, train=True)

            self.model.eval()
            avg_dev_loss, arc_loss_dev, rel_loss_dev = self._run_epoch(dev_loader, train=False)

            print(f"Epoch {epoch + 1}/{nb_epochs}, Training Loss: {avg_train_loss:.4f} (arc loss = {arc_loss_train:.4f},"
                  f" rel loss = {rel_loss_train:.4f}), Dev Loss: {avg_dev_loss:.4f} (arc loss = {arc_loss_dev:.4f}, rel loss = {rel_loss_dev:.4f})")

        return self.history

    def _run_epoch(self, dataloader, train):
        total_loss = 0
        total_arc_loss = 0
        total_rel_loss = 0

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
                arc_loss = self._compute_arc_loss(s_arc, governors)
                rel_loss = self._compute_label_loss(s_rel, governors, labels)

                loss = arc_loss + rel_loss
                if train:
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()
                total_arc_loss += arc_loss.item()
                total_rel_loss += rel_loss.item()

                # Update tqdm progress bar with the current loss
                pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        avg_arc_loss = total_arc_loss / len(dataloader)
        avg_rel_loss = total_rel_loss / len(dataloader)

        loss_key = "train_loss" if train else "dev_loss"
        self.history[loss_key].append(avg_loss)

        return avg_loss, avg_arc_loss, avg_rel_loss

    def _compute_arc_loss(self, S_arcs, heads):
        """
        Computes arc dependency loss while handling padding.

        S_arcs: (batch_size, seq_length, seq_length) - Model predictions
        heads: (batch_size, seq_length) - Ground truth head indices
        """
        heads = heads.to(self.device)

        batch_size, seq_length, _ = S_arcs.shape

        # Reshape output to (batch_size * seq_length, vocab_size)
        S_arcs = S_arcs.view(-1, seq_length)

        # Reshape target to (batch_size * seq_length)
        heads = heads.view(-1)

        # Apply mask to ignore PAD_TOKEN_VAL
        mask = heads != cf.PAD_TOKEN_VAL

        valid_arcs = S_arcs[mask]
        valid_heads = heads[mask]

        return self.criterion(valid_arcs, valid_heads)

    def _compute_label_loss(self, S_rel, heads, labels):
        """
           Computes label dependency loss on the gold arcs while handling padding.

           S_rel: (batch_size, seq_length, seq_length, num_labels) - Model predictions
           heads: (batch_size, seq_length) - Ground truth head indices
           labels: (batch_size, seq_length) - Ground truth label indices
        """

        heads = heads.to(self.device)  # (B, L)
        labels = labels.to(self.device)  # (B, L)

        heads = heads.unsqueeze(-1).unsqueeze(-1)  # (B, L, 1, 1)
        heads = heads.expand(-1, -1, -1, S_rel.size(3))  # (B, L, 1, c)
        # Select only the true head-dependent pairs
        S_rel_gold = torch.gather(S_rel, 2, heads).squeeze(2)  # (B, L, C)
        # Reshape for loss computation
        S_rel_gold = S_rel_gold.view(-1, S_rel_gold.size(-1))  # (B * L, C)
        labels = labels.view(-1)  # (B * L)
        # print(labels.shape)
        # Apply mask to ignore PAD_TOKEN_VAL
        mask = labels != cf.PAD_TOKEN_VAL  # (B, 1, L, 1)
        # print(mask.shape)
        mask = mask.squeeze()  # Remove extra dims
        # print(mask.shape)

        valid_output = S_rel_gold[mask]  # Correct shape (N_valid, c)
        valid_target = labels[mask]  # Correct shape (N_valid,)

        return self.criterion(valid_output, valid_target)
