import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import src.config as cf
from src.modules import TransformerEncoder, RecurrentEncoder


class Trainer:
    """Encapsulates training and evaluation logic."""

    def __init__(self, model, optimizer, criterion, batch_size, device=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.history = {"train_loss": [], "train_arc_loss": [], "train_rel_loss": [],
                        "dev_loss": [], "dev_arc_loss": [], "dev_rel_loss": []}
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, train_data, dev_data, nb_epochs):
        print(train_data.keys())
        extracted_train_data = [tensor for key, tensor in train_data.items() if key.startswith("extracted")]
        train_dataset = TensorDataset(*extracted_train_data)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        print(dev_data.keys())
        extracted_dev_data = [tensor for key, tensor in dev_data.items() if key.startswith("extracted")]
        dev_dataset = TensorDataset(*extracted_dev_data)
        dev_loader = DataLoader(dev_dataset, batch_size=self.batch_size, shuffle=False)

        for epoch in range(nb_epochs):
            self.model.train()
            avg_train_loss, arc_loss_train, rel_loss_train = self._run_epoch(train_loader, train=True)

            self.model.eval()
            avg_dev_loss, arc_loss_dev, rel_loss_dev = self._run_epoch(dev_loader, train=False)

            print(
                f"Epoch {epoch + 1}/{nb_epochs}, Training Loss: {avg_train_loss:.4f} (arc loss = {arc_loss_train:.4f},"
                f" rel loss = {rel_loss_train:.4f}), Dev Loss: {avg_dev_loss:.4f} (arc loss = {arc_loss_dev:.4f}, rel loss = {rel_loss_dev:.4f})")

        return self.history

    def _run_epoch(self, dataloader, train):
        total_loss = 0
        total_arc_loss = 0
        total_rel_loss = 0

        # Use tqdm for progress tracking
        with tqdm(dataloader, unit="batch") as pbar:
            for batch in pbar:
                self.optimizer.zero_grad()

                if isinstance(self.model.encoder, RecurrentEncoder):
                    forms, upos, heads, deprels = batch

                    forms = forms.to(self.device)
                    upos = upos.to(self.device)
                    heads = heads.to(self.device)
                    deprels = deprels.to(self.device)

                    s_arc, s_rel = self.model(train, forms, upos)

                elif isinstance(self.model.encoder, TransformerEncoder):
                    forms, upos, heads, deprels, contextual_embeddings = batch

                    forms = forms.to(self.device)
                    upos = upos.to(self.device)
                    heads = heads.to(self.device)
                    deprels = deprels.to(self.device)
                    contextual_embeddings = contextual_embeddings.to(self.device)

                    s_arc, s_rel = self.model(train, contextual_embeddings, upos)

                arc_loss = self._compute_arc_loss(s_arc, heads)
                rel_loss = self._compute_rel_loss(s_rel, heads, deprels)

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

        for key, value in zip(
                ["loss", "arc_loss", "rel_loss"],
                [avg_loss, avg_arc_loss, avg_rel_loss]
        ):
            self.history[f'{"train" if train else "dev"}_{key}'].append(value)

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

    def _compute_rel_loss(self, S_rel, heads, deprels):
        """
           Computes rel dependency loss on the gold arcs while handling padding.

           S_rel: (batch_size, seq_length, seq_length, num_deprels) - Model predictions
           heads: (batch_size, seq_length) - Ground truth head indices
           deprels: (batch_size, seq_length) - Ground truth deprel indices
        """

        heads = heads.to(self.device)  # (B, L)
        deprels = deprels.to(self.device)  # (B, L)

        heads = heads.unsqueeze(-1).unsqueeze(-1)  # (B, L, 1, 1)
        heads = heads.expand(-1, -1, -1, S_rel.size(3))  # (B, L, 1, c)

        # Select only the true head-dependent pairs
        S_rel_gold = torch.gather(S_rel, 2, heads).squeeze(2)  # (B, L, C)

        # Reshape for loss computation
        S_rel_gold = S_rel_gold.view(-1, S_rel_gold.size(-1))  # (B * L, C)
        deprels = deprels.view(-1)  # (B * L)

        # Apply mask to ignore PAD_TOKEN_VAL
        mask = deprels != cf.PAD_TOKEN_VAL  # (B, 1, L, 1)

        valid_hdp = S_rel_gold[mask]
        valid_deprels = deprels[mask]

        return self.criterion(valid_hdp, valid_deprels)
