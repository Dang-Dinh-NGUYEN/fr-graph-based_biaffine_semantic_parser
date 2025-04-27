import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import src.config as cf
from src.modules import TransformerEncoder, RecurrentEncoder
from src.tools import CustomizedDataset, dynamic_collate_fn
from termcolor import colored


class Trainer:
    """Encapsulates training and evaluation logic with Early Stopping."""

    def __init__(self, model, optimizer, arc_loss_function, label_loss_function, batch_size, patience=5, device=None):
        self.model = model
        self.optimizer = optimizer
        self.arc_loss_function = arc_loss_function
        self.label_loss_function = label_loss_function
        self.batch_size = batch_size
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Early stopping variables
        self.patience = patience
        self.best_dev_loss = float('inf')
        self.epochs_without_improvement = 0
        self.best_model_state = None  # Store the best model

        self.history = {"train_loss": [], "train_arc_loss": [], "train_rel_loss": [],
                        "dev_loss": [], "dev_arc_loss": [], "dev_rel_loss": []}

    def train(self, train_data, dev_data, nb_epochs):
        print(train_data.keys())

        extracted_train_data = {key: tensor for key, tensor in train_data.items() if not key.endswith("vocab")}
        train_dataset = CustomizedDataset(extracted_train_data, required_keys=extracted_train_data.keys())
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                  collate_fn=dynamic_collate_fn)

        print(dev_data.keys())
        extracted_dev_data = {key: tensor for key, tensor in dev_data.items() if not key.endswith("vocab")}
        dev_dataset = CustomizedDataset(extracted_dev_data, required_keys=extracted_dev_data.keys())
        dev_loader = DataLoader(dev_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=dynamic_collate_fn)

        for epoch in range(nb_epochs):
            self.model.train()
            avg_train_loss, arc_loss_train, rel_loss_train = self._run_epoch(train_loader, train=True)

            self.model.eval()
            avg_dev_loss, arc_loss_dev, rel_loss_dev = self._run_epoch(dev_loader, train=False)

            print(
                f"Epoch {epoch + 1}/{nb_epochs}, Training Loss: {avg_train_loss:.4f} (arc loss = {arc_loss_train:.4f},"
                f" rel loss = {rel_loss_train:.4f}), Dev Loss: {avg_dev_loss:.4f} (arc loss = {arc_loss_dev:.4f}, rel loss = {rel_loss_dev:.4f})")

            # Early stopping check
            if avg_dev_loss < self.best_dev_loss:
                print(colored(f"Improvement detected! Saving model at epoch {epoch + 1}.", "green"))
                self.best_dev_loss = avg_dev_loss
                self.epochs_without_improvement = 0
                self.best_model_state = self.model.state_dict()  # Save the best model state
            else:
                self.epochs_without_improvement += 1
                print(colored(f"No improvement for {self.epochs_without_improvement} epochs.", "red"))

            # Stop training if patience is exceeded
            if self.epochs_without_improvement >= self.patience:
                print(colored(f"Early stopping triggered after {self.patience} epochs without improvement!", "red"))
                break  # Exit training loop

        # Restore the best model before exiting
        if self.best_model_state:
            print(colored("Restoring the best model state.", "blue"))
            self.model.load_state_dict(self.best_model_state)

        return self.history

    def _run_epoch(self, dataloader, train):
        total_loss = 0
        total_arc_loss = 0
        total_rel_loss = 0

        with tqdm(dataloader, unit="batch") as pbar:
            for batch in pbar:
                self.optimizer.zero_grad()
                inputs = [v.to(self.device) for k, v in batch.items() if
                          k not in ['extracted_head', 'extracted_deprel']]
                heads = batch.get('extracted_head').to(self.device)
                deprels = batch.get('extracted_deprel').to(self.device)

                s_arc, s_rel = self.model(train, *inputs)

                arc_loss = self._compute_arc_loss(s_arc, heads)
                rel_loss = self._compute_rel_loss(s_rel, heads, deprels)

                loss = 1 * arc_loss + 1 * rel_loss
                if train:
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()
                total_arc_loss += arc_loss.item()
                total_rel_loss += rel_loss.item()

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
        # print(S_arcs.shape)
        # print(heads.shape)
        heads = heads.to(self.device)

        batch_size, seq_length, _ = S_arcs.shape
        S_arcs = S_arcs.view(-1, seq_length)
        heads = heads.view(-1, seq_length)

        # mask = heads != self.model.encoder.pad_token_id
        mask = heads.sum(dim=-1) != 0
        valid_arcs = S_arcs[mask].float()
        # print(valid_arcs.shape)
        valid_heads = heads[mask].float()
        # print(valid_heads.shape)
        return self.arc_loss_function(valid_arcs, valid_heads)

    def _compute_rel_loss(self, S_rel, heads, deprels):
        heads = heads.to(self.device)
        deprels = deprels.to(self.device)

        heads_mask = heads.unsqueeze(-1)  # (B, L, 1, 1)
        # heads = heads.expand(-1, -1, -1, S_rel.size(3))  # (B, L, 1, c)
        # heads_mask = heads.unsqueeze(1)
        # S_rel_gold = torch.gather(S_rel, 2, heads).squeeze(2)  # (B, L, C)
        # S_rel_gold = S_rel_gold.view(-1, S_rel_gold.size(-1))  # (B * L, C)
        S_rel_gold = S_rel * heads_mask
        # print(S_rel_gold.shape)
        S_rel_gold = S_rel_gold.sum(dim=2)
        # deprels = deprels.view(-1)  # (B * L)

        heads_mask = (heads != 0)
        # print(heads_mask.shape)
        # mask = deprels != self.model.encoder.pad_token_id  # (B, 1, L, 1)

        valid_hdp = S_rel[heads_mask]
        valid_deprels = deprels[heads_mask]

        return self.label_loss_function(valid_hdp, valid_deprels)
