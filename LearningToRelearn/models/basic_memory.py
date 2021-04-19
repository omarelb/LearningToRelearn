import itertools
import time

import numpy as np
import torch
from torch import nn
from torch.utils import data
from transformers import AdamW
import wandb

import LearningToRelearn.datasets.utils as dataset_utils
import LearningToRelearn.models.utils as model_utils
from LearningToRelearn.models.base_models import TransformerRLN, TransformerClsModel, MemoryStore,\
                                                 LSTMDecoder, SimpleDecoder,\
                                                 TRANSFORMER_HDIM
from LearningToRelearn.learner import Learner
from LearningToRelearn.datasets.text_classification_dataset import get_continuum, alternating_order, datasets_dict

# logging.basicConfig(level="INFO", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# logger = logging.getLogger("Baseline-Log")

DECODER_MAP = {
    'lstm': LSTMDecoder,
    'simple': SimpleDecoder
}

class BasicMemory(Learner):
    def __init__(self, config, **kwargs):
        """
        Baseline models: sequential and multitask setup.
        """
        super().__init__(config, **kwargs)

        self.lr = config.learner.lr
        self.type = config.learner.type
        self.n_epochs = config.training.epochs
        self.log_freq = config.training.log_freq
        self.loss_fn = nn.CrossEntropyLoss()

        self.key_dim = config.learner.key_dim
        self.n_neighbours = config.learner.n_neighbours

        self.memory = MemoryStore(memory_size=config.learner.memory_size,
                                  key_dim=self.key_dim,
                                  device=self.device)
        self.logger.info(f"Instantiated memory of size {config.learner.memory_size} with key dimension {self.key_dim}")
        self.encoder = TransformerRLN(model_name=config.learner.model_name,
                                      max_length=config.data.max_length,
                                      device=self.device)
        self.logger.info("Loaded {} as model".format(self.encoder.__class__.__name__))
        # self.decoder = LSTMDecoder(key_size=config.learner.key_dim, embedding_size=TRANSFORMER_HDIM).to(self.device)
        decoder = DECODER_MAP[config.learner.decoder]
        self.decoder = decoder(key_size=config.learner.key_dim, embedding_size=TRANSFORMER_HDIM).to(self.device)
        # self.key_encoder = nn.Linear(TRANSFORMER_HDIM, self.key_dim).to(self.device)
        # self.key_decoder = nn.Linear(self.key_dim, TRANSFORMER_HDIM).to(self.device)
        # self.decoder = lambda x, y: x
        self.key_encoder = lambda x: x # for now
        self.key_decoder = lambda x: x # for now
        self.classifier = nn.Linear(TRANSFORMER_HDIM, config.data.n_classes).to(self.device)
        self.key_classifier = nn.Linear(self.key_dim, config.data.n_classes).to(self.device)

        # self.optimizer = AdamW([p for p in
        #                         self.encoder.parameters() + self.decoder.parameters() + self.key_encoder.parameters()
        #                         + self.key_decoder.parameters() + self.key_classifier.parameters() + self.classifier.parameters()
        #                         if p.requires_grad],
        #                         lr=self.lr)
        self.optimizer = AdamW([p for p in
                                itertools.chain(self.encoder.parameters(),
                                                self.decoder.parameters(),
                                                self.classifier.parameters()) 
                                if p.requires_grad],
                                lr=self.lr)

    def forward(self, text, labels, update_memory=True):
        """
        Forward pass using memory architecture.

        Parameters
        ---
        text
        labels
        update_memory: bool
            If false, don't update memory data
        """
        input_dict = self.encoder.encode_text(text)
        text_embedding = self.encoder(input_dict)
        key_embedding = self.key_encoder(text_embedding)

        query_result = self.memory.query(key_embedding, self.n_neighbours)
        prediction_embedding = self.decoder(text_embedding, query_result)
        logits = self.classifier(prediction_embedding)
        key_logits = self.key_classifier(key_embedding)

        if update_memory:
            self.memory.add_entry(embeddings=key_embedding.detach(), labels=labels, query_result=query_result)

        return logits, key_logits

    def training(self, datasets, **kwargs):
        # train_datasets = {dataset_name: dataset for dataset_name, dataset in zip(datasets["order"], datasets["train"])}
        train_datasets = datasets_dict(datasets["train"], datasets["order"])
        val_datasets = datasets_dict(datasets["val"], datasets["order"])

        dataset = get_continuum(train_datasets, order=datasets["order"], n_samples=[5000] * len(datasets["order"]))

        dataloader = data.DataLoader(dataset, batch_size=self.mini_batch_size, shuffle=False)
        # relearning_dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False)
        data_length = len(dataloader)

        all_losses, all_key_losses, all_predictions, all_labels = [], [], [], []

        for text, labels, datasets in dataloader:
            self.encoder.train()
            labels = torch.tensor(labels).to(self.device)
            task = datasets[0]

            logits, key_logits = self.forward(text, labels)
            # compute losses
            loss = self.loss_fn(logits, labels)
            # key_loss = self.loss_fn(key_logits, labels)
            # update here
            
            self.optimizer.zero_grad()
            loss.backward()
            # key_loss.backward()
            self.optimizer.step()
            loss = loss.item()
            # key_loss = key_loss.item()
            key_loss = 0
            self.logger.debug(f"Loss: {loss}")
            # self.logger.debug(f"Key Loss: {key_loss}")
            pred = model_utils.make_prediction(logits.detach())

            all_losses.append(loss)
            all_key_losses.append(key_loss)
            all_predictions.extend(pred.tolist())
            all_labels.extend(labels.tolist())

            acc, prec, rec, f1 = model_utils.calculate_metrics(all_predictions, all_labels)
            self.metrics["online"].append({
                "accuracy": acc,
                "examples_seen": self.examples_seen(),
                "task": task
            })
            if self.current_iter % self.log_freq == 0:
                time_per_iteration, estimated_time_left = self.time_metrics(data_length)
                self.logger.info(
                    "Iteration {}/{} ({:.2f}%) -- {:.3f} (sec/it) -- Time Left: {}\nMetrics: Loss = {:.4f}, key loss: {:.4f}, accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, "
                    "F1 score = {:.4f}".format(self.current_iter + 1, data_length, (self.current_iter + 1) / data_length * 100,
                                            time_per_iteration, estimated_time_left,
                                            np.mean(all_losses), np.mean(all_key_losses), acc, prec, rec, f1))
                if self.config.wandb:
                    wandb.log({
                        "accuracy": acc,
                        "precision": prec,
                        "recall": rec,
                        "f1": f1,
                        "loss": np.mean(all_losses),
                        "key_loss": np.mean(all_key_losses),
                        "examples_seen": self.examples_seen()
                    })
                all_losses, all_key_losses, all_predictions, all_labels = [], [], [], []
                self.start_time = time.time()
            if self.current_iter % self.validate_freq == 0:
                self.validate(val_datasets, n_samples=self.config.training.n_validation_samples)
            self.time_checkpoint()
            self.current_iter += 1

    def examples_seen(self):
        return (self.current_iter + 1) * self.mini_batch_size


    def testing(self, datasets, order):
        """
        Parameters
        ---
        datasets: List[Dataset]
            Test datasets.
        order: List[str]
            Specifies order of encountered datasets
        """
        accuracies, precisions, recalls, f1s = [], [], [], []
        results = {}
        # only have one dataset if type is single
        if self.type == "single":
            train_dataset = datasets[order.index(self.config.learner.dataset)]
            datasets = [train_dataset]
        for dataset in datasets:
            dataset_name = dataset.__class__.__name__
            self.logger.info("Testing on {}".format(dataset_name))
            test_dataloader = data.DataLoader(dataset, batch_size=self.mini_batch_size, shuffle=False,
                                         collate_fn=dataset_utils.batch_encode)
            dataset_results = self.evaluate(dataloader=test_dataloader)
            accuracies.append(dataset_results["accuracy"])
            precisions.append(dataset_results["precision"])
            recalls.append(dataset_results["recall"])
            f1s.append(dataset_results["f1"])
            results[dataset_name] = dataset_results

        mean_results = {
            "accuracy": np.mean(accuracies),
            "precision": np.mean(precisions),
            "recall": np.mean(recalls),
            "f1": np.mean(f1s)
        }
        self.logger.info("Overall test metrics: Accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, "
                    "F1 score = {:.4f}".format(
                        mean_results["accuracy"], mean_results["precision"], mean_results["recall"],
                        mean_results["f1"]
                    ))
        return results, mean_results

    def set_eval(self):
        """Set all network components to evaluate mode"""
        self.encoder.eval()

    def evaluate(self, dataloader):
        all_losses, all_predictions, all_labels = [], [], []

        self.set_eval()

        for i, (text, labels, datasets) in enumerate(dataloader):
            labels = torch.tensor(labels).to(self.device)
            with torch.no_grad():
                logits, key_logits = self.forward(text, labels, update_memory=False)
                loss = self.loss_fn(logits, labels)
            loss = loss.item()
            pred = model_utils.make_prediction(logits.detach())
            all_losses.append(loss)
            all_predictions.extend(pred.tolist())
            all_labels.extend(labels.tolist())
            if i % 20 == 0:
                self.logger.info(f"Batch {i + 1}/{len(dataloader)} processed")

        acc, prec, rec, f1 = model_utils.calculate_metrics(all_predictions, all_labels)
        self.logger.info("Test metrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, "
                    "F1 score = {:.4f}".format(np.mean(all_losses), acc, prec, rec, f1))

        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    def model_state(self):
        return {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "classifier": self.classifier.state_dict(),
            "key_classifier": self.key_classifier.state_dict(),
        }

    def load_model_state(self, checkpoint):
        self.encoder.load_state_dict(checkpoint["model_state"]["encoder"])
        self.decoder.load_state_dict(checkpoint["model_state"]["decoder"])
        self.classifier.load_state_dict(checkpoint["model_state"]["classifier"])
        self.key_classifier.load_state_dict(checkpoint["model_state"]["key_classifier"])

    # def optimizer_state(self):
    #     return self.meta_optimizer.state_dict()


    # def load_optimizer_state(self, checkpoint):
    #     self.meta_optimizer.load_state_dict(checkpoint["optimizer"])

    def save_other_state_information(self, state):
        """Any learner specific state information is added here"""
        state["memory"] = self.memory
        return state

    def load_other_state_information(self, checkpoint):
        """Any learner specific state information is loaded here"""
        self.memory = checkpoint["memory"]