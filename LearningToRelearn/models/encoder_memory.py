import itertools
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AdamW
import wandb

import LearningToRelearn.datasets.utils as dataset_utils
import LearningToRelearn.models.utils as model_utils
from LearningToRelearn.models.base_models import TransformerRLN, TransformerClsModel, MemoryStore,\
                                                 LSTMDecoder, SimpleDecoder,\
                                                 TRANSFORMER_HDIM
from LearningToRelearn.learner import Learner
from LearningToRelearn.datasets.text_classification_dataset import get_continuum, alternating_order, datasets_dict

class EncoderMemory(Learner):
    def __init__(self, config, **kwargs):
        """
        Baseline models: sequential and multitask setup.
        """
        self.key_dim = config.learner.key_dim  # needs to be before super init
        super().__init__(config, **kwargs)

        self.lr = config.learner.lr
        self.type = config.learner.type
        self.n_epochs = config.training.epochs
        self.log_freq = config.training.log_freq
        self.loss_fn = nn.CrossEntropyLoss()

        self.n_neighbours = config.learner.n_neighbours

        # self.memory = MemoryStore(memory_size=config.learner.memory_size,
        #                           key_dim=self.key_dim,
        #                           device=self.device)
        self.logger.info(f"Instantiated memory of size {config.learner.memory_size} with key dimension {self.key_dim}")
        self.encoder = TransformerRLN(model_name=config.learner.model_name,
                                      max_length=config.data.max_length,
                                      device=self.device)
        self.logger.info("Loaded {} as model".format(self.encoder.__class__.__name__))
        # self.decoder = LSTMDecoder(key_size=config.learner.key_dim, embedding_size=TRANSFORMER_HDIM).to(self.device)
        dimensions = [TRANSFORMER_HDIM] + list(self.key_dim)
        self.key_encoders = [nn.Linear(dim, next_dim).to(self.device) for dim, next_dim in zip(dimensions, dimensions[1:])] 
        self.key_decoders = [nn.Linear(next_dim, dim).to(self.device) for dim, next_dim in zip(dimensions, dimensions[1:])] 
        self.logger.info(f"Key encoders: {self.key_encoders} -- key decoders: {self.key_decoders}")
        # self.key_encoder = nn.Linear(TRANSFORMER_HDIM, self.key_dim).to(self.device)
        # self.key_decoder = nn.Linear(self.key_dim, TRANSFORMER_HDIM).to(self.device)
        # self.key_encoder = lambda x: x  # for now
        # self.key_decoder = lambda x: x  # for now
        self.classifier = nn.Linear(TRANSFORMER_HDIM, config.data.n_classes).to(self.device)
        # self.key_classifiers = nn.Linear(self.key_dim, config.data.n_classes).to(self.device)
        self.key_classifiers = [nn.Linear(dim, config.data.n_classes).to(self.device) for dim in self.key_dim]
        self.logger.info(f"Key classifiers: {self.key_classifiers}")

        self.optimizer = AdamW([p for p in
                                itertools.chain(self.encoder.parameters(),
                                                *[key_encoder.parameters() for key_encoder in self.key_encoders],
                                                *[key_decoder.parameters() for key_decoder in self.key_decoders],
                                                *[key_classifier.parameters() for key_classifier in self.key_classifiers],
                                                self.classifier.parameters()
                                )
                                if p.requires_grad],
                                lr=self.lr)

    def training(self, datasets, **kwargs):
        # train_datasets = {dataset_name: dataset for dataset_name, dataset in zip(datasets["order"], datasets["train"])}
        train_datasets = datasets_dict(datasets["train"], datasets["order"])
        val_datasets = datasets_dict(datasets["val"], datasets["order"])

        samples_per_task = self.config.learner.samples_per_task
        order = self.config.task_order if self.config.task_order is not None else datasets["order"]
        n_samples = [samples_per_task] * len(order) if samples_per_task is not None else samples_per_task
        dataset = get_continuum(train_datasets, order=order, n_samples=n_samples)
        dataloader = DataLoader(dataset, batch_size=self.mini_batch_size, shuffle=False)

        for text, labels, datasets in dataloader:
            output = self.training_step(text, labels)
            predictions = model_utils.make_prediction(output["logits"].detach())
            # for logging
            key_predictions = [
                model_utils.make_prediction(key_logits.detach()) for key_logits in output["key_logits"]
            ] 
            # self.logger.debug(f"accuracy prediction from key embedding: {key_metrics['accuracy']}")

            self.update_tracker(output, predictions, key_predictions, labels)
            online_metrics = model_utils.calculate_metrics(predictions.tolist(), labels.tolist())
            self.metrics["online"].append({
                "accuracy": online_metrics["accuracy"],
                "examples_seen": self.examples_seen(),
                "task": datasets[0]  # assumes whole batch is from same task
            })
            if self.current_iter % self.log_freq == 0:
                self.log()
                self.write_metrics()
            if self.current_iter % self.validate_freq == 0:
                self.validate(val_datasets, n_samples=self.config.training.n_validation_samples)
            self.current_iter += 1

    def training_step(self, text, labels):
        self.set_train()
        labels = torch.tensor(labels).to(self.device)
        output = self.forward(text, labels)

        # compute losses
        loss = self.loss_fn(output["logits"], labels)
        key_losses = [self.loss_fn(key_logits, labels) for key_logits in output["key_logits"]]

        # update here
        self.optimizer.zero_grad()

        loss.backward(retain_graph=True)
        for reconstruction_error in output["reconstruction_errors"]:
            reconstruction_error.backward(retain_graph=True)
        for key_loss in key_losses[:-1]:
            key_loss.backward(retain_graph=True)
        key_losses[-1].backward()
        self.optimizer.step()
        loss = loss.item()
        key_losses = [key_loss.item() for key_loss in key_losses]
        # key_loss = 0
        self.logger.debug(f"Loss: {loss} -- key_loss: {key_losses} -- reconstruction errors: {[re.item() for re in output['reconstruction_errors']]}")
        # self.logger.debug(f"Key Loss: {key_loss}")
        return {
            "logits": output["logits"],
            "key_logits": output["key_logits"],
            "loss": loss,
            "key_losses": key_losses,
            "reconstruction_errors": [reconstruction_error.item() for reconstruction_error in output["reconstruction_errors"]]
        }

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
        key_embeddings = self.encode_keys(text_embedding)
        reconstructions = [key_decoder(key_embedding) for key_decoder, key_embedding in zip(self.key_decoders, key_embeddings)]
        reconstruction_errors = [
            ((real_embedding.detach() - reconstruction) ** 2).mean()
            for real_embedding, reconstruction in zip([text_embedding] + key_embeddings[:-1], reconstructions)
        ]

        # query_result = self.memory.query(key_embedding, self.n_neighbours)
        # prediction_embedding = self.decoder(text_embedding, query_result)
        logits = self.classifier(text_embedding)
        key_logits = [key_classifier(key_embedding) for key_classifier, key_embedding in zip(self.key_classifiers, key_embeddings)]

        # if update_memory:
        #     self.memory.add_entry(embeddings=key_embedding.detach(), labels=labels, query_result=query_result)

        return {
            "logits": logits,
            "key_logits": key_logits,
            "reconstructions": reconstructions,
            "reconstruction_errors": reconstruction_errors
        }

    def encode_keys(self, embedding):
        """
        Encode an embedding into key embeddings.
        Each key embedding is compressed using the previous key's embedding.

        Parameters
        ---
        embedding: Tensor, shape (BATCH, TRANSFORMER_HDIM)
            Embedding to be mapped to key embeddings.

        Returns
        ---
        List[tensor], one element for each key_encoder, each tensor of shape BATCH, KEY_DIM corresponding
        to that encoder, specified by self.key_dims.
        """
        key_embeddings = []
        for key_encoder in self.key_encoders:
            # TODO: use embedding.detach() to block gradients between key embedding layers?
            embedding = key_encoder(embedding)
            key_embeddings.append(embedding)
        return key_embeddings

    # def decode_keys(self, key_embeddings):
    #     """
    #     Parameters
    #     ---
    #     key_embeddings: List[tensor]
    #         Each tensor is a key embedding of shape (BATCH, key_size), sizes in the same order as
    #         self.key_dim.
        
    #     Returns
    #     ---
    #     List[tensor], one element for each key_decoder, each tensor of shape (BATCH, prev_dim),
    #     where prev_dim is the dimension of the previous key encoder. The first element should have the 
    #     """
    #     # TODO: instead of going from key
    #     decoded = [key_decoder(key_embedding) for key_decoder, key_embedding in zip(self.key_decoders, self.key_embeddings)]
    #     pass

    def reset_tracker(self):
        """Initializes dictionary that stores performance data during training for logging purposes."""
        self.tracker = {
            "losses": [],
            "key_losses": [[] for _ in range(len(self.key_dim))],
            "reconstruction_errors": [[] for _ in range(len(self.key_dim))],
            "predictions": [],
            "key_predictions": [[] for _ in range(len(self.key_dim))],
            "labels": []
        }

    def update_tracker(self, output, predictions, key_predictions, labels):
        self.tracker["losses"].append(output["loss"])
        self.tracker["predictions"].extend(predictions.tolist())
        self.tracker["labels"].extend(labels.tolist())
        for i in range(len(self.key_dim)):
            self.tracker["key_losses"][i].append(output["key_losses"][i])
            self.tracker["reconstruction_errors"][i].append(output["reconstruction_errors"][i])
            self.tracker["key_predictions"][i].extend(key_predictions[i].tolist())

    def log(self):
        """Log results during training to console and optionally other outputs

        Parameters
        ---
        metrics: dict mapping metric names to their values
        """
        loss = np.mean(self.tracker["losses"])
        key_losses = [np.mean(key_losses) for key_losses in self.tracker["key_losses"]]
        reconstruction_errors = [np.mean(reconstruction_errors) for reconstruction_errors in self.tracker["reconstruction_errors"]]
        metrics = model_utils.calculate_metrics(self.tracker["predictions"], self.tracker["labels"])
        key_metrics = [
            model_utils.calculate_metrics(key_predictions, self.tracker["labels"])
            for key_predictions in self.tracker["key_predictions"]
        ]
        key_accuracy_str = [f'{km["accuracy"]:.4f}' for km in key_metrics]
        self.logger.info(
            f"Iteration {self.current_iter + 1} - Metrics: Loss = {loss:.4f}, "
        f"key loss = {[f'{key_loss:.4f}' for key_loss in key_losses]}, "
            f"reconstruction error = {[f'{reconstruction_error:.4f}' for reconstruction_error in reconstruction_errors]}, "
            f"accuracy = {metrics['accuracy']:.4f} - "
            f"key accuracy = {key_accuracy_str}"
        )
        if self.config.wandb:
            log = {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "loss": loss,
                "examples_seen": self.examples_seen()
            }
            for i, dim in enumerate(self.key_dim):
                log[f"key_accuracy_encoder_{i}_dim_{dim}"] = key_metrics[i]["accuracy"]
                log[f"key_loss_encoder_{i}_dim_{dim}"] = key_losses[i]
                log[f"reconstruction_error_encoder_{i}_dim_{dim}"] = reconstruction_errors[i]
            wandb.log(log)
        self.reset_tracker()

    def examples_seen(self):
        return (self.current_iter + 1) * self.mini_batch_size

    def evaluate(self, dataloader, update_memory=False):
        self.set_eval()
        all_losses, all_predictions, all_labels = [], [], []

        self.logger.info("Starting evaluation...")
        for i, (text, labels, datasets) in enumerate(dataloader):
            labels = torch.tensor(labels).to(self.device)
            with torch.no_grad():
                output = self.forward(text, labels, update_memory=update_memory)
                logits = output["logits"]
                loss = self.loss_fn(logits, labels)
            loss = loss.item()
            pred = model_utils.make_prediction(logits.detach())
            all_losses.append(loss)
            all_predictions.extend(pred.tolist())
            all_labels.extend(labels.tolist())
            if i % 20 == 0:
                self.logger.info(f"Batch {i + 1}/{len(dataloader)} processed")

        results = model_utils.calculate_metrics(all_predictions, all_labels)
        self.logger.info("Test metrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, "
                    "F1 score = {:.4f}".format(np.mean(all_losses), results["accuracy"],
                    results["precision"], results["recall"], results["f1"]))

        return results

    def model_state(self):
        state = {
            "encoder": self.encoder.state_dict(),
            "classifier": self.classifier.state_dict(),
        }
        for i in range(len(self.key_dim)):
            state[f"key_classifier_{i}"] = self.key_classifiers[i].state_dict()
            state[f"key_encoder_{i}"] = self.key_encoders[i].state_dict()
            state[f"key_decoder_{i}"] = self.key_decoders[i].state_dict()

        return state

    def load_model_state(self, checkpoint):
        self.encoder.load_state_dict(checkpoint["model_state"]["encoder"])
        self.classifier.load_state_dict(checkpoint["model_state"]["classifier"])
        for i in range(len(self.key_dim)):
            self.key_classifiers[i].load_state_dict(checkpoint["model_state"][f"key_classifier_{i}"])
            self.key_encoders[i].load_state_dict(checkpoint["model_state"][f"key_encoder_{i}"])
            self.key_decoders[i].load_state_dict(checkpoint["model_state"][f"key_decoder_{i}"])

    # def optimizer_state(self):
    #     return self.meta_optimizer.state_dict()

    # def load_optimizer_state(self, checkpoint):
    #     self.meta_optimizer.load_state_dict(checkpoint["optimizer"])

    def save_other_state_information(self, state):
        """Any learner specific state information is added here"""
        # state["memory"] = self.memory
        return state

    def load_other_state_information(self, checkpoint):
        """Any learner specific state information is loaded here"""
        pass
        # self.memory = checkpoint["memory"]

    def set_train(self):
        """Set underlying pytorch network to train mode.
        
        If learner has multiple models, this method should be overwritten.
        """
        self.encoder.train()

    def set_eval(self):
        """Set underlying pytorch network to evaluation mode.
        
        If learner has multiple models, this method should be overwritten.
        """
        self.encoder.eval()
