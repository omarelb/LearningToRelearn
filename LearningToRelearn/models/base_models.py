from LearningToRelearn.models.utils import euclidean_dist
import math
import random
from collections import defaultdict, Counter

import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertModel

TRANSFORMER_HDIM = 768

class TransformerClsModel(nn.Module):

    def __init__(self, model_name, n_classes, max_length, device):
        super().__init__()
        self.n_classes = n_classes
        self.max_length = max_length
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.linear = nn.Linear(TRANSFORMER_HDIM, n_classes)
        self.to(self.device)

    def encode_text(self, text):
        encode_result = self.tokenizer.batch_encode_plus(text, return_token_type_ids=False, max_length=self.max_length,
                                                         truncation=True, padding="max_length", return_tensors="pt")
        for key in encode_result:
            encode_result[key] = encode_result[key].to(self.device)
        return encode_result

    def forward(self, inputs, out_from="full"):
        if out_from in ("full", "transformers"):
            last_hidden_state = self.encoder(inputs["input_ids"], attention_mask=inputs["attention_mask"])[0]
            cls_representation = last_hidden_state[:,0,:] # cls representation
        if out_from == "full":
            return self.linear(cls_representation)
        elif out_from == "transformers":
            return cls_representation
        elif out_from == "linear":
            return self.linear(inputs)
        else:
            raise ValueError("Invalid value of argument")


class TransformerRLN(nn.Module):

    def __init__(self, model_name, max_length, device):
        super().__init__()
        self.max_length = max_length
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.to(self.device)

    def encode_text(self, text):
        encode_result = self.tokenizer.batch_encode_plus(text, return_token_type_ids=False, max_length=self.max_length,
                                                         truncation=True, padding="max_length", return_tensors="pt")
        for key in encode_result:
            encode_result[key] = encode_result[key].to(self.device)
        return encode_result

    def forward(self, inputs):
        last_hidden_state = self.encoder(inputs["input_ids"], attention_mask=inputs["attention_mask"])[0]
        cls_representation = last_hidden_state[: ,0, :]
        return cls_representation


class LinearPLN(nn.Module):

    def __init__(self, in_dim, out_dim, device):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.linear.to(device)

    def forward(self, input):
        return self.linear(input)


class TransformerNeuromodulator(nn.Module):

    def __init__(self, model_name, device):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder.requires_grad = False
        self.linear = nn.Sequential(nn.Linear(TRANSFORMER_HDIM, TRANSFORMER_HDIM),
                                    nn.ReLU(),
                                    nn.Linear(TRANSFORMER_HDIM, TRANSFORMER_HDIM),
                                    nn.Sigmoid())
        self.to(self.device)

    def forward(self, inputs, out_from="full"):
        last_hidden_state = self.encoder(inputs["input_ids"], attention_mask=inputs["attention_mask"])[0]
        cls_representation = last_hidden_state[:, 0, :]  # cls representation
        out = self.linear(cls_representation)
        return out


class ReplayMemory:

    def __init__(self, write_prob, tuple_size):
        self.buffer = []
        self.write_prob = write_prob
        self.tuple_size = tuple_size

    def write(self, input_tuple):
        if random.random() < self.write_prob:
            self.buffer.append(input_tuple)

    def read(self):
        return random.choice(self.buffer)

    def write_batch(self, *elements):
        element_list = []
        for e in elements:
            if isinstance(e, torch.Tensor):
                element_list.append(e.tolist())
            else:
                element_list.append(e)
        for write_tuple in zip(*element_list):
            self.write(write_tuple)

    def read_batch(self, batch_size):
        contents = [[] for _ in range(self.tuple_size)]
        for _ in range(batch_size):
            read_tuple = self.read()
            for i in range(len(read_tuple)):
                contents[i].append(read_tuple[i])
        return tuple(contents)

    def len(self):
        return len(self.buffer)

    def reset_memory(self):
        self.buffer = []


class MemoryStore:
    """
    Memory component.
    """
    def __init__(self, memory_size, key_dim, relevance_discount=0.99, delete_method="age",
                 device="cpu", initialize=True):
        self.memory_size = memory_size
        self.key_dim = key_dim
        self.device = device
        self.relevance_discount = relevance_discount
        self.delete_method = delete_method

        if initialize:
            self.initialize()

    def query(self, query, n_neighbours):
        """Query memory given an input embedding"""
        if self.added_memories == 0:
            return None
        # TODO: possibly L2-normalize query vector?
        # allow single queries as well, not only batches
        if query.dim() == 1:
            query.unsqueeze_(0)
        idx, distances = self.get_nearest_entries_ixs(query, n_neighbours)
        return {
            "embedding": self.memory_embeddings[idx],
            "label": self.memory_labels[idx],
            "age": self.memory_age[idx],
            "relevance": self.memory_relevance[idx],
            "ix": idx
        }
    
    def get_nearest_entries_ixs(self, queries, n_neighbours):
        """
        Reference code for pairwise distance computation:
        https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3
        Args:
            queries: a torch array with (n_queries, key_dim) size
        Returns:
            embeddings: a torch array with (n_queries, max_neighbours, key_dim) size
            labels: a torch array with (n_queries, max_neighbours) size
        """
        if self.added_memories == 0:
            return None, None
        if self.added_memories < n_neighbours:
            n_neighbours = self.added_memories
        mask_idx = min(
            self.memory_size, 
            self.added_memories
        )
        # Masking not filled memories
        mask_memory = self.memory_embeddings[:mask_idx]

        # Calculating pairwise distances between queries (q) and memory (m) entries
        with torch.no_grad():
            q_norm = torch.sum(queries ** 2, dim=1).view(-1, 1)
            m_norm = torch.sum(mask_memory ** 2, dim=1).view(1, -1)
            qm = torch.mm(queries, mask_memory.transpose(1, 0))
            dist = q_norm - 2 * qm + m_norm

            # Determining indices of nearest memories and fetching corresponding labels and embeddings
            distances, idx = torch.topk(-dist, dim=1, k=n_neighbours)
            distances = -1.0 * distances
        return idx, distances

    def add_entry(self, embeddings, labels, query_result=None):
        """
        Add entries to the memory module.

        Parameters
        ---
        embeddings: a torch tensor with (batch, key_dim) size
        labels: a torch tensor with (batch) size
        query_result: dictionary
            Result of query previously done on memory.
        """
        n_added = embeddings.shape[0]

        if self.added_memories + n_added <= self.memory_size:
            start = self.write_pointer + 1
            end = start + n_added
            self.update_ixs(range(start, end), embeddings, labels, query_result)
            self.write_pointer += n_added
        else:
            capacity_remaining = max(self.memory_size - self.added_memories, 0)
            if capacity_remaining > 0:
                start = self.write_pointer + 1
                self.update_ixs(range(start, self.memory_size), embeddings[:capacity_remaining], labels[:capacity_remaining],
                                query_result, global_update=False)

            n_overflow = n_added - capacity_remaining
            # get least relevant indices
            if self.delete_method == "relevance":
                write_ixs = torch.topk(self.memory_relevance, n_overflow, largest=False).indices
            elif self.delete_method == "age":
                write_ixs = torch.topk(self.memory_age, n_overflow, largest=True).indices
            self.update_ixs(write_ixs, embeddings[capacity_remaining:], labels[capacity_remaining:], query_result)
        self.added_memories += n_added

    def update_ixs(self, ixs, embeddings, labels, query_result=None, global_update=True):
        """Update specific memory slots
        
        If query_result is not supplied, relevance scores are not updated.
        """
        self.memory_embeddings[ixs] = embeddings
        self.memory_labels[ixs] = labels
        self.memory_age[ixs] = 0
        self.memory_relevance[ixs] = 0
        if global_update:
            self.memory_age[self.memory_age != -1] += 1
            if query_result is not None:
                # relevance mask indicates which samples are deemed relevant based on the query results
                # here this is simply which queries were closest to the query embedding, as well as the query itself
                relevance_mask = torch.zeros_like(self.memory_relevance)
                ix_counts = Counter(query_result["ix"].flatten().tolist())
                for ix, count in ix_counts.items():
                    relevance_mask[ix] = count
                relevance_mask[ixs] = 1  # assign a relevance of 1 to newly added entries as well
                # keep exponential moving average
                self.memory_relevance = self.relevance_discount * self.memory_relevance + relevance_mask

    def initialize(self):
        """(Re-)Initialize memory"""
        # Initializing memory to blank embeddings and "n_classes = not seen" labels
        self.memory_embeddings = torch.zeros(
            (self.memory_size, self.key_dim)).to(self.device)
        # initialize to dummy value -1
        self.memory_labels = -1 * torch.ones((self.memory_size,), dtype=torch.long).to(self.device)
        self.memory_age = -1 * torch.ones((self.memory_size,), dtype=torch.long).to(self.device)
        self.memory_relevance = torch.zeros((self.memory_size,), dtype=torch.long).to(self.device)

        self.write_pointer = -1
        self.added_memories = 0

    def __len__(self):
        return min(self.memory_size, self.added_memories)

class ClassMemoryStore(MemoryStore):
    """
    Memory component for class representations.
    """
    def __init__(self, key_dim, relevance_discount=0.99, delete_method="age",
                 n_classes=None, discount_method="mean", class_discount=0.99, device="cpu"):
        super().__init__(n_classes, key_dim, relevance_discount, delete_method, device, initialize=False)
        self.n_classes = n_classes
        self.class_discount_method = discount_method
        self.class_discount = class_discount
        self.device = device
        self.relevance_discount = relevance_discount
        self.delete_method = delete_method
        # keeps track of which classes where updated when
        self.last_updated = defaultdict(int)
        self.last_updated_threshold = 5

        self.initialize()

    def query(self, query, n_neighbours):
        """Query memory given an input embedding"""
        if self.added_memories == 0:
            return None
        # TODO: possibly L2-normalize query vector?
        # allow single queries as well, not only batches
        if query.dim() == 1:
            query.unsqueeze_(0)
        idx, distances = self.get_nearest_entries_ixs(query, n_neighbours)
        return {
            "embedding": self.memory_embeddings[idx],
            "label": self.memory_labels[idx],
            "age": self.memory_age[idx],
            "relevance": self.memory_relevance[idx],
            "ix": idx
        }
    
    def get_nearest_entries_ixs(self, queries, n_neighbours):
        """
        Reference code for pairwise distance computation:
        https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3
        Args:
            queries: a torch array with (n_queries, key_dim) size
        Returns:
            embeddings: a torch array with (n_queries, max_neighbours, key_dim) size
            labels: a torch array with (n_queries, max_neighbours) size
        """
        if self.added_memories == 0:
            return None, None
        if self.added_memories < n_neighbours:
            n_neighbours = self.added_memories
        mask_idx = min(
            self.memory_size, 
            self.added_memories
        )
        # Masking not filled memories
        mask_memory = self.memory_embeddings[:mask_idx]

        # Calculating pairwise distances between queries (q) and memory (m) entries
        with torch.no_grad():
            q_norm = torch.sum(queries ** 2, dim=1).view(-1, 1)
            m_norm = torch.sum(mask_memory ** 2, dim=1).view(1, -1)
            qm = torch.mm(queries, mask_memory.transpose(1, 0))
            dist = q_norm - 2 * qm + m_norm

            # Determining indices of nearest memories and fetching corresponding labels and embeddings
            distances, idx = torch.topk(-dist, dim=1, k=n_neighbours)
            distances = -1.0 * distances
        return idx, distances

    def add_entry(self, embeddings, labels, query_result=None):
        """
        Add entries to the memory module.

        Parameters
        ---
        embeddings: a torch tensor with (batch, key_dim) size
        labels: a torch tensor with (batch) size
        query_result: dictionary
            Result of query previously done on memory.
        """
        n_added = embeddings.shape[0]
        class_discount = self.class_discount
        for embedding, label in zip(embeddings, labels):
            n = self.n_class_embeddings[label]
            if self.class_discount_method == "mean":
                class_discount = 1 / (n + 1)
            self.class_representations[label] = (1 - class_discount) * self.class_representations[label] \
                                                + class_discount * embedding
            self.n_class_embeddings[label] += 1

        self.added_memories += n_added

    def update(self, class_means, unique_labels, logger=None):
        """
        Update class representations with new class means.

        Parameters
        ---
        class_means: Tensor (BATCH, HIDDEN_DIM)
        unique_labels: Tensor
            unique classes in batch
        """
        n_added = class_means.shape[0]
        class_discount = self.class_discount
        to_update = unique_labels
        normalized_class_means = None
        normalized_class_representations = None
        self.update_last_updated(unique_labels)

        # selection of old class representations here
        old_class_representations = self.class_representations[to_update]
        # if old class representations haven't been given values yet, don't bias towards 0 by exponential update
        n = self.n_class_embeddings[to_update]
        if self.class_discount_method == "mean":
            class_discount = (1 / (n + 1)).unsqueeze(-1)
        elif self.class_discount_method == "adaptive":
            class_means_norm = class_means.norm(dim=1).unsqueeze(-1)
            old_class_representations_norm = old_class_representations.norm(dim=1).unsqueeze(-1)
            class_means_norm[class_means_norm == 0] = 1 # avoid nans
            old_class_representations_norm[old_class_representations_norm == 0] = 1 # avoid nans
            normalized_class_means = class_means / class_means_norm
            normalized_class_representations = old_class_representations / old_class_representations_norm 
            
            # class_dists = euclidean_dist(class_means, old_class_representations).diag()
            normalized_class_dists = euclidean_dist(normalized_class_means, normalized_class_representations).diag()

            b = 1.3
            class_discount = (2 / (2 ** b) * normalized_class_dists ** b).unsqueeze(-1)
            class_discount = torch.clamp_max(class_discount, 1)
            if logger is not None:
                logger.debug(f"Normalized Dists: {normalized_class_dists}")
        # memory update rule here
        # classes of which prototypes haven't been updated yet
        uninitialized_vector_classes = (old_class_representations == 0).bool().all(dim=1)
        if uninitialized_vector_classes.all():
            class_discount = 1
        elif not isinstance(class_discount, float):
            # otherwise it is a tensor
            class_discount[uninitialized_vector_classes] = 1
        new_class_representations = (1 - class_discount) * old_class_representations + class_discount * class_means
        self.n_class_embeddings[to_update] += 1
        if logger is not None:
            logger.debug(f"Labels: {unique_labels}")
            logger.debug(f"Uninitialized Classes: {unique_labels[uninitialized_vector_classes]}")
            logger.debug(f"Class discount: {class_discount}")
            logger.debug(f"Updating class representations for classes {unique_labels}.\n"
                            f"Distance old class representations and class means: {[round(z, 2) for z in (old_class_representations - class_means).norm(dim=1).tolist()]}\n"
                            f"Distance old and new class representations: {[round(z, 2) for z in (new_class_representations - old_class_representations).norm(dim=1).tolist()]}"
                            )
        # for returning new class representations while keeping gradients intact
        result = torch.clone(self.class_representations)
        result[to_update] = new_class_representations
        # update memory
        self.class_representations[to_update] = new_class_representations.detach()

        self.added_memories += n_added
        return {"new_class_representations": result, "class_discount": class_discount}

    def update_last_updated(self, labels):
        """
        Parameters
        ---
        labels: iterable of (unique) class labels
        """
        for label in list(self.last_updated.keys()):
            self.last_updated[label] += 1
            if self.last_updated[label] >= self.last_updated_threshold:
                del self.last_updated[label]
        for label in labels:
            self.last_updated[label] = 0

    def update_ixs(self, ixs, embeddings, labels, query_result=None, global_update=True):
        """Update specific memory slots
        
        If query_result is not supplied, relevance scores are not updated.
        """
        self.memory_embeddings[ixs] = embeddings
        self.memory_labels[ixs] = labels
        self.memory_age[ixs] = 0
        self.memory_relevance[ixs] = 0
        if global_update:
            self.memory_age[self.memory_age != -1] += 1
            if query_result is not None:
                # relevance mask indicates which samples are deemed relevant based on the query results
                # here this is simply which queries were closest to the query embedding, as well as the query itself
                relevance_mask = torch.zeros_like(self.memory_relevance)
                ix_counts = Counter(query_result["ix"].flatten().tolist())
                for ix, count in ix_counts.items():
                    relevance_mask[ix] = count
                relevance_mask[ixs] = 1  # assign a relevance of 1 to newly added entries as well
                # keep exponential moving average
                self.memory_relevance = self.relevance_discount * self.memory_relevance + relevance_mask

    def initialize(self):
        """(Re-)Initialize memory"""
        # Initializing memory to blank embeddings and "n_classes = not seen" labels
        self.memory_embeddings = torch.zeros(
            (self.memory_size, self.key_dim)).to(self.device)
        # initialize to dummy value -1
        self.memory_labels = -1 * torch.ones((self.memory_size,), dtype=torch.long).to(self.device)
        self.memory_age = -1 * torch.ones((self.memory_size,), dtype=torch.long).to(self.device)
        self.memory_relevance = torch.zeros((self.memory_size,), dtype=torch.long).to(self.device)
        if self.n_classes is not None:
            self.class_representations = torch.zeros((self.n_classes, self.key_dim)).to(self.device)
            self.n_class_embeddings = torch.zeros(self.n_classes).to(self.device)

        self.write_pointer = -1
        self.added_memories = 0

    def __len__(self):
        return min(self.memory_size, self.added_memories)
    
class LSTMDecoder(nn.Module):
    """LSTM decoder
    
    Concatenates neighbour embeddings and labels. Initial state is the embedding of interest.
    Returns representation after decoding which can be used for classification.
    """
    def __init__(self, key_size, embedding_size):
        """
        Parameters
        ---
        key_size: size of memory keys after going through the key decoder
        embedding_size: size of query embedding
        """
        super().__init__()
        # + 1 for label
        self.lstm = nn.LSTM(input_size=key_size + 1, hidden_size=embedding_size, batch_first=True)
    
    def forward(self, embedding, query_result):
        """
        Parameters
        ---
        embedding: tensor shape (BATCH_SIZE, EMBEDDING_DIM)
            embedding of input we want to make a prediction for
        query_result: dictionary
            containing at least keys "embedding" and "label" of sizes
            (BATCH_SIZE, N_NEIGHBOURS, EMBEDDING_DIM) and BATCH_SIZE respectively.
        """
        # concatenate neighbour embeddings and labels
        if query_result is None:
            return embedding
        # shape (batch, seq_len, input_size)
        neighbours = torch.cat((query_result["embedding"], query_result["label"].unsqueeze(-1)), dim=-1)
        h_0 = embedding.unsqueeze(0).contiguous() # for LSTM API
        c_0 = torch.zeros_like(h_0)
        
        _, (h_n, _) = self.lstm(neighbours, (h_0, c_0))
        return h_n.squeeze(0)

class SimpleDecoder(nn.Module):
    def __init__(self, key_size, embedding_size):
        super().__init__()
        self.decoder = nn.Linear(key_size + 1 + embedding_size, embedding_size)

    def forward(self, embedding, query_result):
        if query_result is None:
            return embedding
        # mean over neighbours
        # todo: change to mean over embedding and neighbours instead of adding them
        neighbours = torch.cat((query_result["embedding"], query_result["label"].unsqueeze(-1)), dim=-1)
        neighbours_mean = neighbours.mean(dim=1)
        concatenation = torch.cat((neighbours_mean, embedding), dim=-1)
        concatenation = torch.cat((torch.zeros_like(neighbours_mean), embedding), dim=-1)
        return self.decoder(concatenation)