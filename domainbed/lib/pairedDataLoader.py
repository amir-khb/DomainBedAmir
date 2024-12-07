import torch
from torch.utils.data import Sampler
import numpy as np


class DomainPairSampler:
    def __init__(self, datasets, domains, batch_size, test_domains):
        self.datasets = datasets
        self.batch_size = batch_size
        assert batch_size % 2 == 0, "Batch size must be even for pairs"

        print("Number of domains:", len(self.datasets))
        print("Batch size:", batch_size)
        print("Test domains:", test_domains)

        # Get indices for each class and domain (excluding test domains)
        self.class_domain_indices = {}

        # For each domain
        for domain_idx, domain_dataset in enumerate(self.datasets):
            if domain_idx in test_domains:
                continue

            print(f"Processing domain {domain_idx} with {len(domain_dataset)} samples")

            # Get all samples from this domain
            for idx in range(len(domain_dataset)):
                item = domain_dataset[idx]
                label = item[1]  # Label is the second element

                if label not in self.class_domain_indices:
                    self.class_domain_indices[label] = {}
                if domain_idx not in self.class_domain_indices[label]:
                    self.class_domain_indices[label][domain_idx] = []

                self.class_domain_indices[label][domain_idx].append(idx)

        # Convert lists to numpy arrays for faster sampling
        for label in self.class_domain_indices:
            for domain in self.class_domain_indices[label]:
                self.class_domain_indices[label][domain] = np.array(self.class_domain_indices[label][domain])

        self.labels = list(self.class_domain_indices.keys())
        self.train_domains = [d for d in domains if d not in test_domains]

        print("Number of classes found:", len(self.labels))
        print("Training domains:", self.train_domains)

        # Print some statistics
        for domain in self.train_domains:
            labels_in_domain = [label for label in self.labels
                                if domain in self.class_domain_indices[label]]
            print(f"Domain {domain} has {len(labels_in_domain)} classes")

    def sample_batch(self):
        batch = []
        while len(batch) < self.batch_size:
            # Select a random class that exists in at least 2 training domains
            valid_labels = [l for l in self.labels
                            if len([d for d in self.train_domains
                                    if d in self.class_domain_indices[l]]) >= 2]

            if not valid_labels:
                continue

            label = np.random.choice(valid_labels)

            # Get domains that have this class
            valid_domains = [d for d in self.train_domains
                             if d in self.class_domain_indices[label]]

            if len(valid_domains) < 2:
                continue

            # Select two different domains
            domain1, domain2 = np.random.choice(valid_domains, size=2, replace=False)

            # Get indices for this class from each domain
            idx1 = np.random.choice(self.class_domain_indices[label][domain1])
            idx2 = np.random.choice(self.class_domain_indices[label][domain2])

            batch.extend([(domain1, idx1), (domain2, idx2)])

        return batch


class PairedDataLoader:
    def __init__(self, dataset, domains, batch_size, num_workers, test_domains):
        self.dataset = dataset
        self.domains = domains
        self.batch_size = batch_size
        self.sampler = DomainPairSampler(dataset, domains, batch_size, test_domains)

    def __iter__(self):
        return self

    def __next__(self):
        # Generate new batch
        batch_indices = self.sampler.sample_batch()

        images = []
        labels = []
        domain_labels = []

        for domain_idx, idx in batch_indices:
            item = self.dataset[domain_idx][idx]
            images.append(item[0])
            labels.append(item[1])

            # Create 4D domain label (for DANN compatibility)
            domain_label = torch.zeros(4)  # Fixed 4D for DANN
            domain_label[domain_idx] = 1.0
            domain_labels.append(domain_label)

        # Stack all tensors
        images = torch.stack(images)
        labels = torch.tensor(labels)
        domain_labels = torch.stack(domain_labels)

        # Print first batch info for debugging
        if not hasattr(self, 'printed_debug'):
            print("First batch debug info:")
            print(f"Batch size: {len(images)}")
            print(f"Domain labels shape: {domain_labels.shape}")
            print(f"Sample domain labels:\n{domain_labels[:4]}")
            self.printed_debug = True

        return (images, labels, domain_labels)