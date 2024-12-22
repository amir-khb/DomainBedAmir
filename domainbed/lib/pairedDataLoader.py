import torch
from torch.utils.data import Sampler
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import torch.nn.functional as F
import random


class DomainPairSampler:
    """
    Samples pairs of images from same class but different domains, applies mixup,
    and returns original pairs along with mixed versions.
    """

    def __init__(self, datasets, domains, batch_size, test_domains, seed=42):
        """
        Initialize the sampler with datasets and parameters.

        Args:
            datasets: List of domain datasets
            domains: List of domain indices
            batch_size: Number of samples per batch (must be divisible by 3)
            test_domains: List of domain indices to exclude from training
            seed: Random seed for reproducibility
        """
        # Set all random seeds
        self.seed = seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        random.seed(self.seed)

        self.datasets = datasets
        self.batch_size = batch_size

        assert batch_size % 3 == 0, "Batch size must be divisible by 3 to accommodate originals and mixed samples"

        print("\nInitializing DomainPairSampler:")
        print(f"Number of domains: {len(self.datasets)}")
        print(f"Batch size: {batch_size}")
        print(f"Test domains: {test_domains}")
        print(f"Random seed: {seed}")

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
                self.class_domain_indices[label][domain] = np.array(
                    self.class_domain_indices[label][domain]
                )

        self.labels = list(self.class_domain_indices.keys())
        self.train_domains = [d for d in domains if d not in test_domains]

        print(f"Number of classes found: {len(self.labels)}")
        print(f"Training domains: {self.train_domains}")

        # Print class distribution across domains
        for domain in self.train_domains:
            labels_in_domain = [
                label for label in self.labels
                if domain in self.class_domain_indices[label]
            ]
            print(f"Domain {domain} has {len(labels_in_domain)} classes")

    def sample_random_lambda(self):
        """Sample lambda from uniform distribution between 0.2 and 0.8."""
        np.random.seed(self.seed)
        return np.random.uniform(0.2, 0.8)

    def sample_batch(self):
        """
        Sample a batch of pairs and generate mixup parameters.

        Returns:
            batch_indices: List of (domain1, idx1, domain2, idx2) tuples
            mixup_lambdas: List of mixup interpolation weights
        """
        # Reset all random seeds before batch sampling
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        random.seed(self.seed)

        batch = []
        mixup_lambdas = []

        # We need batch_size/3 pairs (each pair generates 3 samples)
        pairs_needed = self.batch_size // 3

        while len(batch) < pairs_needed:
            # Select a random class that exists in at least 2 training domains
            valid_labels = [
                l for l in self.labels
                if len([d for d in self.train_domains
                        if d in self.class_domain_indices[l]]) >= 2
            ]

            if not valid_labels:
                continue

            label = np.random.choice(valid_labels)

            # Get domains that have this class
            valid_domains = [
                d for d in self.train_domains
                if d in self.class_domain_indices[label]
            ]

            if len(valid_domains) < 2:
                continue

            # Select two different domains
            domain1, domain2 = np.random.choice(valid_domains, size=2, replace=False)

            # Get indices for this class from each domain
            idx1 = np.random.choice(self.class_domain_indices[label][domain1])
            idx2 = np.random.choice(self.class_domain_indices[label][domain2])

            # Generate random lambda between 0.2 and 0.8
            lam = self.sample_random_lambda()

            mixup_lambdas.append(lam)
            batch.append((domain1, idx1, domain2, idx2))

            # Increment seed for next iteration to ensure different samples
            self.seed += 1

        return batch, mixup_lambdas

def visualize_mixup(image1, image2, mixed_image, lambda_val, save_path=None):
    """
    Visualize original images and their mixup.

    Args:
        image1: First image tensor [C,H,W]
        image2: Second image tensor [C,H,W]
        mixed_image: Mixed image tensor [C,H,W]
        lambda_val: Mixup coefficient
        save_path: Optional path to save the visualization
    """
    # Denormalize images (assuming ImageNet normalization)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def denorm(x):
        return torch.clamp(x * std + mean, 0, 1)

    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Convert tensors to images
    img1 = denorm(image1).permute(1, 2, 0).cpu().numpy()
    img2 = denorm(image2).permute(1, 2, 0).cpu().numpy()
    mixed = denorm(mixed_image).permute(1, 2, 0).cpu().numpy()

    # Plot images
    ax1.imshow(img1)
    ax1.set_title('Domain A')
    ax1.axis('off')

    ax2.imshow(img2)
    ax2.set_title('Domain B')
    ax2.axis('off')

    ax3.imshow(mixed)
    ax3.set_title(f'Mixed (λ={lambda_val:.2f})')
    ax3.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


class PairedDataLoader:
    def __init__(self, dataset, domains, batch_size, num_workers, test_domains,
                 visualize_first_n=5, seed=42):
        """
        Initialize the data loader with reproducible sampling.

        Args:
            dataset: List of domain datasets
            domains: List of domain indices
            batch_size: Number of samples per batch
            num_workers: Number of worker processes
            test_domains: List of domain indices to exclude from training
            visualize_first_n: Number of initial mixups to visualize
            seed: Random seed for reproducibility
        """
        # Set all random seeds
        self.seed = seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        random.seed(self.seed)

        self.dataset = dataset
        self.domains = domains
        self.batch_size = batch_size
        self.sampler = DomainPairSampler(dataset, domains, batch_size, test_domains,
                                        seed=seed)
        self.batch_count = 0
        self.visualize_first_n = visualize_first_n
        self.visualized_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        # Reset random seeds before each batch
        np.random.seed(self.seed + self.batch_count)
        torch.manual_seed(self.seed + self.batch_count)
        torch.cuda.manual_seed(self.seed + self.batch_count)
        random.seed(self.seed + self.batch_count)

        batch_indices, mixup_lambdas = self.sampler.sample_batch()

        all_images = []
        all_labels = []
        all_domain_labels = []

        for idx, ((domain1_idx, idx1, domain2_idx, idx2), lam) in enumerate(
                zip(batch_indices, mixup_lambdas)):

            # Get items
            item1 = self.dataset[domain1_idx][idx1]
            item2 = self.dataset[domain2_idx][idx2]

            # Create domain labels
            domain1_label = torch.zeros(4)
            domain2_label = torch.zeros(4)
            domain1_label[domain1_idx] = 1.0
            domain2_label[domain2_idx] = 1.0

            # Add originals
            all_images.extend([item1[0], item2[0]])
            all_labels.extend([item1[1], item2[1]])
            all_domain_labels.extend([domain1_label, domain2_label])

            # Create mixed image
            mixed_image = lam * item1[0] + (1 - lam) * item2[0]
            mixed_domain_label = lam * domain1_label + (1 - lam) * domain2_label

            # Add mixed
            all_images.append(mixed_image)
            all_labels.append(item1[1])
            all_domain_labels.append(mixed_domain_label)

            # Visualize first N pairs
            if self.visualized_count < self.visualize_first_n:
                visualize_mixup(
                    item1[0], item2[0], mixed_image, lam,
                    save_path=f'mixup_visualization_{self.visualized_count}.png'
                )
                print(f"\nMixup {self.visualized_count + 1}:")
                print(f"Domain A: {domain1_label.tolist()}")
                print(f"Domain B: {domain2_label.tolist()}")
                print(f"Mixed domain: {mixed_domain_label.tolist()}")
                print(f"Lambda: {lam:.3f}")
                self.visualized_count += 1

        self.batch_count += 1

        return (torch.stack(all_images),
                torch.tensor(all_labels),
                torch.stack(all_domain_labels))


# class DomainPairSampler:
#     """
#     Samples pairs of images from same class but different domains.
#     """
#
#     def __init__(self, datasets, domains, batch_size, test_domains, seed=42):
#         """
#         Initialize the sampler with datasets and parameters.
#
#         Args:
#             datasets: List of domain datasets
#             domains: List of domain indices
#             batch_size: Number of samples per batch (must be even)
#             test_domains: List of domain indices to exclude from training
#             seed: Random seed for reproducibility
#         """
#         # Set all random seeds
#         self.seed = seed
#         np.random.seed(self.seed)
#         torch.manual_seed(self.seed)
#         torch.cuda.manual_seed(self.seed)
#         random.seed(self.seed)
#
#         self.datasets = datasets
#         self.batch_size = batch_size
#
#         assert batch_size % 2 == 0, "Batch size must be even to accommodate pairs"
#
#         print("\nInitializing DomainPairSampler:")
#         print(f"Number of domains: {len(self.datasets)}")
#         print(f"Batch size: {batch_size}")
#         print(f"Test domains: {test_domains}")
#         print(f"Random seed: {seed}")
#
#         # Get indices for each class and domain (excluding test domains)
#         self.class_domain_indices = {}
#
#         # For each domain
#         for domain_idx, domain_dataset in enumerate(self.datasets):
#             if domain_idx in test_domains:
#                 continue
#
#             print(f"Processing domain {domain_idx} with {len(domain_dataset)} samples")
#
#             # Get all samples from this domain
#             for idx in range(len(domain_dataset)):
#                 item = domain_dataset[idx]
#                 label = item[1]  # Label is the second element
#
#                 if label not in self.class_domain_indices:
#                     self.class_domain_indices[label] = {}
#                 if domain_idx not in self.class_domain_indices[label]:
#                     self.class_domain_indices[label][domain_idx] = []
#
#                 self.class_domain_indices[label][domain_idx].append(idx)
#
#         # Convert lists to numpy arrays for faster sampling
#         for label in self.class_domain_indices:
#             for domain in self.class_domain_indices[label]:
#                 self.class_domain_indices[label][domain] = np.array(
#                     self.class_domain_indices[label][domain]
#                 )
#
#         self.labels = list(self.class_domain_indices.keys())
#         self.train_domains = [d for d in domains if d not in test_domains]
#
#         print(f"Number of classes found: {len(self.labels)}")
#         print(f"Training domains: {self.train_domains}")
#
#         # Print class distribution across domains
#         for domain in self.train_domains:
#             labels_in_domain = [
#                 label for label in self.labels
#                 if domain in self.class_domain_indices[label]
#             ]
#             print(f"Domain {domain} has {len(labels_in_domain)} classes")
#
#     def sample_batch(self):
#         """
#         Sample a batch of pairs.
#
#         Returns:
#             batch_indices: List of (domain1, idx1, domain2, idx2) tuples
#         """
#         # Reset all random seeds before batch sampling
#         np.random.seed(self.seed)
#         torch.manual_seed(self.seed)
#         torch.cuda.manual_seed(self.seed)
#         random.seed(self.seed)
#
#         batch = []
#         pairs_needed = self.batch_size // 2
#
#         while len(batch) < pairs_needed:
#             # Select a random class that exists in at least 2 training domains
#             valid_labels = [
#                 l for l in self.labels
#                 if len([d for d in self.train_domains
#                         if d in self.class_domain_indices[l]]) >= 2
#             ]
#
#             if not valid_labels:
#                 continue
#
#             label = np.random.choice(valid_labels)
#
#             # Get domains that have this class
#             valid_domains = [
#                 d for d in self.train_domains
#                 if d in self.class_domain_indices[label]
#             ]
#
#             if len(valid_domains) < 2:
#                 continue
#
#             # Select two different domains
#             domain1, domain2 = np.random.choice(valid_domains, size=2, replace=False)
#
#             # Get indices for this class from each domain
#             idx1 = np.random.choice(self.class_domain_indices[label][domain1])
#             idx2 = np.random.choice(self.class_domain_indices[label][domain2])
#
#             batch.append((domain1, idx1, domain2, idx2))
#
#             # Increment seed for next iteration to ensure different samples
#             self.seed += 1
#
#         return batch
#
#
# def visualize_pair(image1, image2, save_path=None):
#     """
#     Visualize paired images.
#
#     Args:
#         image1: First image tensor [C,H,W]
#         image2: Second image tensor [C,H,W]
#         save_path: Optional path to save the visualization
#     """
#     # Denormalize images (assuming ImageNet normalization)
#     mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
#     std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
#
#     def denorm(x):
#         return torch.clamp(x * std + mean, 0, 1)
#
#     # Create figure
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
#
#     # Convert tensors to images
#     img1 = denorm(image1).permute(1, 2, 0).cpu().numpy()
#     img2 = denorm(image2).permute(1, 2, 0).cpu().numpy()
#
#     # Plot images
#     ax1.imshow(img1)
#     ax1.set_title('Domain A')
#     ax1.axis('off')
#
#     ax2.imshow(img2)
#     ax2.set_title('Domain B')
#     ax2.axis('off')
#
#     plt.tight_layout()
#
#     if save_path:
#         plt.savefig(save_path)
#         plt.close()
#     else:
#         plt.show()
#
#
# class PairedDataLoader:
#     def __init__(self, dataset, domains, batch_size, num_workers, test_domains,
#                  visualize_first_n=5, seed=42):
#         """
#         Initialize the data loader with reproducible sampling.
#
#         Args:
#             dataset: List of domain datasets
#             domains: List of domain indices
#             batch_size: Number of samples per batch (must be even)
#             num_workers: Number of worker processes
#             test_domains: List of domain indices to exclude from training
#             visualize_first_n: Number of initial pairs to visualize
#             seed: Random seed for reproducibility
#         """
#         # Set all random seeds
#         self.seed = seed
#         np.random.seed(self.seed)
#         torch.manual_seed(self.seed)
#         torch.cuda.manual_seed(self.seed)
#         random.seed(self.seed)
#
#         self.dataset = dataset
#         self.domains = domains
#         self.batch_size = batch_size
#         self.sampler = DomainPairSampler(dataset, domains, batch_size, test_domains, seed=seed)
#         self.batch_count = 0
#         self.visualize_first_n = visualize_first_n
#         self.visualized_count = 0
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         # Reset random seeds before each batch
#         np.random.seed(self.seed + self.batch_count)
#         torch.manual_seed(self.seed + self.batch_count)
#         torch.cuda.manual_seed(self.seed + self.batch_count)
#         random.seed(self.seed + self.batch_count)
#
#         batch_indices = self.sampler.sample_batch()
#
#         all_images = []
#         all_labels = []
#         all_domain_labels = []
#
#         for idx, (domain1_idx, idx1, domain2_idx, idx2) in enumerate(batch_indices):
#             # Get items
#             item1 = self.dataset[domain1_idx][idx1]
#             item2 = self.dataset[domain2_idx][idx2]
#
#             # Create domain labels
#             domain1_label = torch.zeros(4)
#             domain2_label = torch.zeros(4)
#             domain1_label[domain1_idx] = 1.0
#             domain2_label[domain2_idx] = 1.0
#
#             # Add pair to batch
#             all_images.extend([item1[0], item2[0]])
#             all_labels.extend([item1[1], item2[1]])
#             all_domain_labels.extend([domain1_label, domain2_label])
#
#             # Visualize first N pairs
#             if self.visualized_count < self.visualize_first_n:
#                 visualize_pair(
#                     item1[0], item2[0],
#                     save_path=f'pair_visualization_{self.visualized_count}.png'
#                 )
#                 print(f"\nPair {self.visualized_count + 1}:")
#                 print(f"Domain A: {domain1_label.tolist()}")
#                 print(f"Domain B: {domain2_label.tolist()}")
#                 self.visualized_count += 1
#
#         self.batch_count += 1
#
#         return (torch.stack(all_images),
#                 torch.tensor(all_labels),
#                 torch.stack(all_domain_labels))