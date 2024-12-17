# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import re
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset, ConcatDataset, Dataset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate

from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.fmow_dataset import FMoWDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Small images
    "ColoredMNIST",
    "RotatedMNIST",
    # Big images
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    "SVIRO",
    # WILDS datasets
    "WILDSCamelyon",
    "WILDSFMoW",
    # Spawrious datasets
    "SpawriousO2O_easy",
    "SpawriousO2O_medium",
    "SpawriousO2O_hard",
    "SpawriousM2M_easy",
    "SpawriousM2M_medium",
    "SpawriousM2M_hard",
]


def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001  # Default, subclasses may override
    CHECKPOINT_FREQ = 100  # Default, subclasses may override
    N_WORKERS = 8  # Default, subclasses may override
    ENVIRONMENTS = None  # Subclasses should override
    INPUT_SHAPE = None  # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )


class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']


class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']


class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes


class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']

    def __init__(self, root, test_envs, hparams):
        super(ColoredMNIST, self).__init__(root, [0.1, 0.2, 0.9],
                                           self.color_dataset, (2, 28, 28,), 2)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
                                                         1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']

    def __init__(self, root, test_envs, hparams):
        super(RotatedMNIST, self).__init__(root, [0, 15, 30, 45, 60, 75],
                                           self.rotate_dataset, (1, 28, 28,), 10)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                                               interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)


class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                                      transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)


class SoftLabelImageFolder(ImageFolder):
    def __init__(self, root, transform=None, domain_weights=None):
        super().__init__(root=root, transform=transform)
        self.domain_weights = domain_weights

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        return img, label, self.domain_weights


class MultipleEnvironmentImageFolderSoft(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        print("\nDirectory structure:")
        print(f"Root directory: {root}")
        print("Folders found:", environments)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []

        # Process each environment/folder
        for i, environment in enumerate(environments):
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)

            # Determine domain weights
            if any(x in environment for x in ['0.', '1.']):  # Mixed domain folder
                domain_weights = self.parse_domain_weights(environment)
            else:  # Pure domain folder
                domain_weights = self.create_pure_domain_weights(environment)

            print(f"\nEnvironment: {environment}")
            print(f"Domain weights: {domain_weights}")

            env_dataset = SoftLabelImageFolder(
                path,
                transform=env_transform,
                domain_weights=domain_weights
            )

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)

    ENVIRONMENTS = ['A', 'C', 'P', 'S']  # [Art, Cartoon, Photo, Sketch]

    def parse_domain_weights(self, folder_name):
        """Parse domain weights from folder name (e.g., '0.17art_painting0.83cartoon')"""
        weights = [0.0] * len(self.ENVIRONMENTS)
        parts = re.findall(r'(\d+\.\d+)(art_painting|cartoon|photo|sketch)', folder_name)

        for weight_str, domain in parts:
            weight = float(weight_str)
            if 'art_painting' in domain:
                weights[0] = weight
            elif 'cartoon' in domain:
                weights[1] = weight
            elif 'photo' in domain:
                weights[2] = weight
            elif 'sketch' in domain:
                weights[3] = weight

        return torch.tensor(weights)

    def create_pure_domain_weights(self, folder_name):
        """Create one-hot domain weights for pure domain folders"""
        weights = [0.0] * len(self.ENVIRONMENTS)
        # Convert folder name to match ENVIRONMENTS format
        folder_name = folder_name.replace('photo', 'P').replace('cartoon', 'C').replace('art', 'A').replace('sketch',
                                                                                                            'S')
        domain_name = folder_name[0].upper()  # Get first letter and uppercase

        for i, env in enumerate(self.ENVIRONMENTS):
            if env == domain_name:
                weights[i] = 1.0
                break

        return torch.tensor(weights)


class VLCS(MultipleEnvironmentImageFolderSoft):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["C", "L", "S", "V"]  # Caltech, LabelMe, Sun, VOC

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

    def parse_domain_weights(self, folder_name):
        """Parse domain weights from folder name (e.g., '0.17Caltech0.83LabelMe')"""
        weights = [0.0] * len(self.ENVIRONMENTS)
        parts = re.findall(r'(\d+\.\d+)(Caltech|LabelMe|Sun|VOC)', folder_name)

        for weight_str, domain in parts:
            weight = float(weight_str)
            if domain == 'Caltech':
                weights[0] = weight
            elif domain == 'LabelMe':
                weights[1] = weight
            elif domain == 'Sun':
                weights[2] = weight
            elif domain == 'VOC':
                weights[3] = weight

        return torch.tensor(weights)

    def create_pure_domain_weights(self, folder_name):
        """Create one-hot domain weights for pure domain folders"""
        weights = [0.0] * len(self.ENVIRONMENTS)
        # Map full domain names to single letter codes
        domain_mapping = {
            'Caltech': 'C',
            'LabelMe': 'L',
            'Sun': 'S',
            'VOC': 'V'
        }

        # Get the single letter code for the domain
        for full_name, code in domain_mapping.items():
            if full_name in folder_name:
                domain_name = code
                break
        else:
            domain_name = folder_name[0].upper()  # Fallback to first letter if not found

        for i, env in enumerate(self.ENVIRONMENTS):
            if env == domain_name:
                weights[i] = 1.0
                break

        return torch.tensor(weights)


class OfficeHome(MultipleEnvironmentImageFolderSoft):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "R"]  # Art, Clipart, Product, Real World

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

    def parse_domain_weights(self, folder_name):
        """Parse domain weights from folder name (e.g., '0.17Art0.83Product')"""
        weights = [0.0] * len(self.ENVIRONMENTS)
        parts = re.findall(r'(\d+\.\d+)(Art|Clipart|Product|Real_World)', folder_name)

        for weight_str, domain in parts:
            weight = float(weight_str)
            if domain == 'Art':
                weights[0] = weight
            elif domain == 'Clipart':
                weights[1] = weight
            elif domain == 'Product':
                weights[2] = weight
            elif domain == 'Real_World':
                weights[3] = weight

        return torch.tensor(weights)

    def create_pure_domain_weights(self, folder_name):
        """Create one-hot domain weights for pure domain folders"""
        weights = [0.0] * len(self.ENVIRONMENTS)
        if folder_name == 'Art':
            weights[0] = 1.0
        elif folder_name == 'Clipart':
            weights[1] = 1.0
        elif folder_name == 'Product':
            weights[2] = 1.0
        elif folder_name == 'Real_World':
            weights[3] = 1.0
        return torch.tensor(weights)


class TerraIncognita(MultipleEnvironmentImageFolderSoft):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "terra_incognita/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

    def create_pure_domain_weights(self, folder_name):
        """Create one-hot domain weights for pure domain folders"""
        weights = [0.0] * len(self.ENVIRONMENTS)

        # Map location names to indices
        domain_mapping = {
            'L100': 0,
            'L38': 1,
            'L43': 2,
            'L46': 3
        }

        # Find which domain this folder represents
        for domain_name, index in domain_mapping.items():
            if domain_name in folder_name:
                weights[index] = 1.0
                break

        return torch.tensor(weights)


class SVIRO(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["aclass", "escape", "hilux", "i3", "lexus", "tesla", "tiguan", "tucson", "x5", "zoe"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "sviro/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class WILDSEnvironment:
    def __init__(
            self,
            wilds_dataset,
            metadata_name,
            metadata_value,
            transform=None):
        self.name = metadata_name + "_" + str(metadata_value)

        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_array = wilds_dataset.metadata_array
        subset_indices = torch.where(
            metadata_array[:, metadata_index] == metadata_value)[0]

        self.dataset = wilds_dataset
        self.indices = subset_indices
        self.transform = transform

    def __getitem__(self, i):
        x = self.dataset.get_input(self.indices[i])
        if type(x).__name__ != "Image":
            x = Image.fromarray(x)

        y = self.dataset.y_array[self.indices[i]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)


class WILDSDataset(MultipleDomainDataset):
    INPUT_SHAPE = (3, 224, 224)

    def metadata_values(self, wilds_dataset, metadata_name):
        # Print metadata fields
        print("\nFMoW Dataset Metadata Information:")
        print("-----------------------------")
        print(f"Available metadata fields: {wilds_dataset.metadata_fields}")
        print(f"Looking for field: {metadata_name}")

        # Get the index and values
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        print(f"Found {metadata_name} at index: {metadata_index}")

        # Get the full region column
        metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
        print(f"\nFull {metadata_name} field values:")
        print(metadata_vals)
        print(f"\nShape of {metadata_name} field: {metadata_vals.shape}")

        # Count occurrences of each region
        unique_vals, counts = torch.unique(metadata_vals, return_counts=True)
        print(f"\nValue distribution in {metadata_name} field:")
        for val, count in zip(unique_vals.tolist(), counts.tolist()):
            print(f"Region {val}: {count} samples")

        # Get unique values for return
        unique_vals = sorted(list(set(metadata_vals.view(-1).tolist())))
        print(f"\nUnique values: {unique_vals}")
        print("-----------------------------\n")

        return unique_vals

    def __init__(self, dataset, metadata_name, test_envs, augment, hparams):
        super().__init__()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []

        # Get and print unique metadata values
        unique_values = self.metadata_values(dataset, metadata_name)
        print("Creating environments for each value:")
        for i, metadata_value in enumerate(unique_values):
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            env_dataset = WILDSEnvironment(
                dataset, metadata_name, metadata_value, env_transform)

            print(f"Created environment: {env_dataset.name}")
            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = dataset.n_classes

    # def metadata_values(self, wilds_dataset, metadata_name):
    #     metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
    #     metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
    #     return sorted(list(set(metadata_vals.view(-1).tolist())))


class WILDSCamelyon(WILDSDataset):
    ENVIRONMENTS = ["hospital_0", "hospital_1", "hospital_2", "hospital_3",
                    "hospital_4"]

    def __init__(self, root, test_envs, hparams):
        dataset = Camelyon17Dataset(root_dir=root)
        super().__init__(
            dataset, "hospital", test_envs, hparams['data_augmentation'], hparams)


class WILDSFMoW(WILDSDataset):
    ENVIRONMENTS = ["region_0", "region_1", "region_2", "region_3", "region_4", "region_5"]

    def __init__(self, root, test_envs, hparams):
        dataset = FMoWDataset(root_dir=root)
        print("\nInitializing WILDSFMoW dataset...")
        super().__init__(
            dataset, "region", test_envs, hparams['data_augmentation'], hparams)


## Spawrious base classes
class CustomImageFolder(Dataset):
    """
    A class that takes one folder at a time and loads a set number of images in a folder and assigns them a specific class
    """

    def __init__(self, folder_path, class_index, limit=None, transform=None):
        self.folder_path = folder_path
        self.class_index = class_index
        self.image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if
                            img.endswith(('.png', '.jpg', '.jpeg'))]
        if limit:
            self.image_paths = self.image_paths[:limit]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.class_index, dtype=torch.long)
        return img, label


class SpawriousBenchmark(MultipleDomainDataset):
    ENVIRONMENTS = ["Test", "SC_group_1", "SC_group_2"]
    input_shape = (3, 224, 224)
    num_classes = 4
    class_list = ["bulldog", "corgi", "dachshund", "labrador"]

    def __init__(self, train_combinations, test_combinations, root_dir, augment=True, type1=False):
        self.type1 = type1
        train_datasets, test_datasets = self._prepare_data_lists(train_combinations, test_combinations, root_dir,
                                                                 augment)
        self.datasets = [ConcatDataset(test_datasets)] + train_datasets

    # Prepares the train and test data lists by applying the necessary transformations.
    def _prepare_data_lists(self, train_combinations, test_combinations, root_dir, augment):
        test_transforms = transforms.Compose([
            transforms.Resize((self.input_shape[1], self.input_shape[2])),
            transforms.transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if augment:
            train_transforms = transforms.Compose([
                transforms.Resize((self.input_shape[1], self.input_shape[2])),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            train_transforms = test_transforms

        train_data_list = self._create_data_list(train_combinations, root_dir, train_transforms)
        test_data_list = self._create_data_list(test_combinations, root_dir, test_transforms)

        return train_data_list, test_data_list

    # Creates a list of datasets based on the given combinations and transformations.
    def _create_data_list(self, combinations, root_dir, transforms):
        data_list = []
        if isinstance(combinations, dict):

            # Build class groups for a given set of combinations, root directory, and transformations.
            for_each_class_group = []
            cg_index = 0
            for classes, comb_list in combinations.items():
                for_each_class_group.append([])
                for ind, location_limit in enumerate(comb_list):
                    if isinstance(location_limit, tuple):
                        location, limit = location_limit
                    else:
                        location, limit = location_limit, None
                    cg_data_list = []
                    for cls in classes:
                        path = os.path.join(root_dir, f"{0 if not self.type1 else ind}/{location}/{cls}")
                        data = CustomImageFolder(folder_path=path, class_index=self.class_list.index(cls), limit=limit,
                                                 transform=transforms)
                        cg_data_list.append(data)

                    for_each_class_group[cg_index].append(ConcatDataset(cg_data_list))
                cg_index += 1

            for group in range(len(for_each_class_group[0])):
                data_list.append(
                    ConcatDataset(
                        [for_each_class_group[k][group] for k in range(len(for_each_class_group))]
                    )
                )
        else:
            for location in combinations:
                path = os.path.join(root_dir, f"{0}/{location}/")
                data = ImageFolder(root=path, transform=transforms)
                data_list.append(data)

        return data_list

    # Buils combination dictionary for o2o datasets
    def build_type1_combination(self, group, test, filler):
        total = 3168
        counts = [int(0.97 * total), int(0.87 * total)]
        combinations = {}
        combinations['train_combinations'] = {
            ## correlated class
            ("bulldog",): [(group[0], counts[0]), (group[0], counts[1])],
            ("dachshund",): [(group[1], counts[0]), (group[1], counts[1])],
            ("labrador",): [(group[2], counts[0]), (group[2], counts[1])],
            ("corgi",): [(group[3], counts[0]), (group[3], counts[1])],
            ## filler
            ("bulldog", "dachshund", "labrador", "corgi"): [(filler, total - counts[0]), (filler, total - counts[1])],
        }
        ## TEST
        combinations['test_combinations'] = {
            ("bulldog",): [test[0], test[0]],
            ("dachshund",): [test[1], test[1]],
            ("labrador",): [test[2], test[2]],
            ("corgi",): [test[3], test[3]],
        }
        return combinations

    # Buils combination dictionary for m2m datasets
    def build_type2_combination(self, group, test):
        total = 3168
        counts = [total, total]
        combinations = {}
        combinations['train_combinations'] = {
            ## correlated class
            ("bulldog",): [(group[0], counts[0]), (group[1], counts[1])],
            ("dachshund",): [(group[1], counts[0]), (group[0], counts[1])],
            ("labrador",): [(group[2], counts[0]), (group[3], counts[1])],
            ("corgi",): [(group[3], counts[0]), (group[2], counts[1])],
        }
        combinations['test_combinations'] = {
            ("bulldog",): [test[0], test[1]],
            ("dachshund",): [test[1], test[0]],
            ("labrador",): [test[2], test[3]],
            ("corgi",): [test[3], test[2]],
        }
        return combinations


## Spawrious classes for each Spawrious dataset
class SpawriousO2O_easy(SpawriousBenchmark):
    def __init__(self, root_dir, test_envs, hparams):
        group = ["desert", "jungle", "dirt", "snow"]
        test = ["dirt", "snow", "desert", "jungle"]
        filler = "beach"
        combinations = self.build_type1_combination(group, test, filler)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir,
                         hparams['data_augmentation'], type1=True)


class SpawriousO2O_medium(SpawriousBenchmark):
    def __init__(self, root_dir, test_envs, hparams):
        group = ['mountain', 'beach', 'dirt', 'jungle']
        test = ['jungle', 'dirt', 'beach', 'snow']
        filler = "desert"
        combinations = self.build_type1_combination(group, test, filler)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir,
                         hparams['data_augmentation'], type1=True)


class SpawriousO2O_hard(SpawriousBenchmark):
    def __init__(self, root_dir, test_envs, hparams):
        group = ['jungle', 'mountain', 'snow', 'desert']
        test = ['mountain', 'snow', 'desert', 'jungle']
        filler = "beach"
        combinations = self.build_type1_combination(group, test, filler)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir,
                         hparams['data_augmentation'], type1=True)


class SpawriousM2M_easy(SpawriousBenchmark):
    def __init__(self, root_dir, test_envs, hparams):
        group = ['desert', 'mountain', 'dirt', 'jungle']
        test = ['dirt', 'jungle', 'mountain', 'desert']
        combinations = self.build_type2_combination(group, test)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir,
                         hparams['data_augmentation'])


class SpawriousM2M_medium(SpawriousBenchmark):
    def __init__(self, root_dir, test_envs, hparams):
        group = ['beach', 'snow', 'mountain', 'desert']
        test = ['desert', 'mountain', 'beach', 'snow']
        combinations = self.build_type2_combination(group, test)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir,
                         hparams['data_augmentation'])


class SpawriousM2M_hard(SpawriousBenchmark):
    ENVIRONMENTS = ["Test", "SC_group_1", "SC_group_2"]

    def __init__(self, root_dir, test_envs, hparams):
        group = ["dirt", "jungle", "snow", "beach"]
        test = ["snow", "beach", "dirt", "jungle"]
        combinations = self.build_type2_combination(group, test)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir,
                         hparams['data_augmentation'])
