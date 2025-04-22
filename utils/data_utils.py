import numpy as np
import torch
import random
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt


def load_image(image_path, label, input_size):
    image = Image.open(image_path)
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(input_size),
            transforms.ToTensor(),
        ]
    )
    image = transform(image)
    return image, label

def add_noise(image, noise_type, noise_intensity):
    # Gaussian Noise
    if noise_type == "gaussian":
        mean = 0
        std = noise_intensity
        gauss = torch.normal(mean, std, image.size())
        noisy = image + gauss
        return torch.clamp(noisy, 0, 1)

    # Salt & Pepper Noise
    elif noise_type == "salt_pepper":
        salt_prob = noise_intensity / 2
        pepper_prob = noise_intensity / 2
        mask_salt = torch.rand(image.size()) < salt_prob
        mask_pepper = torch.rand(image.size()) < pepper_prob
        image[mask_salt] = 1
        image[mask_pepper] = 0
        return image
    
    # White Noise
    elif noise_type == "stationary":
        noise = torch.randn(image.size()) * noise_intensity
        noisy = image + noise
        return torch.clamp(noisy, 0, 1)

    # Periodic Noise
    elif noise_type == "periodic":
        rows, cols = image.size(1), image.size(2)
        y = np.linspace(0, 2 * np.pi, rows)
        x = np.linspace(0, 2 * np.pi, cols)
        x, y = np.meshgrid(x, y)
        sin_wave = torch.tensor(np.sin(x + y)).float()
        noisy = image + noise_intensity * sin_wave
        return torch.clamp(noisy, 0, 1)


def data_augmentation(
    image,
    augment_prob=0.8,
    flip_horizontal=True,
    flip_vertical=True,
    flip_prob=0.5,
    random_brightness=True,
    brightness_factor=0.2,
    random_contrast=True,
    contrast_factor=0.2,
    random_rotation=True,
    rotation_factor=20,
    random_translation=True,
    translation_factor=(10, 10),
    random_zoom=True,
    zoom_factors=(0.9, 1.1),
    random_erasing=True,
    erasing_prob=0.1,
    erasing_scale=(0.05, 0.10),
    erasing_ratio=(0.3, 3.3),
    random_noise=True,
    noise_prob=0.5,
):
    transforms_list = []

    if flip_horizontal:
        transforms_list.append(transforms.RandomHorizontalFlip(p=flip_prob))

    if flip_vertical:
        transforms_list.append(transforms.RandomVerticalFlip(p=flip_prob))

    if random_brightness:
        transforms_list.append(transforms.ColorJitter(brightness=brightness_factor))

    if random_contrast:
        transforms_list.append(transforms.ColorJitter(contrast=contrast_factor))

    if random_rotation or random_translation or random_zoom:
        transforms_list.append(
            transforms.RandomAffine(
                degrees=rotation_factor if random_rotation else 0,
                translate=translation_factor if random_translation else None,
                scale=zoom_factors if random_zoom else None,
            )
        )

    if random_erasing:
        transforms_list.append(
            transforms.RandomErasing(
                p=erasing_prob, scale=erasing_scale, ratio=erasing_ratio
            )
        )

    augment_transform = transforms.RandomApply(transforms_list, p=augment_prob)
    image = augment_transform(image)

    if random_noise and random.random() < noise_prob:
        
        noise_types = ["gaussian", "salt_pepper", "stationary", "periodic"]
        selected_noise_type = random.choice(noise_types)
        
        selected_noise_intensity = random.uniform(0.05, 0.2)
        
        image = add_noise(image, selected_noise_type, selected_noise_intensity)


    return image


def load_transform_image(
    image_path, label, input_size, augment_config={}, augment=True
):
    image, label = load_image(image_path, label, input_size)

    if augment:
        image = data_augmentation(image, **augment_config)

    return image, label


class CustomDataset(Dataset):
    def __init__(
        self, df, input_size, augment, augment_config, multitask, label, num_augmented_images
    ):
        self.df = df
        self.input_size = input_size
        self.augment = augment
        self.augment_config = augment_config
        self.multitask = multitask
        self.singletask = label if not multitask else None
        self.num_augmented_images = num_augmented_images if augment else 1

    def __len__(self):
        return len(self.df) * self.num_augmented_images

    def __getitem__(self, idx):
        row_idx = idx // self.num_augmented_images

        if self.multitask:
            image_path = self.df.iloc[row_idx]["path"]
            label1 = self.df.iloc[row_idx]["sex"]
            label2 = self.df.iloc[row_idx]["age_in_years"]
            labels = (label1, label2)
            image, label = load_transform_image(
                image_path, labels, self.input_size, self.augment_config, self.augment
            )
            return image, labels

        else:
            image_path = self.df.iloc[row_idx]["path"]

            if self.singletask == "age_in_years":
                label = self.df.iloc[row_idx]["age_in_years"]
            elif self.singletask == "sex":
                label = self.df.iloc[row_idx]["sex"]

            image, label = load_transform_image(
                image_path, label, self.input_size, self.augment_config, self.augment
            )
            return image, label


def prepare_dataset(
    df,
    batch_size=32,
    shuffle=True,
    augment=True,
    multitask=False,
    label="age_in_years",
    augment_config={},
    input_size=(299, 299),
    num_augmented_images=5,
):
    dataset = CustomDataset(
        df, input_size, augment, augment_config, multitask, label, num_augmented_images
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        timeout=300,
    )

    return dataloader


def eval_dataset(dataset, multitask=False, label="age_in_years", NUM_ROWS=2, NUM_COLS=2):
    def tensor_to_image(tensor):
        img = tensor.numpy()  # Convert the tensor to a numpy array
        img = img[0]
        return img

    def class_to_sex(class_idx):
        return "M" if class_idx == 0 else "F"

    # Load a batch of images and labels

    images, labels = next(iter(dataset))


    # Display tensor shape and number of batches

    print(f"Number of Batches: {len(dataset)}")
    print(f"Batch Tensor Shape: {images.shape}")

    # Verify basic statistics of the images and labels from the dataloader

    print(f"Minimum value of the tensors: {images.cpu().numpy().min()}")
    print(f"Maximum value of the tensors: {images.cpu().numpy().max()}")
    print(f"Mean value of the tensors: {images.cpu().numpy().mean()}")
    print(f"Median value of the tensors: {np.median(images.cpu().numpy())}")
    print(f"Standard deviation of the tensors: {images.cpu().numpy().std()}")
    print(
        "Disclamer: The values above are from the batch loaded by the dataloader. They are not the mean, median and standard deviation of the entire dataset."
    )

    # show the images and their labels

    NUM_ROWS = 2
    NUM_COLS = 2

    fig, axes = plt.subplots(NUM_ROWS, NUM_COLS, figsize=(5, 5))

    if multitask:
        for row in range(NUM_ROWS):
            for col in range(NUM_COLS):
                idx = row * NUM_COLS + col
                ax = axes[row, col]
                ax.imshow(tensor_to_image(images[idx]), cmap="gray")
                sex_class = labels[0][idx]
                gender = class_to_sex(sex_class)
                age = labels[1][idx].item()
                ax.set_title(f"Sex: {gender}, Age: {age}")
                ax.axis("off")
    else:
        if label == "age_in_years":
            for row in range(NUM_ROWS):
                for col in range(NUM_COLS):
                    idx = row * NUM_COLS + col
                    ax = axes[row, col]
                    ax.imshow(tensor_to_image(images[idx]), cmap="gray")
                    ax.set_title(f"Age: {labels[idx].item()}")
                    ax.axis("off")

        elif label == "sex":
            for row in range(NUM_ROWS):
                for col in range(NUM_COLS):
                    idx = row * NUM_COLS + col
                    ax = axes[row, col]
                    ax.imshow(tensor_to_image(images[idx]), cmap="gray")
                    sex_class = labels[idx]
                    gender = class_to_sex(sex_class)
                    ax.set_title(f"Sex: {gender}")
                    ax.axis("off")

    plt.tight_layout()
    plt.show()
