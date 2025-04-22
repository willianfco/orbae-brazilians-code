import os
import pandas as pd
import math
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image



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

    return image


def normalize(x, min_x, max_x, min_value, max_value):
    return (x - min_x) * (max_value - min_value) / (max_x - min_x) + min_value

def add_augmentation_multiplier(df: pd.DataFrame, min_multiplier: int, max_multiplier: int):
    age_group_freq = df["age_group"].value_counts(normalize=True).to_dict()
    df["train_freq"] = df["age_group"].apply(lambda x: age_group_freq[x])
    
    inverted_freq = df["train_freq"].apply(lambda x: 1 / x)
    min_inv_freq = inverted_freq.min()
    max_inv_freq = inverted_freq.max()

    df["augmentation_multiplier"] = inverted_freq.apply(
        lambda x: math.ceil(normalize(x, min_inv_freq, max_inv_freq, min_multiplier, max_multiplier))
    )

    return df

def save_augmented_images(df, input_size, augment_config, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # Create a new DataFrame to store the paths and labels of the augmented images
    augmented_df = pd.DataFrame(columns=["path", "age_in_years", "sex", "age_group"])

    for index, row in df.iterrows():
        image_path = row["path"]
        age_in_years = row["age_in_years"]
        sex = row["sex"]
        age_group = row["age_group"]

        # Load the original image
        image, _ = load_image(image_path, age_in_years, input_size)

        # Save the original image
        filename = os.path.basename(image_path).split(".")[0]
        save_path = os.path.join(save_dir, f"{filename}_original.jpg")
        save_image(image, save_path)

        # Add the original image to the new DataFrame
        new_row = pd.DataFrame({"path": [save_path], "age_in_years": [age_in_years], "sex": [sex], "age_group": [age_group]})
        augmented_df = pd.concat([augmented_df, new_row], ignore_index=True)

        # Generate and save augmented images
        num_augmented_images = row["augmentation_multiplier"]
        for i in range(num_augmented_images - 1):
            augmented_image = data_augmentation(image, **augment_config)
            save_path = os.path.join(save_dir, f"{filename}_augmented_{i + 1}.jpg")
            save_image(augmented_image, save_path)

            # Add the augmented image to the new DataFrame
            new_row = pd.DataFrame({"path": [save_path], "age_in_years": [age_in_years], "sex": [sex], "age_group": [age_group]})
            augmented_df = pd.concat([augmented_df, new_row], ignore_index=True)

    # Save the new DataFrame as a CSV file
    augmented_csv_path = os.path.join(save_dir, "augmented_labels.csv")
    augmented_df.to_csv(augmented_csv_path, index=False)
