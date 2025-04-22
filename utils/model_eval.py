import torch
import os
import numpy as np
from torch import nn
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from captum.attr import IntegratedGradients, GradientShap, Saliency
from utils.data_utils import load_image
from skimage.transform import resize


def predict(model, data_loader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            outputs = model(images)
            # Verifica se a saída é um único valor (dimensão 0)
            if outputs.squeeze().ndim == 0:
                # Trata o caso singular, colocando o valor em uma lista
                predictions.append(outputs.squeeze().item())
            else:
                # Caso comum, com um batch de saídas
                predictions.extend(outputs.squeeze().tolist())

    return predictions



def visualize_integrated_gradients_sex(
    image_path, image_id, label, model, input_size, device, output_size=None, save=False, save_path=""
):
    model.eval()

    img, label = load_image(image_path, label, input_size)
    img_tensor = img.unsqueeze(0).to(device)

    # Predict the sex of the image
    with torch.no_grad():
        logits = model(img_tensor)
        pred = torch.argmax(logits).item()  # Assuming logits are for classes [0, 1]
        confidence = torch.nn.functional.softmax(logits, dim=1)[0, pred].item()

    integrated_gradients = IntegratedGradients(model)
    atributos = integrated_gradients.attribute(
        img_tensor, target=pred, n_steps=200, internal_batch_size=32
    )

    atributos_np = atributos.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    atributos_np = np.clip(atributos_np, 0, 1)

    img_np = img.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    img_np = np.clip(img_np, 0, 1)

    # Convert the attributions to a heatmap
    heatmap = np.mean(atributos_np, axis=2)

    # Normalize the heatmap
    norm = Normalize(vmin=heatmap.min(), vmax=heatmap.max())
    heatmap_norm = norm(heatmap)

    # Applies the 'Reds' colormap to the heatmap
    cmap = cm.get_cmap("Reds")
    heatmap_reds = cmap(heatmap_norm)

    # Combines the original image with the red heatmap
    combined_image = np.copy(img_np)
    alpha = 0.6
    for i in range(3):
        combined_image[:, :, i] = (
            img_np[:, :, i] * (1 - alpha) + heatmap_reds[:, :, i] * alpha
        )

    # Resize the images to the desired size
    if output_size is not None:
        output_size = output_size
        img_np = resize(img_np, output_size)
        heatmap = resize(heatmap, output_size, anti_aliasing=True)
        combined_image = resize(combined_image, output_size)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Modify the titles for the subplots
    sex = 'Female' if label == 1 else 'Male'
    pred_sex = 'Female' if pred == 1 else 'Male'

    ax[0].imshow(img_np)
    ax[1].imshow(heatmap, cmap="Reds")
    ax[2].imshow(combined_image)

    ax[0].set_title(f"Original Image - Sex: {sex}", fontsize=16)
    ax[1].set_title("Integrated Gradient Heatmap", fontsize=16)
    ax[2].set_title(f"Pred: {pred_sex} - Conf: {confidence:.2f}", fontsize=16)

    for i in range(3):
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    # Save the figure to the disk
    if save:
        os.makedirs(save_path, exist_ok=True)

        plt.savefig(f"{save_path}/ig_{image_id}.png", bbox_inches='tight', pad_inches = 0.2)
        plt.close()

    else:
        plt.show()

def visualize_integrated_gradients_age(
    image_path, image_id, label, model, input_size, device, output_size=None, save=False, save_path=""
):
    model.eval()

    img, label = load_image(image_path, label, input_size)
    img_tensor = img.unsqueeze(0).to(device)

    # Predict the age of the image
    with torch.no_grad():
        pred = model(img_tensor).item()

    integrated_gradients = IntegratedGradients(model)
    atributos = integrated_gradients.attribute(
        img_tensor, target=None, n_steps=200, internal_batch_size=32
    )

    atributos_np = atributos.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    atributos_np = np.clip(atributos_np, 0, 1)

    img_np = img.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    img_np = np.clip(img_np, 0, 1)

    # Convert the attributions to a heatmap
    heatmap = np.mean(atributos_np, axis=2)

    # Normalize the heatmap
    norm = Normalize(vmin=heatmap.min(), vmax=heatmap.max())
    heatmap_norm = norm(heatmap)

    # Applies the 'Reds' colormap to the heatmap
    cmap = cm.get_cmap("Reds")
    heatmap_reds = cmap(heatmap_norm)

    # Combines the original image with the red heatmap
    combined_image = np.copy(img_np)
    alpha = 0.6
    for i in range(3):
        combined_image[:, :, i] = (
            img_np[:, :, i] * (1 - alpha) + heatmap_reds[:, :, i] * alpha
        )

    # Resize the images to the desired size
    if output_size is not None:
        output_size = output_size
        img_np = resize(img_np, output_size)
        heatmap = resize(heatmap, output_size, anti_aliasing=True)
        combined_image = resize(combined_image, output_size)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(img_np)
    ax[1].imshow(heatmap, cmap="Reds")
    ax[2].imshow(combined_image)

    ax[0].set_title(f"Original Image - Age: {label:.2f} years", fontsize=16)
    ax[1].set_title("Integrated Gradient Heatmap", fontsize=16)
    ax[2].set_title(f"Predicted Age: {pred:.2f} years", fontsize=16)

    for i in range(3):
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    # Save the figure to the disk
    if save:
        os.makedirs(save_path, exist_ok=True)

        plt.savefig(f"{save_path}/ig_{image_id}.png", bbox_inches='tight', pad_inches = 0.2)
        plt.close()

    else:
        plt.show()


def visualize_integrated_gradients_four_images(
    image_path, image_id, label, model, input_size, device, output_size=None, save=False, save_path=""
):
    model.eval()

    # Load and preprocess the image
    img, label = load_image(image_path, label, input_size)
    img_tensor = img.unsqueeze(0).to(device)

    # Predict the age of the image
    with torch.no_grad():
        pred = model(img_tensor).item()

    # Create baseline (black image)
    baseline_tensor = torch.zeros_like(img_tensor).to(device)

    # Compute Integrated Gradients
    integrated_gradients = IntegratedGradients(model)
    atributos = integrated_gradients.attribute(
        img_tensor, baselines=baseline_tensor, target=None, n_steps=200, internal_batch_size=32
    )

    atributos_np = atributos.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    atributos_np = np.clip(atributos_np, 0, 1)

    img_np = img.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    img_np = np.clip(img_np, 0, 1)

    # Convert the attributions to a heatmap
    heatmap = np.mean(atributos_np, axis=2)

    # Normalize the heatmap
    norm = Normalize(vmin=heatmap.min(), vmax=heatmap.max())
    heatmap_norm = norm(heatmap)

    # Apply the 'inferno' colormap to the IG Heatmap and Heatmap Overlap
    cmap_inferno = cm.get_cmap("inferno")
    heatmap_inferno = cmap_inferno(heatmap_norm)

    # Combine the original image with the inferno heatmap for overlap
    combined_image = np.copy(img_np)
    alpha = 0.6
    for i in range(3):
        combined_image[:, :, i] = (
            img_np[:, :, i] * (1 - alpha) + heatmap_inferno[:, :, i] * alpha
        )

    # Ensure baseline image is initialized
    baseline_np = np.zeros_like(img_np)

    # Resize the images to the desired size
    if output_size is not None:
        img_np = resize(img_np, output_size)
        heatmap = resize(heatmap, output_size, anti_aliasing=True)
        combined_image = resize(combined_image, output_size)
        baseline_np = resize(baseline_np, output_size)

    # Create subplots for the four images in a 2x2 grid
    fig, ax = plt.subplots(2, 2, figsize=(12, 10), dpi=150)

    ax[0, 0].imshow(img_np)
    ax[0, 1].imshow(baseline_np, cmap="gray")
    ax[1, 0].imshow(heatmap, cmap="inferno")  # IG Heatmap with inferno
    ax[1, 1].imshow(combined_image)  # Overlap with inferno

    ax[0, 0].set_title("Input Image", fontsize=14)
    ax[0, 1].set_title("Baseline (Black Image)", fontsize=14)
    ax[1, 0].set_title("Integrated Gradients Heatmap", fontsize=14)
    ax[1, 1].set_title("Heatmap Overlap", fontsize=14)

    for i in range(2):
        for j in range(2):
            ax[i, j].axis("off")

    # Adjust spacing between subplots
    plt.tight_layout()

    # Save the figure to the disk
    if save:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/ig_{image_id}.png", bbox_inches='tight', pad_inches=0.2)
        plt.close()
    else:
        plt.show()

def visualize_gradient_shap(
    image_path, label, model, input_size, device, output_size=None
):
    model.eval()

    img, label = load_image(image_path, label, input_size)
    img_tensor = img.unsqueeze(0).to(device)
    baselines = torch.zeros_like(img_tensor)

    # Realiza a previsão
    with torch.no_grad():
        pred = model(img_tensor).item()

    gradient_shap = GradientShap(model)
    atributos = gradient_shap.attribute(img_tensor, baselines, n_samples=50, stdevs=0.2)
    atributos_np = atributos.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    atributos_np = np.clip(atributos_np, 0, 1)

    img_np = img.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    img_np = np.clip(img_np, 0, 1)

    # Converte os atributos em um heatmap
    heatmap = np.mean(atributos_np, axis=2)

    # Normaliza o heatmap
    norm = Normalize(vmin=heatmap.min(), vmax=heatmap.max())
    heatmap_norm = norm(heatmap)

    # Aplica o colormap 'Reds' ao heatmap
    cmap = cm.get_cmap("Reds")
    heatmap_reds = cmap(heatmap_norm)

    # Combina a imagem original com o heatmap vermelho
    combined_image = np.copy(img_np)
    alpha = 0.6
    for i in range(3):
        combined_image[:, :, i] = (
            img_np[:, :, i] * (1 - alpha) + heatmap_reds[:, :, i] * alpha
        )

    # Redimensiona as imagens para o tamanho desejado
    if output_size is not None:
        output_size = output_size
        img_np = resize(img_np, output_size)
        heatmap = resize(heatmap, output_size, anti_aliasing=True)
        combined_image = resize(combined_image, output_size)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(img_np)
    ax[1].imshow(heatmap, cmap="Reds")
    ax[2].imshow(combined_image)

    ax[0].set_title(f"Original Image - Age: {label:.2f} years")
    ax[1].set_title("Gradient Shap Heatmap")
    ax[2].set_title(f"Predicted Age: {pred:.2f} years")

    for i in range(3):
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    plt.show()


def visualize_saliency(
    image_path, label, model, input_size, device, output_size=None
):
    model.eval()

    img, label = load_image(image_path, label, input_size)
    img_tensor = img.unsqueeze(0).to(device)

    # Realiza a previsão
    with torch.no_grad():
        pred = model(img_tensor).item()

    saliency = Saliency(model)
    atributos = saliency.attribute(img_tensor, abs=False)
    atributos_np = atributos.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    atributos_np = np.clip(atributos_np, 0, 1)

    img_np = img.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    img_np = np.clip(img_np, 0, 1)

    # Converte os atributos em um heatmap
    heatmap = np.mean(atributos_np, axis=2)

    # Normaliza o heatmap
    norm = Normalize(vmin=heatmap.min(), vmax=heatmap.max())
    heatmap_norm = norm(heatmap)

    # Aplica o colormap 'Reds' ao heatmap
    cmap = cm.get_cmap("Reds")
    heatmap_reds = cmap(heatmap_norm)

    # Combina a imagem original com o heatmap vermelho
    combined_image = np.copy(img_np)
    alpha = 0.6
    for i in range(3):
        combined_image[:, :, i] = (
            img_np[:, :, i] * (1 - alpha) + heatmap_reds[:, :, i] * alpha
        )

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Redimensiona as imagens para o tamanho desejado
    if output_size is not None:
        output_size = output_size
        img_np = resize(img_np, output_size)
        heatmap = resize(heatmap, output_size, anti_aliasing=True)
        combined_image = resize(combined_image, output_size)

    ax[0].imshow(img_np)
    ax[1].imshow(heatmap, cmap="Reds")
    ax[2].imshow(combined_image)

    ax[0].set_title(f"Original Image - Age: {label:.2f} years")
    ax[1].set_title("Saliency Heatmap")
    ax[2].set_title(f"Predicted Age: {pred:.2f} years")

    for i in range(3):
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    plt.show()


def generate_heatmap(attributes, cmap_name):
    attributes = np.abs(attributes)
    heatmap = np.mean(attributes, axis=2)
    norm = Normalize(vmin=heatmap.min(), vmax=heatmap.max())
    heatmap_norm = norm(heatmap)
    cmap = cm.get_cmap(cmap_name)
    return cmap(heatmap_norm)


def combine_heatmap_with_image(image, heatmap, alpha=0.6):
    combined_image = np.copy(image)
    for i in range(3):
        combined_image[:, :, i] = (
            image[:, :, i] * (1 - alpha) + heatmap[:, :, i] * alpha
        )
    return combined_image


def visualize_captum_methods(
    image_path, label, model, input_size, device, output_size=None
):
    img, label = load_image(image_path, label, input_size)
    img_tensor = img.unsqueeze(0).to(device)
    model.eval()
    model.to(device)

    # Realiza a previsão
    with torch.no_grad():
        pred = model(img_tensor).item()

    # Calcula atributos usando diferentes métodos
    integrated_gradients = IntegratedGradients(model)
    ig_attribs = integrated_gradients.attribute(img_tensor)
    ig_attribs_np = ig_attribs.squeeze().detach().cpu().numpy().transpose(1, 2, 0)

    gradient_shap = GradientShap(model)
    baselines = torch.zeros_like(img_tensor).to(device)
    gs_attribs = gradient_shap.attribute(img_tensor, baselines)
    gs_attribs_np = gs_attribs.squeeze().detach().cpu().numpy().transpose(1, 2, 0)

    saliency = Saliency(model)
    saliency_attribs = saliency.attribute(img_tensor)
    saliency_attribs_np = (
        saliency_attribs.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    )

    # Prepara a imagem original para visualização
    img_hwc = img_tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    img_hwc = np.clip(img_hwc, 0, 1)

    # Gera os heatmaps
    ig_heatmap = generate_heatmap(ig_attribs_np, "Reds")
    gs_heatmap = generate_heatmap(gs_attribs_np, "Reds")
    saliency_heatmap = generate_heatmap(saliency_attribs_np, "Reds")

    # Combina a imagem original com os heatmaps
    ig_combined = combine_heatmap_with_image(img_hwc, ig_heatmap)
    gs_combined = combine_heatmap_with_image(img_hwc, gs_heatmap)
    saliency_combined = combine_heatmap_with_image(img_hwc, saliency_heatmap)

    # Redimensiona as imagens para o tamanho desejado
    if output_size is not None:
        output_size = output_size
        img_hwc = resize(img_hwc, output_size)
        ig_combined = resize(ig_combined, output_size)
        gs_combined = resize(gs_combined, output_size)
        saliency_combined = resize(saliency_combined, output_size)

    # Visualiza os atributos
    fig, ax = plt.subplots(1, 4, figsize=(15, 5))
    ax[0].imshow(img_hwc)
    ax[0].set_title(f"Original Image - A: {label:.2f} | P: {pred:.2f}")

    ax[1].imshow(ig_combined)
    ax[1].set_title("Integrated Gradients")

    ax[2].imshow(gs_combined)
    ax[2].set_title("Gradient Shap")

    ax[3].imshow(saliency_combined)
    ax[3].set_title("Saliency")

    for i in range(4):
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    plt.show()
