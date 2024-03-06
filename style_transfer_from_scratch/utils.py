import torch
from torch.autograd import Variable
import cv2
import torch.nn.functional as f
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os


IMAGENET_MEAN = torch.tensor([123.675, 116.28, 103.53])
IMAGENET_STD = torch.tensor([1, 1, 1])
preprocessing_img = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255)),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def unnorm_image(img: torch.Tensor):
    img = img[0].detach().cpu().numpy().transpose(1, 2, 0)
    img += np.array(IMAGENET_MEAN).reshape((1, 1, 3))
    img = np.clip(img, 0, 255).astype('uint8')
    return img

def show_loss_plot(losses: list):
    plt.plot(losses)
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss vs Epochs")
    plt.show()

def read_preprocess_image(img_path: str, img_height: int):
    img = cv2.imread(img_path)[:, :, ::-1]
    img = cv2.resize(img, dsize=(int(img.shape[1] * img_height / img.shape[0]), img_height), 
                     interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32)
    img /= 255.0
    img = preprocessing_img(img)
    img = img.unsqueeze(0) # Adding batch dimension
    return img

def content_loss(features_gt: torch.Tensor, features_pred: torch.Tensor):
    return f.mse_loss(features_gt, features_pred, reduction='mean')

def calculate_gram_matrix(features: torch.Tensor):
    (b, c, h, w) = features.size()
    features = features.view(b, c, w * h)
    features_t = features.transpose(1, 2)
    gram_matrix = features.bmm(features_t)
    gram_matrix /= c * h * w # Normalization step
    return gram_matrix

def style_loss(features_gt: torch.Tensor, features_pred: torch.Tensor, layers_weights: list):
    total_style_loss = 0
    for i in range(len(features_gt)):
        gt = calculate_gram_matrix(features_gt[i])
        pred = calculate_gram_matrix(features_pred[i])
        loss = f.mse_loss(gt, pred, reduction='sum') * layers_weights[i]
        total_style_loss += loss
    return total_style_loss

def total_variation_loss(image: torch.Tensor):
    return torch.sum(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + torch.sum(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))

def initialize_image(method: str, size: tuple, device: str):
    if method == 'white_noise':
        img = np.random.uniform(0., 255., size).astype(np.float32)
    elif method == 'normal_distr':
        img = np.random.normal(loc=0., scale=90., size=size).astype(np.float32)
    img = torch.from_numpy(img).to(device)
    img = Variable(img, requires_grad=True) # We want to optimize the image
    return img

def create_experiment_folder(experiment_name: str, experiments_folder_path: str):
    os.makedirs(f'{experiments_folder_path}/{experiment_name}', exist_ok=True)