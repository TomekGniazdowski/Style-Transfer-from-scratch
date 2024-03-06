import sys
sys.path.append('..')

import cv2
from torch import optim, cuda, int64

from style_transfer_from_scratch.model import VGG19FeaturesExtractor
from style_transfer_from_scratch.style_transfer import run_style_transfer
from style_transfer_from_scratch.utils import read_preprocess_image, initialize_image, unnorm_image, create_experiment_folder


def experiment_vgg19(
    img_content_path: str,
    img_style_path: str,
    img_height: int,
    initial_image_distribution: str,
    content_weight: float,
    style_weight: float,
    total_variation_weight: float,
    max_iter: int,
    experiment_name: str,
    experiments_folder_path: str = '../data/experiments'
):
    
    DEVICE = 'cuda' if cuda.is_available() else 'cpu'

    img_content = read_preprocess_image(img_path=img_content_path, img_height=img_height).to(DEVICE)
    img_style = read_preprocess_image(img_path=img_style_path, img_height=img_height).to(DEVICE)
    create_experiment_folder(experiment_name, experiments_folder_path)

    cv2.imwrite(f'{experiments_folder_path}/{experiment_name}/content_img.jpg', img_content[0].type(dtype=int64).cpu().numpy().transpose(1, 2, 0)[..., ::-1])
    cv2.imwrite(f'{experiments_folder_path}/{experiment_name}/content_img_unnorm.jpg', unnorm_image(img_content)[..., ::-1])
    cv2.imwrite(f'{experiments_folder_path}/{experiment_name}/style_img.jpg', img_style[0].type(dtype=int64).cpu().numpy().transpose(1, 2, 0)[..., ::-1])
    cv2.imwrite(f'{experiments_folder_path}/{experiment_name}/style_img_unnorm.jpg', unnorm_image(img_style)[..., ::-1])
    print('content image shape:', img_content.shape, 'style image shape:', img_style.shape)

    generated_img = initialize_image(method=initial_image_distribution, size=img_content.size(), device=DEVICE)
    vgg19_features_extractor = VGG19FeaturesExtractor().to(DEVICE)
    optimizer = optim.LBFGS([generated_img], max_iter=max_iter)

    generated_img = run_style_transfer(
        generated_img=generated_img,
        features_extractor=vgg19_features_extractor,
        optimizer=optimizer,
        content_img=img_content,
        style_img=img_style,
        content_weight=content_weight,
        style_weight=style_weight,
        total_variation_weight=total_variation_weight,
        experiment_name=experiment_name,
        experiments_path=experiments_folder_path,
        save_generated_img=True
    )
    
    cv2.imwrite(f'{experiments_folder_path}/{experiment_name}/generated_img_final.jpg', unnorm_image(generated_img)[..., ::-1])
    print(f'All the images are saved in the experiment folder.')