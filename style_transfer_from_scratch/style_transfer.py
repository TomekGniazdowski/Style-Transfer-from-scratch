import cv2
import torch
from torch import nn, optim

from style_transfer_from_scratch.utils import (
    content_loss, style_loss, total_variation_loss, show_loss_plot, unnorm_image
)


def run_style_transfer(
    generated_img: torch.Tensor,
    features_extractor: nn.Module, 
    optimizer: optim.Optimizer,
    content_img: torch.Tensor,
    style_img: torch.Tensor,
    content_weight: float, # In the paper, alpha
    style_weight: float, # In the paper, beta
    total_variation_weight: float,
    experiment_name: str,
    experiments_path: str,
    save_generated_img: bool = True,
    ):
    
    features_extractor.eval()
    content_gt = features_extractor(content_img)['content_features']
    style_gt = features_extractor(style_img)['style_features']
    epoch = 0
    losses = []

    def closure():
        optimizer.zero_grad()
        
        model_pred = features_extractor(generated_img)
        content_pred = model_pred['content_features']
        style_pred = model_pred['style_features'] 

        content_loss_ = content_loss(content_gt, content_pred)
        style_loss_ = style_loss(style_gt, style_pred, layers_weights=[1/5]*5)
        total_variation_loss_ = total_variation_loss(generated_img)
        loss = content_loss_ * content_weight + style_loss_ * style_weight + total_variation_loss_ * total_variation_weight
        loss.backward(retain_graph=True)
        
        with torch.no_grad():
            nonlocal epoch
            epoch += 1
            losses.append(loss.item())
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: \
                    Loss: {loss:.1f}, \
                    CL: {content_loss_:.1f}, \
                    SL: {style_loss_:.1f}, \
                    TVL: {total_variation_loss_:.1f}")
            if save_generated_img and (epoch+1) % 10 == 0:
                img_to_save = unnorm_image(generated_img)
                cv2.imwrite(
                    f"{experiments_path}/{experiment_name}/generated_img.jpg",
                    img_to_save[..., ::-1]
                    )
        
        return loss
    
    optimizer.step(closure)
    show_loss_plot(losses)
    
    return generated_img