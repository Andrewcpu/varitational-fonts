import torch.nn.functional as F
import torch
def add_noise(images, noise_factor=0.1):
    noise = torch.randn_like(images) * noise_factor
    noisy_images = images + noise
    noisy_images = torch.clamp(noisy_images, 0, 1)
    return noisy_images


def get_features(image, model, layers):
    features = {}
    # Convert 1-channel grayscale images to 3-channel RGB images
    x = image.repeat(1, 3, 1, 1)  # Repeat the grayscale channel to make it 3-channel
    for name, layer in enumerate(model.children()):
        x = layer(x)
        if str(name) in layers:
            # print(str(name))
            features[str(name)] = x
    return features


def perceptual_loss(recon_x, x, model, layers):
    recon_features = get_features(recon_x, model, layers)
    x_features = get_features(x, model, layers)
    loss = 0
    for name in layers:
        loss = loss + F.mse_loss(recon_features[name], x_features[name])
    return loss



def laplacian_filter(image, device):
    """Applies the Laplacian filter to an image."""
    laplacian_kernel = torch.tensor([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    laplacian = F.conv2d(image, laplacian_kernel, padding=1)
    return laplacian


def horizontal_line_filter(image, device):
    """Applies a horizontal line filter to an image."""
    horizontal_kernel = torch.tensor([[-1, -1, -1],
                                      [2, 2, 2],
                                      [-1, -1, -1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    horizontal_lines = F.conv2d(image, horizontal_kernel, padding=1)
    return horizontal_lines


def sobel_filter(image, device, eps=1e-8):
    """Applies the Sobel filter to an image."""
    sobel_kernel_x = torch.tensor([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_kernel_y = torch.tensor([[-1, -2, -1],
                                   [0, 0, 0],
                                   [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_x = F.conv2d(image, sobel_kernel_x, padding=1)
    sobel_y = F.conv2d(image, sobel_kernel_y, padding=1)
    sobel = torch.sqrt(sobel_x ** 2 + sobel_y ** 2 + eps)
    sobel = (sobel - sobel.min()) / (sobel.max() - sobel.min() + eps)  # Normalize the output
    return sobel


def vertical_line_filter(image, device):
    """Applies a vertical line filter to an image."""
    vertical_kernel = torch.tensor([[-1, 2, -1],
                                    [-1, 2, -1],
                                    [-1, 2, -1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    vertical_lines = F.conv2d(image, vertical_kernel, padding=1)
    return vertical_lines


def multi_scale_filters(image, device, scales=[1, 0.5, 0.25]):
    """Applies Laplacian, horizontal, vertical line filters, and Sobel filter to an image at multiple scales."""
    filtered_images = []
    for scale in scales:
        if scale != 1:
            # Resize image to the new scale
            scaled_image = F.interpolate(image, scale_factor=scale, mode='bilinear', align_corners=True,
                                         recompute_scale_factor=True)
        else:
            scaled_image = image
        laplacian = laplacian_filter(scaled_image, device)
        horizontal_lines = horizontal_line_filter(scaled_image, device)
        vertical_lines = vertical_line_filter(scaled_image, device)
        sobel = sobel_filter(scaled_image, device)
        filtered_images.append((laplacian, horizontal_lines, vertical_lines, sobel))
    return filtered_images


def multi_scale_curve_loss(recon_x, x, device, scales=None, sobel_weight=0.5):
    """Computes multi-scale curve loss between reconstructed images and target images."""
    if scales is None:
        scales = [1.0, 0.5, 0.25]
    total_loss = 0
    recon_filtered = multi_scale_filters(recon_x, device, scales)
    target_filtered = multi_scale_filters(x, device, scales)
    for (recon_laplacian, recon_horizontal, recon_vertical, recon_sobel), (
    target_laplacian, target_horizontal, target_vertical, target_sobel) in zip(recon_filtered, target_filtered):
        laplacian_loss = F.mse_loss(recon_laplacian, target_laplacian)
        # horizontal_loss = F.mse_loss(recon_horizontal, target_horizontal)
        # vertical_loss = F.mse_loss(recon_vertical, target_vertical)
        # sobel_loss = F.l1_loss(recon_sobel, target_sobel)  # Use L1 loss for Sobel
        total_loss += laplacian_loss
    return total_loss / len(scales)

