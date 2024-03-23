from torchvision.models import vgg19, VGG19_Weights, vgg16, VGG16_Weights

from gan import Discriminator, CharacterDiscriminator, StyleDiscriminator
from loss_utils import multi_scale_curve_loss, perceptual_loss, add_noise, multi_scale_filters
from save_utils import AbstractAdvancedModule, AbstractLossLayer
from vae import VAE
import torch

import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AdvDiscriminatorLoss(AbstractLossLayer):
    def __init__(self, model):
        super().__init__(model)

    def calculate_loss(self, batch_data, model_output, prepared_input_passthrough_data, shared_output=None):
        predictions = model_output
        loss = F.binary_cross_entropy_with_logits(predictions, prepared_input_passthrough_data)
        return loss


class AdvStyleLoss(AbstractLossLayer):
    def __init__(self, model):
        super().__init__(model)

    def calculate_loss(self, batch_data, model_output, prepared_input_passthrough_data, shared_output=None):
        combined_labels = prepared_input_passthrough_data
        predictions = model_output

        loss = F.binary_cross_entropy_with_logits(predictions, combined_labels)
        return loss


class AdvCurveLoss(AbstractLossLayer):
    def __init__(self, model, weight=None):  # 5000
        super().__init__(model)
        self.weight = weight

    def _multi_scale_curve_loss(self, recon_x, x, device, scales=None):
        """Computes multi-scale curve loss between reconstructed images and target images."""
        if scales is None:
            scales = [1.0, 0.75, 0.5]
        total_loss = 0
        recon_filtered = multi_scale_filters(recon_x, device, scales)
        target_filtered = multi_scale_filters(x, device, scales)
        for (recon_laplacian, recon_horizontal, recon_vertical, recon_sobel), (
                target_laplacian, target_horizontal, target_vertical, target_sobel) in zip(recon_filtered,
                                                                                           target_filtered):
            laplacian_loss = F.mse_loss(recon_laplacian, target_laplacian)
            total_loss += laplacian_loss
        return total_loss / len(scales)

    def calculate_loss(self, batch_data, model_output, prepared_input_passthrough_data, shared_output=None):
        recon_batch, mu, logvar, out_z, actual_z, style = model_output
        images, char_embs, cap_idx, secondary_images, secondary_char_embs, secondary_idx, label, random_image = prepared_input_passthrough_data
        return multi_scale_curve_loss(recon_batch, secondary_images, device) * self.weight


class AdvSimilarityLoss(AbstractLossLayer):
    def __init__(self, model, similarity_weight=None, kl_divergence_beta=None):  # 0.25, ?
        super().__init__(model)
        self.similarity_weight = similarity_weight
        self.kl_divergence_beta = kl_divergence_beta

    @property
    def performs_own_passthrough(self):
        """
        We will perform a passthrough operation on another input
        """
        return True

    def passthrough(self, batch_data, prepared_input_passthrough_data, shared_output=None):
        images, char_embs, cap_idx, secondary_images, secondary_char_embs, secondary_idx, label, random_image = prepared_input_passthrough_data
        self.model.encoder.eval()
        with torch.no_grad():
            mu2, logvar2 = self.model.encoder(secondary_images, secondary_char_embs, secondary_idx)
        self.model.encoder.train()
        return mu2, logvar2

    def calculate_loss(self, batch_data, model_output, prepared_input_passthrough_data, shared_output=None):
        recon_batch, mu, logvar, out_z, actual_z, style = shared_output
        # images, char_embs, cap_idx, secondary_images, secondary_char_embs, secondary_idx, label, random_image = shared_output
        mu2, logvar2 = model_output
        similarity_loss = F.mse_loss(mu.clone().detach(), mu2) * self.similarity_weight
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return similarity_loss + (kl_div * self.kl_divergence_beta)


class AdvPerceptualLoss(AbstractLossLayer):
    def __init__(self, model, layers=None, scale=None, pretrained_vgg=None):  # 100.0
        super().__init__(model)
        self.layers = layers
        self.feature_extractor = pretrained_vgg
        self.scale = scale

    def _get_features(self, image, model, layers):
        features = {}
        # Convert 1-channel grayscale images to 3-channel RGB images
        x = image.repeat(1, 3, 1, 1)  # Repeat the grayscale channel to make it 3-channel
        for name, layer in enumerate(model.children()):
            x = layer(x)
            if str(name) in layers:
                # print(str(name))
                features[str(name)] = x
        return features

    def _perceptual_loss(self, recon_x, x, model, layers):
        recon_features = self._get_features(recon_x, model, layers)
        x_features = self._get_features(x, model, layers)
        loss = 0
        for name in layers:
            loss = loss + F.mse_loss(recon_features[name], x_features[name])
        return loss

    def calculate_loss(self, batch_data, model_output, prepared_input_passthrough_data, shared_output=None):
        recon_batch, mu, logvar, out_z, actual_z, style = model_output
        images, char_embs, cap_idx, secondary_images, secondary_char_embs, secondary_idx, label, random_image = prepared_input_passthrough_data
        return self._perceptual_loss(recon_batch, secondary_images, self.feature_extractor, self.layers) * self.scale


class AdvMSELoss(AbstractLossLayer):
    def __init__(self, model, ratio=None, secondary_ratio=None):
        super().__init__(model)
        self.ratio = ratio
        self.secondary_ratio = secondary_ratio

    def calculate_loss(self, batch_data, model_output, prepared_input_passthrough_data, shared_output=None):
        recon_batch, mu, logvar, out_z, actual_z, style = model_output
        images, char_embs, cap_idx, secondary_images, secondary_char_embs, secondary_idx, label, random_image = prepared_input_passthrough_data
        loss = F.mse_loss(recon_batch, secondary_images, reduction='sum') * self.ratio
        loss = loss + F.mse_loss(out_z, secondary_images, reduction='sum') * self.secondary_ratio
        return loss

class AdvGANMSELoss(AbstractLossLayer):
    def __init__(self, model, ratio):
        super().__init__(model)
        self.ratio = ratio

    def calculate_loss(self, batch_data, model_output, prepared_input_passthrough_data, shared_output=None):
        images, char_embs, cap_idx, secondary_images, secondary_char_embs, secondary_idx, label, random_image = prepared_input_passthrough_data
        return F.mse_loss(model_output, images, reduction='sum') * self.ratio

class AdvGANLoss(AbstractLossLayer):
    def __init__(self, model, primary_ratio, style_ratio, discriminator, style_discriminator):
        super().__init__(model)
        self.primary_ratio = primary_ratio
        self.style_ratio = style_ratio
        self.noise_factor = 0.25
        self.discriminator = discriminator.model
        self.style_discriminator = style_discriminator.model

    def calculate_loss(self, batch_data, model_output, prepared_input_passthrough_data, shared_output=None):
        recon_batch, mu, logvar, out_z, actual_z, style = model_output
        images, char_embs, cap_idx, secondary_images, secondary_char_embs, secondary_idx, label, random_image = prepared_input_passthrough_data

        noisy_fake_images = add_noise(out_z, self.noise_factor)
        noisy_source_images = add_noise(images, self.noise_factor)

        # The generator tries to fool the discriminator
        self.discriminator.eval()
        discriminator_loss = F.binary_cross_entropy_with_logits(self.discriminator(noisy_fake_images),
                                                                torch.ones_like(self.discriminator(noisy_fake_images)))
        self.discriminator.train()

        # The generator tries to match the style of the source images
        self.style_discriminator.eval()
        style_disc_output = self.style_discriminator(noisy_source_images, recon_batch)
        style_loss = F.binary_cross_entropy_with_logits(style_disc_output, torch.ones_like(style_disc_output))
        self.style_discriminator.train()

        # Combine the losses with a weight factor
        # weight_factor = 100.0  # Adjust this value based on the desired balance between the two losses
        style_loss = style_loss * self.style_ratio
        total_loss = self.primary_ratio * discriminator_loss + style_loss

        return total_loss
