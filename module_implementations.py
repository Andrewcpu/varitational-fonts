from torchvision.models import vgg19, VGG19_Weights, vgg16, VGG16_Weights

from gan import Discriminator, CharacterDiscriminator, StyleDiscriminator
from loss_module_implementations import AdvMSELoss, AdvPerceptualLoss, AdvSimilarityLoss, AdvCurveLoss, \
    AdvDiscriminatorLoss, AdvStyleLoss, AdvGANLoss, AdvGANMSELoss
from loss_utils import multi_scale_curve_loss, perceptual_loss, add_noise, multi_scale_filters
from save_utils import AbstractAdvancedModule, AbstractLossLayer
from vae import VAE, ChunkEnhancer, ImageChunkProcessor
import torch

import torch.optim as optim
import torch.nn.functional as F

latent_dims = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# model = VAE(input_channels=1, hidden_dims=16, latent_dims=latent_dims, device=device)

class EnhancementModule(AbstractAdvancedModule):
    def __init__(self, root_dir, device, image_size=224):
        self.device = device
        self.image_size = 224
        self.chunk_size = 32
        super().__init__(root_dir)
        self.add_loss_layer('MSEg', AdvGANMSELoss(self.model, ratio=0.1))

    def setup_model(self):
        return ImageChunkProcessor(self.chunk_size,
                                   ChunkEnhancer(32, latent_dims, self.image_size // self.chunk_size * 2,
                                                 self.chunk_size), self.image_size // self.chunk_size * 2, self.device).to(device)

    def setup_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=0.0001)

    def preprocess_training_input(self, batch_data):
        (x, style_input, latent_input), input = batch_data # x, style_input, latent_input
        return (x, style_input, latent_input), input


class VAEModule(AbstractAdvancedModule):
    def __init__(self, root_dir, discriminator, style_discriminator, enhancement_module=None):
        self.enhancement = enhancement_module
        super().__init__(root_dir)
        params = {
            'mse_ratio': 0.3,
            'perceptual_layers': ['2', '7'],
            'perceptual_scale': 100.0,
            'similarity_weight': 0.01,
            'kl_divergence_beta': 0.0001,
            'curve_weight': 2000,
            'secondary_mse_ratio': 0.085,
            'primary_gan_loss_ratio': 10.0,
            'style_gan_loss_ratio': 200.0,
        }
        pretrained_vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()
        for idx, layer in enumerate(pretrained_vgg):
            print(idx, layer)

        for param in pretrained_vgg.parameters():
            param.requires_grad = False  # Freeze the model
        pretrained_vgg.to(device)
        pretrained_vgg.eval()
        self.add_loss_layer('MSE', AdvMSELoss(self.model, params['mse_ratio'], params['secondary_mse_ratio']))
        self.add_loss_layer('Perceptual',
                            AdvPerceptualLoss(self.model, params['perceptual_layers'], params['perceptual_scale'],
                                              pretrained_vgg))
        self.add_loss_layer('Similarity',
                            AdvSimilarityLoss(self.model, params['similarity_weight'], params['kl_divergence_beta']))
        self.add_loss_layer('Curve', AdvCurveLoss(self.model, params['curve_weight']))
        self.add_loss_layer('GAN',
                            AdvGANLoss(self.model, params['primary_gan_loss_ratio'], params['style_gan_loss_ratio'],
                                       discriminator, style_discriminator))
        # self.input_channels = input_channels
        # self.hidden_dims = hidden_dims
        # self.latent_dims = latent_dims
        # self.device = device

    def setup_model(self):
        return VAE(input_channels=1, hidden_dims=16, latent_dims=latent_dims, enhancement=self.enhancement,
                   device=device).to(device)

    def setup_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=0.001)

    def preprocess_training_input(self, batch_data):
        images, char_embs, cap_idx, secondary_images, secondary_char_embs, secondary_idx, label, random_image = batch_data
        return (images, char_embs, secondary_char_embs, cap_idx, secondary_idx), (
        images, char_embs, cap_idx, secondary_images, secondary_char_embs, secondary_idx, label, random_image)


class DiscriminatorModule(AbstractAdvancedModule):
    def __init__(self, root_dir):
        super().__init__(root_dir)
        self.add_loss_layer('discriminator', AdvDiscriminatorLoss(self.model))

    def preprocess_training_input(self, batch_data):
        real_images, fake_images = batch_data
        # Combine real and fake images
        combined_images = torch.cat([real_images, fake_images], dim=0)

        # Create labels for real and fake images, with optional label smoothing for real labels
        real_labels = torch.full((real_images.size(0), 1), 0.9,
                                 device=real_images.device)  # Real labels slightly less than 1
        fake_labels = torch.zeros(fake_images.size(0), 1, device=fake_images.device)  # Fake labels
        combined_labels = torch.cat([real_labels, fake_labels], dim=0)

        # Shuffle the combined set
        indices = torch.randperm(combined_images.size(0))
        combined_images, combined_labels = combined_images[indices], combined_labels[indices]

        return combined_images, combined_labels

    def setup_model(self):
        return Discriminator().to(device)

    def setup_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=0.00001, weight_decay=0.0000001)


class StyleDiscriminatorModule(AbstractAdvancedModule):
    def __init__(self, root_dir, invert_probability=0.2):
        super().__init__(root_dir)
        self.add_loss_layer('style_discriminator', AdvStyleLoss(self.model))
        self.invert_probability = invert_probability

    def preprocess_training_input(self, batch_data):
        source_images, test_images, real_test_images = batch_data
        batch_size = test_images.size(0)

        test_indices = torch.randperm(test_images.size(0))[:batch_size // 2]
        real_test_indices = torch.randperm(real_test_images.size(0))[:batch_size // 2]
        selected_test_images = test_images[test_indices]
        selected_real_test_images = real_test_images[real_test_indices]

        # Randomly invert colors of some real test images
        invert_mask = torch.rand(batch_size // 2, device=real_test_images.device) < self.invert_probability
        inverted_real_test_images = 1.0 - selected_real_test_images
        selected_real_test_images[invert_mask] = inverted_real_test_images[invert_mask]

        # Combine selected test images and real test images
        combined_test_images = torch.cat([selected_test_images, selected_real_test_images], dim=0)

        # Create labels for selected test images and real test images
        test_labels = torch.zeros(batch_size // 2, 1, device=test_images.device)
        real_test_labels = torch.ones(batch_size // 2, 1, device=real_test_images.device)
        real_test_labels[invert_mask] = 0
        combined_labels = torch.cat([test_labels, real_test_labels], dim=0)

        # Shuffle the combined test images and labels
        indices = torch.randperm(combined_test_images.size(0))
        combined_test_images = combined_test_images[indices]
        combined_labels = combined_labels[indices]

        return (source_images, combined_test_images), combined_labels

    def setup_model(self):
        return StyleDiscriminator().to(device)

    def setup_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=0.00001, weight_decay=0.0000001)
