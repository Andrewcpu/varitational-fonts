import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.models import vgg16, VGG16_Weights

from advanced_dataset import AdvancedImageDataset
from logging_util import display_sample_image, plot_loss_graph
from loss_utils import add_noise
from module_implementations import VAEModule, StyleDiscriminatorModule, DiscriminatorModule, EnhancementModule

batch_indices = []

# model = VAEModule('J:/model_checkpoints/vae')
def display_chart(epoch, batch_idx, images, secondary_images, recon_batch, labels, loss_lists, n_rows=8):
    img_len = len(images)
    prefix = f'Epoch {epoch}, Batch {batch_idx}: '

    fig = plt.figure(figsize=(11, n_rows + 1), dpi=120)
    gs = plt.GridSpec(n_rows + 1, 7, figure=fig)

    with torch.no_grad():
        for i in range(n_rows):
            for j in range(2):
                idx = i + j * 8
                if idx >= len(images):
                    continue

                input_image = images[idx].cpu().numpy().squeeze()
                secondary_image = secondary_images[idx].cpu().numpy().squeeze()
                recon_image = recon_batch[idx].cpu().numpy().squeeze()
                current_label = labels[idx]

                ax1 = fig.add_subplot(gs[i, j * 3])
                display_sample_image(ax1, f'({current_label})', input_image)

                ax2 = fig.add_subplot(gs[i, j * 3 + 1])
                display_sample_image(ax2, '', secondary_image)

                ax3 = fig.add_subplot(gs[i, j * 3 + 2])
                display_sample_image(ax3, '', recon_image)

        loss_ax = fig.add_subplot(gs[n_rows, :])
        plot_multiple_losses(loss_ax, loss_lists)

    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.subplots_adjust(hspace=0)
    fig.savefig(f'figures/generation_{epoch}_batch_{batch_idx}.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    plt.close('all')


def plot_multiple_losses(ax, loss_lists):
    colors = ['blue', 'green', 'red', 'pink', 'cyan', 'magenta', 'yellow', 'brown']
    batch_indices_np = np.array(batch_indices)

    for i, (loss_name, loss_values) in enumerate(loss_lists):
        loss_values_np = np.array(loss_values)

        # Calculate the number of points to average
        num_points = len(loss_values)
        avg_points = min(num_points, 50)

        # Average every avg_points batches
        num_averaged_points = num_points // avg_points
        averaged_indices = batch_indices_np[:num_averaged_points * avg_points:avg_points]
        averaged_values = np.mean(loss_values_np[:num_averaged_points * avg_points].reshape(-1, avg_points), axis=1)

        # Normalize the averaged values between 0 and 1
        min_val = np.min(averaged_values)
        max_val = np.max(averaged_values)
        normalized_values = (averaged_values - min_val) / (max_val - min_val)

        ax.plot(averaged_indices, normalized_values, label=loss_name, color=colors[i], linestyle='-')

        # Add labels at the highest and lowest points of each line with actual loss values
        max_idx = np.argmax(averaged_values)
        min_idx = np.argmin(averaged_values)
        ax.annotate(f'{loss_name} Max: {averaged_values[max_idx]:.4f}',
                    xy=(averaged_indices[max_idx], normalized_values[max_idx]),
                    xytext=(5, 0), textcoords='offset points', color=colors[i], fontsize='small')
        ax.annotate(f'{loss_name} Min: {averaged_values[min_idx]:.4f}',
                    xy=(averaged_indices[min_idx], normalized_values[min_idx]),
                    xytext=(5, 0), textcoords='offset points', color=colors[i], fontsize='small')

    ax.set_xlabel('Batch')
    ax.set_ylabel('Normalized Loss')

    # Create a custom legend with colored labels
    legend_handles = [plt.Line2D([0], [0], color=colors[i], lw=2, label=loss_name)
                      for i, (loss_name, _) in enumerate(loss_lists)]
    legend_labels = [loss_name for loss_name, _ in loss_lists]

    # Position the legend on the top left corner of the chart
    ax.legend(handles=legend_handles, labels=legend_labels, loc='upper left',
              bbox_to_anchor=(0, 1), ncol=2, fontsize='small', frameon=False)

    # Set the y-limits between 0 and 1
    ax.set_ylim(bottom=0, top=1)

pretrained_vgg = None
epoch = 0
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # selected_layers = ['12', '14', '19', '21']  # Example layer indices for VGG16
    # selected_layers = ['2', '7', '12', '14', '19', '21', '26', '28']  # Example layer indices for VGG16
    # selected_layers = ['2', '7', '12', '14']  # Example layer indices for VGG16
    # selected_layers = ['2', '7', '21']  # Example layer indices for VGG16
    # font_dataset = AdvancedImageDataset(directories=["./new_dataset/", "./google_fonts/"])
    font_dataset = AdvancedImageDataset(directories=["./new_dataset/"])
    batch_size = 16
    style_module = StyleDiscriminatorModule('J:/model_checkpoints/style/')
    discriminator_module = DiscriminatorModule('J:/model_checkpoints/disc/')
    enhancement = EnhancementModule('J:/model_checkpoints/enhancement/', device=device)
    vae_module = VAEModule('J:/model_checkpoints/vae/', discriminator=discriminator_module, style_discriminator=style_module, enhancement_module=enhancement)
    # vae_module.load_state()
    # discriminator_module.load_state()
    # style_module.load_state()
    # enhancement.load_state()
    dataloader = DataLoader(font_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    latent_dims = 16
    for batch_idx, (images, char_embs, cap_idx, secondary_images,
                    secondary_char_embs, secondary_idx, label, random_image) in enumerate(dataloader):
        images, char_embs, secondary_images, secondary_char_embs, cap_idx, secondary_idx, random_image = images.to(
            device), char_embs.to(device), secondary_images.to(device), secondary_char_embs.to(
            device), cap_idx.to(device), secondary_idx.to(device), random_image.to(device)

        # x, labels, target_labels, input_cap, target_cap
        vae_loss, (recon_batch, mu, logvar, stylized_result, actual_z, style_input) = vae_module.step((images, char_embs, cap_idx, secondary_images, secondary_char_embs, secondary_idx, label, random_image))
        # r(style_discriminator, images, random_image, secondary_images,
        #   style_discriminator_optimizer)

        # detached_recon = recon_batch.detach()
        # detached_recon_new = torch.autograd.Variable(detached_recon, requires_grad=True)

        detached_style = style_input.detach()
        detached_style_new = torch.autograd.Variable(detached_style, requires_grad=True)

        detached_latent = actual_z.detach()
        detached_latent_new = torch.autograd.Variable(detached_latent, requires_grad=True)

        detached_new_recon = add_noise(secondary_images, 0.2)

        enhancement.step(((detached_new_recon, detached_style_new, detached_latent_new), (images, char_embs, cap_idx, secondary_images,
                    secondary_char_embs, secondary_idx, label, random_image)))
        style_loss, style_output = style_module.step((images, random_image, secondary_images))
        # disc_loss = train_discriminator(real_discriminator, secondary_images, recon_batch, real_discriminator_optimizer)
        # Detach the value from the graph
        detached_value = recon_batch.detach()
        # Wrap the detached value in a new Variable with requires_grad=True
        new_value = torch.autograd.Variable(detached_value, requires_grad=True)
        batch_indices.append(batch_idx + epoch * len(dataloader))
        discriminator_loss, discriminator_output = discriminator_module.step((secondary_images, new_value), expand=False)

        loss_lists = []
        for module in [vae_module, style_module, discriminator_module, enhancement]:
            for key, value in module.loss_history.items():
                loss_lists.append((key, value))

        if batch_idx % 100 == 0:
            vae_module.log_status()
            style_module.log_status()
            discriminator_module.log_status()
            enhancement.log_status()
            display_chart(0, batch_idx, images, recon_batch, stylized_result, label, loss_lists,)
        if batch_idx % 1000 == 0:
            vae_module.save_state()
            style_module.save_state()
            discriminator_module.save_state()
            enhancement.save_state()


        # Style: test_images, real_test_images
        # Discriminator: real_images, fake_images
        # VAE: Batch
