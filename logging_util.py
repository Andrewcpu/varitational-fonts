import numpy as np
import torch
from matplotlib import pyplot as plt


def log_loss(batch_idx, center_loss, disc_loss, epoch, gan_loss, img_len, kl_div, perceptual_loss_val, prefix,
             recon_loss, sim_loss, vae_loss_calc, enc_loss, e_loss, c_loss, style_loss, e_style_loss):
    print(
        f'Epoch {epoch}, Batch {batch_idx}:  GEN {(vae_loss_calc.item() / img_len):.5f}  /  GAN: {(gan_loss.item() / img_len):.5f}  /  E-STYLE: {e_style_loss.item() / img_len:.5f}')
    print(' ' * len(prefix),
          f'Recon: {(recon_loss.item() / img_len):.5f},  Perceptual: {(perceptual_loss_val.item() / img_len):.5f}')
    print(' ' * len(prefix),
          f'Style: {(style_loss.item() / img_len):.5f},  KL-Diverge: {(kl_div.item() / img_len):.5f}')
    print(' ' * len(prefix), f'Disc: {(disc_loss.item()):.5f}')
    print(' ' * len(prefix), f'Edge: {(e_loss / img_len):.5f}, Curve: {(c_loss / img_len):.5f}')
    print("")

def display_sample_image(ax, title, image):
    ax.imshow(image, cmap='gray')
    ax.set_title(title)
    ax.axis('off')  # Hides the axis
    ax.set_xticks([])  # Hides x-axis ticks
    ax.set_yticks([])  # Hides y-axis ticks


def display_chart(batch_idx, batch_indices, center_loss, disc_loss, e_loss, c_loss, enc_adv_loss, epoch, gan_loss,
                  images,
                  kl_div, label, loss_values, perceptual_loss_values, gan_loss_values, estyle_loss, n_rows, open,
                  perceptual_loss_val, recon_batch, recon_loss,
                  secondary_images, sim_loss, vae_loss_calc, style_loss, e_style_loss):
    img_len = len(images)
    prefix = f'Epoch {epoch}, Batch {batch_idx}: '
    log_loss(batch_idx, center_loss, disc_loss, epoch, gan_loss, img_len, kl_div, perceptual_loss_val, prefix,
             recon_loss, sim_loss, vae_loss_calc, enc_adv_loss, e_loss, c_loss, style_loss, e_style_loss)
    fig = plt.figure(figsize=(11, n_rows + 1), dpi=120)
    gs = plt.GridSpec(n_rows + 1, 7, figure=fig)
    with torch.no_grad():
        # Loop through the first n_rows of samples to display
        for i in range(n_rows):
            idx = i  # Sequential; use np.random.randint(0, images.size(0)) for random

            # Extract images for the current index
            input_image = images[idx].cpu().numpy().squeeze()
            secondary_image = secondary_images[idx].cpu().numpy().squeeze()
            recon_image = recon_batch[idx].cpu().numpy().squeeze()
            current_label = label[idx]
            # Plot Original Image
            ax1 = fig.add_subplot(gs[i, 0])  # Adds a subplot for the original image in the grid
            display_sample_image(ax1, f'({current_label})', input_image)

            ax2 = fig.add_subplot(gs[i, 1])  # Adds a subplot for the secondary image
            display_sample_image(ax2, '', secondary_image)

            ax3 = fig.add_subplot(gs[i, 2])  # Adds a subplot for the reconstructed image
            display_sample_image(ax3, '', recon_image)

            sidx = idx + 8
            if sidx >= len(images):
                continue
            # Extract images for the current index
            input_image = images[sidx].cpu().numpy().squeeze()
            secondary_image = secondary_images[sidx].cpu().numpy().squeeze()
            recon_image = recon_batch[sidx].cpu().numpy().squeeze()
            current_label = label[sidx]
            # Plot Original Image
            # ax0 = fig.add_subplot(gs[i,3 ])

            ax1 = fig.add_subplot(gs[i, 4])  # Adds a subplot for the original image in the grid
            display_sample_image(ax1, f'({current_label})', input_image)

            ax2 = fig.add_subplot(gs[i, 5])  # Adds a subplot for the secondary image
            display_sample_image(ax2, '', secondary_image)

            ax3 = fig.add_subplot(gs[i, 6])  # Adds a subplot for the reconstructed image
            display_sample_image(ax3, '', recon_image)

        loss_ax = fig.add_subplot(gs[n_rows, :])
        # Assuming axes is a 2D array of subplot objects and the last row is reserved for the loss plot
        if len(loss_values) > 1:
            # Plot the first loss graph on the primary y-axis
            plot_loss_graph(loss_ax, batch_indices, loss_values, 'Loss 1', 'blue', '-')
            loss_ax.set_xlabel('Batch')
            loss_ax.set_ylabel('Loss 1')
            loss_ax.tick_params(axis='y', labelcolor='blue')

            # Create a second y-axis and plot the second loss graph
            ax2t = loss_ax.twinx()
            plot_loss_graph(ax2t, batch_indices, perceptual_loss_values, 'Perceptual', 'green', '-')
            ax2t.set_ylabel('Perceptual')
            ax2t.tick_params(axis='y', labelcolor='green')

            # Create a third y-axis and plot the third loss graph
            ax3t = ax2t.twinx()
            plot_loss_graph(ax3t, batch_indices, gan_loss_values, 'GAN', 'red', '-')
            ax3t.set_ylabel('GAN')
            ax3t.tick_params(axis='y', labelcolor='red')
            # Create a third y-axis and plot the third loss graph
            ax4t = ax2t.twinx()
            plot_loss_graph(ax4t, batch_indices, estyle_loss, 'EStyle', 'pink', '-')
            ax4t.set_ylabel('EStyle')
            ax4t.tick_params(axis='y', labelcolor='pink')

            # plot_loss_graph(loss_ax, batch_indices, loss_values)

        plt.tight_layout(pad=0, w_pad=0, h_pad=0)
        plt.subplots_adjust(hspace=0)  # Adjust spacing as needed
        fig.savefig(f'figures/generation_{epoch}_batch_{batch_idx}.png', dpi=120, bbox_inches='tight')
        plt.close(fig)
        plt.close('all')


def plot_loss_graph(loss_ax, batch_indices, loss_values, label, color, linestyle):
    loss_values_np = np.array(loss_values)
    batch_indices_np = np.array(batch_indices)

    segment_length = 50
    num_segments = len(loss_values_np) // segment_length
    remainder = len(loss_values_np) % segment_length

    averages = []
    averaged_indices = []

    # Compute averages for complete segments
    for i in range(num_segments):
        start_index = i * segment_length
        end_index = start_index + segment_length
        segment_average = np.mean(loss_values_np[start_index:end_index])
        averages.append(segment_average)
        # Use the start index of each segment for plotting
        averaged_indices.append(batch_indices_np[start_index])

    # Handle remainder if it exists
    if remainder > 0:
        remainder_start_index = num_segments * segment_length
        remainder_average = np.mean(loss_values_np[remainder_start_index:])
        averages.append(remainder_average)
        # Use the start index of the remainder for plotting
        averaged_indices.append(batch_indices_np[remainder_start_index])

    # Ensure both lists are numpy arrays for plotting
    averaged_indices_np = np.array(averaged_indices)
    averages_np = np.array(averages)

    # Plotting
    loss_ax.plot(averaged_indices_np, averages_np, label=label, color=color, linestyle=linestyle)
