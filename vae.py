import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from performer_pytorch import Performer  # Assuming you're using lucidrains' implementation
from torch.nn import Sequential


def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAMBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, attention_kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(attention_kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.LeakyReLU(inplace=False, negative_slope=0.01),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ImprovedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=2):
        super(ImprovedResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=False)  # Using LeakyReLU
        self.se = SELayer(out_channels)
        self.cbam = CBAMBlock(out_channels)

        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))  # LeakyReLU activation
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.cbam(out)
        out = out + residual
        out = self.relu(out)  # LeakyReLU activation
        return out


class Encoder(nn.Module):
    def __init__(self, input_channels, hidden_dims, latent_dims, char_emb, char_emb_dim, dropout, cap_embedding,
                 cap_dim, character_fuser):
        super(Encoder, self).__init__()
        adjusted_input_channels = input_channels + char_emb_dim + cap_dim  # Adjust for character embedding
        self.char_emb = char_emb
        self.cap_embedding = cap_embedding
        self.character_fuser = character_fuser
        # Initial convolutional blocks
        self.resblock1 = ImprovedResidualBlock(adjusted_input_channels, hidden_dims, stride=2, dilation=1)
        self.resblock2 = ImprovedResidualBlock(hidden_dims, hidden_dims * 2, stride=2, dilation=1)
        self.resblock3 = ImprovedResidualBlock(hidden_dims * 2, hidden_dims * 4, stride=2, dilation=1)

        # Additional reduction block to reduce dimensionality before Performer
        self.reduction_conv = nn.Conv2d(hidden_dims * 4, hidden_dims * 4, kernel_size=3, stride=2, padding=1)

        self.reduction_pool = nn.AdaptiveAvgPool2d((6, 6))  # Adaptive pooling to a smaller fixed size
        self.character_dropout = nn.Dropout(dropout)

        # Performer for efficient attention
        self.performer = Performer(
            dim_head=32,
            dim=hidden_dims * 6 * 6 * 4,  # Adjusted dimension based on reduced feature map size
            depth=1,
            heads=8,
            causal=False,
            feature_redraw_interval=1000,
            generalized_attention=False,
            kernel_fn=torch.nn.LeakyReLU(inplace=False, negative_slope=0.01),
            use_scalenorm=False,
            use_rezero=True,
            shift_tokens=False,
            ff_mult=4,
        )

        # Output layers for the latent space representation
        self.fc_mu = nn.Linear(hidden_dims * 6 * 6 * 4, latent_dims)  # Adjusted for reduced dimensionality
        self.fc_var = nn.Linear(hidden_dims * 6 * 6 * 4, latent_dims)
        self.leaky_relu = nn.ReLU(inplace=False)

    def forward(self, x, char_idx, cap_idx):
        # Character embedding
        char_emb = self.char_emb(char_idx)
        cap_emb = self.cap_embedding(cap_idx)
        fused = self.character_fuser(torch.cat((char_emb, cap_emb), dim=1))
        # z = torch.cat((self.dropout3(z), self.dropout3(fused)), dim=1).unsqueeze(1)
        char_emb = self.character_dropout(char_emb)
        char_emb = char_emb.unsqueeze(-1).unsqueeze(-1)
        char_emb = char_emb.expand(-1, -1, x.size(2), x.size(3))
        # char_emb = char_emb.expand(-1, -1, x.size(2), x.size(3))

        # Capability embedding
        cap_emb = self.character_dropout(cap_emb)  # Assuming dropout is suitable for both embeddings
        cap_emb = cap_emb.unsqueeze(-1).unsqueeze(-1)
        cap_emb = cap_emb.expand(-1, -1, x.size(2), x.size(3))

        # fused = self.character_fuser(torch.cat((char_emb, fused), dim=1)).unsqueeze(1)

        # Concatenate character and capability embeddings with input tensor
        x = torch.cat((x, cap_emb, char_emb), dim=1)

        # Convolutional blocks
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)

        # Dimensionality reduction
        x = self.reduction_conv(x)
        x = self.reduction_pool(x)

        # Flatten for Performer
        x_flat = torch.flatten(x, start_dim=1)

        x_unsqueezed = x_flat.unsqueeze(1)
        # Performer for attention
        x_attended = self.performer(x_unsqueezed, return_embeddings=True).squeeze(1)

        # Activation before fully connected layers
        x = self.leaky_relu(x_attended)

        # Latent space representation
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return mu, log_var


# class Encoder(nn.Module):
#     def __init__(self, input_channels, hidden_dims, latent_dims, char_emb, char_emb_dim, dropout, cap_embedding, cap_dim):
#         super(Encoder, self).__init__()
#         adjusted_input_channels = input_channels + char_emb_dim + cap_dim # Adjust for character embedding
#         self.char_emb = char_emb
#         self.cap_embedding = cap_embedding
#
#         # Initial convolutional blocks
#         self.resblock1 = ImprovedResidualBlock(adjusted_input_channels, hidden_dims, stride=2, dilation=1)
#         self.resblock2 = ImprovedResidualBlock(hidden_dims, hidden_dims * 2, stride=2, dilation=1)
#
#         # Additional reduction block to reduce dimensionality before Performer
#         self.reduction_conv = nn.Conv2d(hidden_dims * 2, hidden_dims * 2, kernel_size=3, stride=2, padding=1)
#
#
#         self.reduction_pool = nn.AdaptiveAvgPool2d((7, 7))  # Adaptive pooling to a fixed size
#         self.character_dropout = nn.Dropout(dropout)
#
#         # Performer for efficient attention
#         self.performer = Performer(
#             dim_head=32,
#             dim=hidden_dims * 2 * 7 * 7,  # Adjusted dimension based on reduced feature map size
#             depth=1,
#             heads=8,
#             causal=False,
#             feature_redraw_interval=1000,
#             generalized_attention=False,
#             kernel_fn=torch.nn.LeakyReLU(inplace=False, negative_slope=0.01),
#             use_scalenorm=False,
#             use_rezero=True,
#             shift_tokens=True,
#             ff_mult=4,
#         )
#
#         # Output layers for the latent space representation
#         self.fc_mu = nn.Linear(hidden_dims * 2 * 7 * 7, latent_dims)  # Adjusted for reduced dimensionality
#         self.fc_var = nn.Linear(hidden_dims * 2 * 7 * 7, latent_dims)
#         self.leaky_relu = nn.ReLU(inplace=False)
#
#     def forward(self, x, char_idx, cap_idx):
#         # Character embedding
#         char_emb = self.char_emb(char_idx)
#         char_emb = self.character_dropout(char_emb)
#         char_emb = char_emb.unsqueeze(-1).unsqueeze(-1)
#         char_emb = char_emb.expand(-1, -1, x.size(2), x.size(3))
#
#         # Capability embedding
#         cap_emb = self.cap_embedding(cap_idx)
#         cap_emb = self.character_dropout(cap_emb)  # Assuming dropout is suitable for both embeddings
#         cap_emb = cap_emb.unsqueeze(-1).unsqueeze(-1)
#         cap_emb = cap_emb.expand(-1, -1, x.size(2), x.size(3))
#
#         # Concatenate character and capability embeddings with input tensor
#         x = torch.cat((x, char_emb, cap_emb), dim=1)
#
#         # Convolutional blocks
#         x = self.resblock1(x)
#         x = self.resblock2(x)
#
#         # Dimensionality reduction
#         x = self.reduction_conv(x)
#         x = self.reduction_pool(x)
#
#         # Flatten for Performer
#         x_flat = torch.flatten(x, start_dim=1)
#
#         x_unsqueezed = x_flat.unsqueeze(1)
#         # Performer for attention
#         x_attended = self.performer(x_unsqueezed, return_embeddings=True).squeeze(1)
#
#         # Activation before fully connected layers
#         x = self.leaky_relu(x_attended)
#
#         # Latent space representation
#         mu = self.fc_mu(x)
#         log_var = self.fc_var(x)
#
#         return mu, log_var
#

class LeakyReLUWithLearnable(nn.Module):
    def __init__(self, initial_slope=0.01):
        super(LeakyReLUWithLearnable, self).__init__()
        self.negative_slope = nn.Parameter(torch.tensor(initial_slope))

    def forward(self, x):
        return nn.functional.leaky_relu(x, negative_slope=self.negative_slope.item())


class AttentionModule(nn.Module):
    """Feature Attention Module with learnable negative slope"""

    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // 2, 1, bias=False)
        self.fc2 = nn.Conv2d(in_channels // 2, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.relu = LeakyReLUWithLearnable()

        # Initialize the learnable negative slope parameter

    def forward(self, x):
        y = self.global_pool(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y


class SpatialAttentionModule(nn.Module):
    """Improved Spatial Attention Module with BatchNorm and flexible kernel size."""

    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.batchnorm = nn.InstanceNorm2d(1)  # Add BatchNorm for stability
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        pool = torch.cat([max_pool, avg_pool], dim=1)
        y = self.conv1(pool)
        y = self.batchnorm(y)  # Apply BatchNorm
        y = self.sigmoid(y)
        return x * y


class EnhancedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate=1):
        super(EnhancedResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation_rate,
                               dilation=dilation_rate, bias=False)
        self.bn1 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation_rate,
                               dilation=dilation_rate, bias=False)
        self.bn2 = nn.InstanceNorm2d(out_channels)

        self.feature_attention = AttentionModule(out_channels)
        self.spatial_attention = SpatialAttentionModule()

        if in_channels != out_channels:
            self.residual_adjust = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels))
        else:
            self.residual_adjust = nn.Identity()

        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        residual = self.residual_adjust(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.feature_attention(out)
        out = self.spatial_attention(out)

        residual = self.residual_scale * residual
        out = self.relu(out + residual)

        return out


class DecoderRefinementModule(nn.Module):
    def __init__(self, in_channels, out_channels, residual_in_channels=0, residual_out_channels=0, dropout=0.3,
                 shuffle_scale=2,
                 kernel_size=3, stride=1, padding=1, pixel_shuffler=True, residual=True, conv_module=nn.ConvTranspose2d,
                 normalization_module=nn.BatchNorm2d, activation=nn.LeakyReLU):
        super(DecoderRefinementModule, self).__init__()
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self.convtrans = conv_module(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.pixel_shuffler = pixel_shuffler
        if self.pixel_shuffler:
            self.pixel_shuffle = nn.PixelShuffle(shuffle_scale)
        self.residual = residual
        if normalization_module is not None:
            self.bn = normalization_module(out_channels)
        else:
            self.bn = None
        if self.residual:
            self.residual_block = EnhancedResidualBlock(residual_in_channels, residual_out_channels)
        self.activation = activation()


    def forward(self, z, return_residual=False):
        z = self.convtrans(z)
        if self.bn is not None:
            z = self.bn(z)
        z = self.activation(z)
        if self.dropout is not None:
            z = self.dropout(z)

        if self.pixel_shuffler:
            z = self.pixel_shuffle(z)

        if self.residual:
            _z = self.residual_block(z)
            if return_residual:
                return z, _z
            else:
                return _z
        return z

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ImageChunkProcessor(nn.Module):
    def __init__(self, chunk_size, specialized_module, num_pos_encodings, device):
        super(ImageChunkProcessor, self).__init__()
        self.chunk_size = chunk_size
        self.specialized_module = specialized_module
        self.num_pos_encodings = num_pos_encodings

        # Generate positional encodings once in the constructor
        self.pos_encoding = self.generate_pos_encoding(256, 256).to(device)
        # self.pos_encoding = nn.Parameter(torch.randn(1, num_pos_encodings, chunk_size, chunk_size))

        # print('pit', self.pos_encoding.shape)

    def forward(self, x, style_input, latent_input):
        batch_size, _, image_size, _ = x.size()
        num_chunks = 224 // self.chunk_size

        # Break the image into chunks and add positional encoding to each chunk
        chunks = []
        for i in range(num_chunks):
            for j in range(num_chunks):
                chunk = x[:, :, i * self.chunk_size:(i + 1) * self.chunk_size,
                        j * self.chunk_size:(j + 1) * self.chunk_size]

                # Get the positional encoding for the current chunk
                pos_encoding = self.pos_encoding[i * num_chunks + j]
                pos_encoding = pos_encoding.unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(batch_size, 1,
                                                                                          self.chunk_size,
                                                                                          self.chunk_size)

                # Concatenate the chunk with its positional encoding
                chunk = torch.cat([chunk, pos_encoding], dim=1)
                chunks.append(chunk)

        # Stack the chunks and flatten them for processing
        chunks = torch.stack(chunks, dim=1)
        chunks = chunks.view(batch_size * num_chunks * num_chunks, -1, self.chunk_size, self.chunk_size)

        # Process the flattened chunks through the specialized module
        processed_chunks = self.specialized_module(chunks, style_input, latent_input)

        # Reshape the processed chunks back to the original shape
        processed_chunks = processed_chunks.view(batch_size, num_chunks, num_chunks, -1, self.chunk_size,
                                                 self.chunk_size)
        processed_chunks = processed_chunks.permute(0, 3, 1, 4, 2, 5).contiguous()
        processed_chunks = processed_chunks.view(batch_size, -1, image_size, image_size)

        return processed_chunks
    def generate_pos_encoding(self, num_pos_encodings, max_position):
        pos_encoding = torch.zeros(max_position, num_pos_encodings)
        position = torch.arange(0, max_position, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, num_pos_encodings, 2).float() * (-math.log(10000.0) / num_pos_encodings))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding

# class ImageChunkProcessor(nn.Module):
#     def __init__(self, chunk_size, specialized_module, num_pos_encodings, device):
#         super(ImageChunkProcessor, self).__init__()
#         self.chunk_size = chunk_size
#         self.specialized_module = specialized_module
#         self.num_pos_encodings = num_pos_encodings
#
#         # Generate positional encodings once in the constructor
#         self.pos_encoding = self.generate_pos_encoding(num_pos_encodings, chunk_size).to(device)
#         # self.pos_encoding = nn.Parameter(torch.randn(1, num_pos_encodings, chunk_size, chunk_size))
#
#         # print('pit', self.pos_encoding.shape)
#
#     def forward(self, x, style_input, latent_input):
#         batch_size, _, image_size, _ = x.size()
#         num_chunks = image_size // self.chunk_size
#
#         # Break the image into chunks and flatten them
#         chunks = []
#         for i in range(num_chunks):
#             for j in range(num_chunks):
#                 chunk = x[:, :, i * self.chunk_size:(i + 1) * self.chunk_size,
#                         j * self.chunk_size:(j + 1) * self.chunk_size]
#
#                 # Add positional encoding to each chunk
#                 pos_encoding = self.pos_encoding[i, j]
#                 # print(pos_encoding.shape, 'pos')
#                 # print(chunk.shape)
#                 pos_encoding = pos_encoding.unsqueeze(0).repeat(chunk.size(0), 1).unsqueeze(-1).unsqueeze(-1).repeat(1,1,self.chunk_size, self.chunk_size)
#                 # print(chunk.shape, pos_encoding.shape)
#                 # Add learned positional encoding to each chunk
#                 # pos_encoding = self.pos_encoding.repeat(chunk.size(0), 1, 1, 1)
#                 chunk = torch.cat([chunk, pos_encoding], dim=1)
#                 chunks.append(chunk)
#
#         chunks = torch.stack(chunks, dim=1)
#         chunks = chunks.view(batch_size, num_chunks, num_chunks, -1, self.chunk_size, self.chunk_size)
#
#         # Flatten the chunks for processing
#         chunks = chunks.view(batch_size * num_chunks * num_chunks, -1, self.chunk_size, self.chunk_size)
#
#         # Repeat the latent input for each chunk
#         # style_input = style_input.repeat(num_chunks * num_chunks, 1)
#         # latent_input = latent_input.repeat(num_chunks * num_chunks, 1)
#         # print(style_input.shape, latent_input.shape)
#
#         # Process the flattened chunks through the specialized module
#         processed_chunks = self.specialized_module(chunks, style_input, latent_input)
#
#         # Reshape the processed chunks back to the original shape
#         processed_chunks = processed_chunks.view(batch_size, num_chunks, num_chunks, -1, self.chunk_size,
#                                                  self.chunk_size)
#         processed_chunks = processed_chunks.permute(0, 3, 1, 4, 2, 5).contiguous()
#         processed_chunks = processed_chunks.view(batch_size, -1, image_size, image_size)
#
#         return processed_chunks
#
#     def generate_pos_encoding(self, num_pos_encodings, chunk_size):
#         # Generate sine and cosine positional encodings
#         pos_encoding = torch.zeros(chunk_size, chunk_size, num_pos_encodings)
#
#         for i in range(num_pos_encodings // 2):
#             freq = 2 ** i
#             for j in range(chunk_size):
#                 for k in range(chunk_size):
#                     pos_encoding[j, k, 2 * i] = torch.sin(torch.tensor(j / freq)) * 0.25
#                     pos_encoding[j, k, 2 * i + 1] = torch.cos(torch.tensor(k / freq)) * 0.25
#
#         return pos_encoding
#

class ChunkEnhancer(nn.Module):
    def __init__(self, style_dim, latent_dim, num_pos_encodings, chunk_size):
        super(ChunkEnhancer, self).__init__()
        self.latent_dim = latent_dim
        self.style_dim = style_dim
        self.chunk_size = chunk_size
        self.conv1 = nn.Conv2d(257, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.latent_fc = nn.Linear(latent_dim, 32)
        self.style_fc = nn.Linear(style_dim, 32)
        self.combined_fc = nn.Linear(64,self.chunk_size * self.chunk_size)

    def forward(self, chunk, style_input, latent_input):
        # Process the chunk through convolutional layers
        x = self.relu(self.conv1(chunk))
        x = self.relu(self.conv2(x))

        # Process the latent and style inputs
        latent_features = self.latent_fc(latent_input)
        style_features = self.style_fc(style_input)

        # Concatenate the latent and style features
        combined_features = torch.cat([latent_features, style_features], dim=1)

        # Process the combined features through a fully connected layer
        combined_features = self.relu(self.combined_fc(combined_features))

        # Reshape the combined features to match the shape of the flattened chunks
        batch_size, _, _, _ = x.size()
        # print(x.shape, combined_features.shape)
        combined_features = combined_features.view(1, -1, self.chunk_size, self.chunk_size)
        # Modulate the chunk features with the combined latent and style features
        x = self.relu(x * combined_features)

        # Process the modulated features through the final convolutional layer
        x = self.sigmoid(self.conv3(x))

        return x
class Decoder(nn.Module):
    def __init__(self, latent_dims, hidden_dims, output_channels, num_characters, char_emb, char_emb_dim, num_heads,
                 dropout, cap_embedding, cap_dim, character_fuser):
        super(Decoder, self).__init__()
        self.cap_embedding = cap_embedding
        self.character_fuser = character_fuser
        self.char_emb_dim = char_emb_dim
        self.cap_dim = cap_dim
        self.hidden_dims = hidden_dims
        self.total_emb_dim = latent_dims + char_emb_dim + cap_dim
        self.char_emb = char_emb
        self.prelinear_1 = nn.Linear(self.total_emb_dim + self.char_emb_dim, self.total_emb_dim)
        self.attention = nn.MultiheadAttention(embed_dim=self.total_emb_dim, num_heads=num_heads, batch_first=True)
        self.performer1 = Performer(
            dim_head=32,
            dim=self.total_emb_dim,
            depth=1,
            heads=num_heads,
            causal=False,
            feature_redraw_interval=1000,
            generalized_attention=False,
            kernel_fn=torch.nn.LeakyReLU(inplace=False),
            use_scalenorm=False,
            shift_tokens=True,
            use_rezero=True,
            ff_mult=4,
        )
        self.layer_norm = nn.LayerNorm(self.total_emb_dim)

        self.ffn = nn.Sequential(
            nn.Linear(self.total_emb_dim, self.total_emb_dim * 8),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(self.total_emb_dim * 8, self.total_emb_dim),
        )

        self.fc1 = nn.Linear(self.total_emb_dim, hidden_dims * 8 * 7 * 7)

        self.decoder_block = DecoderRefinementModule(hidden_dims * 8, hidden_dims * 8, hidden_dims * 2, hidden_dims * 2,
                                                     dropout=dropout)

        self.decoder_block_2 = DecoderRefinementModule(hidden_dims * 2, hidden_dims * 2, hidden_dims // 2,
                                                       hidden_dims // 2, dropout=dropout)

        self.refine_block_1 = DecoderRefinementModule(hidden_dims // 2, hidden_dims // 2, dropout=dropout / 2.0,
                                                      shuffle_scale=0,
                                                      kernel_size=4, stride=2, padding=1, pixel_shuffler=False,
                                                      residual=False)

        self.refine_block_2 = DecoderRefinementModule(hidden_dims // 2, hidden_dims // 4, dropout=dropout / 2.0,
                                                      kernel_size=4, stride=2, padding=1, pixel_shuffler=False,
                                                      residual=False)
        self.refine_block_3 = DecoderRefinementModule(hidden_dims // 4, hidden_dims // 4, dropout=dropout / 3.0,
                                                      kernel_size=4, stride=2, padding=1, pixel_shuffler=False,
                                                      residual=False)
        self.refine_unnormalized_1 = DecoderRefinementModule(hidden_dims // 4, hidden_dims // 4, dropout=dropout / 3.0,
                                                             kernel_size=3, stride=1, padding=1, pixel_shuffler=False,
                                                             residual=False, activation=nn.LeakyReLU,
                                                             normalization_module=None, conv_module=nn.Conv2d)
        self.refine_unnormalized_2 = DecoderRefinementModule(hidden_dims // 4, hidden_dims // 4, dropout=None,
                                                             kernel_size=3, stride=1, padding=1, pixel_shuffler=False,
                                                             residual=False, activation=nn.LeakyReLU,
                                                             normalization_module=None, conv_module=nn.Conv2d)

        self.fine_block_1 = DecoderRefinementModule(hidden_dims // 4, hidden_dims // 4, dropout=dropout / 3.0,
                                                    kernel_size=3, stride=1, padding=1, pixel_shuffler=False,
                                                    residual=False, activation=nn.LeakyReLU, conv_module=nn.Conv2d)
        self.fine_block_2 = DecoderRefinementModule(hidden_dims // 4, output_channels, dropout=None,
                                                    kernel_size=3, stride=1, padding=1, pixel_shuffler=False,
                                                    residual=False, activation=nn.LeakyReLU, conv_module=nn.Conv2d,
                                                    normalization_module=nn.InstanceNorm2d)

        self.fine_block_3 = DecoderRefinementModule(output_channels, output_channels, dropout=None, kernel_size=3,
                                                    stride=1, padding=1, pixel_shuffler=False, residual=False,
                                                    conv_module=nn.Conv2d, normalization_module=nn.InstanceNorm2d,
                                                    activation=nn.LeakyReLU)
        self.fine_block_4 = DecoderRefinementModule(output_channels, output_channels, dropout=None, kernel_size=3,
                                                    stride=1, padding=1, pixel_shuffler=False, residual=False,
                                                    conv_module=nn.Conv2d, normalization_module=nn.InstanceNorm2d,
                                                    activation=nn.LeakyReLU)

        self.final_conv = nn.Conv2d(output_channels, output_channels, kernel_size=1)

        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.controlled_brightness = nn.Linear(latent_dims, 1)
        self.style_linear = nn.Linear(char_emb_dim + latent_dims, char_emb_dim)

    def forward(self, z, char_idx, cap_idx):
        char_emb = self.char_emb(char_idx)
        og_char_emb = char_emb
        cap_embedding = self.cap_embedding(cap_idx)
        # brightness = F.sigmoid(self.controlled_brightness(self.dropout3(z)))
        # Ensure char_emb has three dimensions
        if char_emb.dim() == 2:
            char_emb = char_emb.unsqueeze(1)

        # Combine the character embedding with the latent parameter
        style_emb = z.unsqueeze(1).expand(-1, char_emb.size(1), -1)
        char_style_emb = torch.cat((char_emb, style_emb), dim=2)

        # Reshape char_style_emb before passing it through the style_linear layer
        batch_size, num_chars, emb_dim = char_style_emb.size()
        char_style_emb = char_style_emb.view(batch_size * num_chars, emb_dim)

        # Pass the combined character and style embedding through a linear layer
        char_style_emb = self.style_linear(char_style_emb)

        # Reshape char_style_emb back to the original shape
        char_style_emb = char_style_emb.view(batch_size, num_chars, -1)

        # Combine the character style embedding, latent parameter, and caption embedding
        z = torch.cat((self.dropout3(z), og_char_emb, cap_embedding, char_style_emb.mean(dim=1)), dim=1).unsqueeze(1)
        z = self.prelinear_1(z)
        attention_output, _ = self.attention(z, z, z)
        z = attention_output + z  # Apply skip connection around the attention mechanism
        z = z.squeeze(1)
        # print(z.shape)
        style_shape = z
        # Flatten for performer
        z_flat = torch.flatten(z, start_dim=1)
        # Apply Performer directly after attention
        z_performed1 = self.performer1(z_flat.unsqueeze(1), return_embeddings=True).squeeze(1)

        z_norm = self.layer_norm(z)  # Apply layer normalization
        z_ffn = self.dropout2(self.ffn(z_norm))  # Feed-forward network
        z = F.relu(z + z_ffn + z_performed1)  # Skip connection around the FFN

        # Now, leading up to fc1
        z = self.dropout2(self.fc1(z))

        z = z.view(-1, self.hidden_dims * 8, 7, 7)  # Ensure correct shape for convolutional layers

        z = self.decoder_block(z, return_residual=False)
        z = self.decoder_block_2(z, return_residual=False)
        z = self.refine_block_1(z)
        z = self.refine_block_2(z)
        z = self.refine_block_3(z)
        z = self.refine_unnormalized_1(z)
        z = self.refine_unnormalized_2(z)
        z = self.fine_block_1(z)
        # Reshape brightness_factors for broadcasting: [batch_size, 1, 1, 1]
        # brightness_factors = brightness.view(-1, 1, 1, 1)

        # Make sure brightness_factors can be broadcasted to the shape of z
        # brightness_factors_expanded = brightness_factors.expand_as(z)

        # Apply the different brightness factors to each item in the batch
        # and invert the brightness
        # inverted_z = 1.0 - (z * brightness_factors_expanded)

        z = self.fine_block_2(z)
        # print(z.shape)
        out_z = F.sigmoid(z)
        z = self.fine_block_3(z)
        z = self.fine_block_4(z)
        z = F.sigmoid(self.final_conv(z))

        return z, out_z, style_shape


class VAE(nn.Module):
    def __init__(self, input_channels=1, hidden_dims=64, latent_dims=128, num_characters=37, char_emb_dim=8,
                 cap_embed=8, chunk_size=32, device=None, post_processing=None, enhancement=None):
        super(VAE, self).__init__()
        self.char_emb = nn.Embedding(num_characters, char_emb_dim)
        self.cap_embedding = nn.Embedding(2, cap_embed)
        self.character_fuser = Sequential(
            nn.Linear(char_emb_dim + cap_embed, char_emb_dim + char_emb_dim + cap_embed + cap_embed),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.Linear(char_emb_dim + char_emb_dim + cap_embed + cap_embed, char_emb_dim + cap_embed)
        )
        self.encoder = Encoder(input_channels, hidden_dims, latent_dims, self.char_emb, char_emb_dim, dropout=0.25,
                               cap_dim=cap_embed, cap_embedding=self.cap_embedding,
                               character_fuser=self.character_fuser)
        self.decoder = Decoder(latent_dims, hidden_dims, input_channels, num_characters, self.char_emb, char_emb_dim,
                               num_heads=8, dropout=0.15, cap_dim=cap_embed, cap_embedding=self.cap_embedding,
                               character_fuser=self.character_fuser)
        size = latent_dims + char_emb_dim + cap_embed
        self.enhancement = enhancement
        # self.specialized_module = ChunkEnhancer(32, latent_dims, 224 // chunk_size * 2, chunk_size)
        # self.processor = ImageChunkProcessor(chunk_size, self.specialized_module, 224 // chunk_size * 2, device)
        # self.module_1 = nn.Sequential(
        #     ImageChunkProcessor(16, ChunkEnhancer(40, latent_dims, 224 // 16 * 2, 16), 224 // 16 * 2, device)
        # )
        # self.specialized_module2 = ChunkEnhancer(40, latent_dims, 224 // 32 * 2, 32)
        # self.processor2 = ImageChunkProcessor(32, self.specialized_module2, 224 // 32 * 2, device)
    def initialize_weights(self):
        for m in self.modules():
            initialize_weights(m)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)  # Standard deviation
        eps = torch.randn_like(std)  # Sample epsilon from a standard normal distribution
        return mu + eps * std

    def forward(self, x, labels, target_labels, input_cap, target_cap):
        mu, log_var = self.encoder(x, labels, input_cap)
        z = self.reparameterize(mu, log_var)
        refined, unrefined, style = self.decoder(z, target_labels, target_cap)  # , mu2, log_var2
        img = refined
        max_len = 1
        for i in range(0, max_len):
            img = self.enhancement.model(img, style, z)
        return img, mu, log_var, unrefined, z, style

        # return self.decoder(z, target_labels), mu, log_var, mu2, log_var2
