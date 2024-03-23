import torch
import torch.nn as nn
import torch.nn.functional as F

from vae import LeakyReLUWithLearnable


class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        return out + x  # Skip Connection

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Replacing standard convolutions with depthwise separable convolutions
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, groups=1),
            # Initial conv, not depthwise, groups=1 is standard convolution
            nn.Conv2d(64, 64, kernel_size=1, stride=1),  # Pointwise
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, groups=64),  # Depthwise
            nn.Conv2d(64, 128, kernel_size=1, stride=1),  # Pointwise
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=False)
        )
        # Adding attention before the final convolutional layer
        self.attention = SelfAttention(256)
        # Using dilated convolution to increase receptive field
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=2, dilation=2)  # Dilated
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(576 * 8 * 8, 1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.02, inplace=False)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.02, inplace=False)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.02, inplace=False)
        x = self.attention(x)  # Applying attention
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.02, inplace=False)
        x = F.leaky_relu(self.conv5(x), 0.2, inplace=False)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


class StyleDiscriminator(nn.Module):
    def __init__(self):
        super(StyleDiscriminator, self).__init__()

        # Shared layers for both source and test images
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, groups=1),
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, groups=64),
            nn.Conv2d(64, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=2, dilation=2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1)

        self.bn1 = nn.InstanceNorm2d(64)
        self.bn2 = nn.InstanceNorm2d(128)
        self.bn3 = nn.InstanceNorm2d(256)
        self.bn4 = nn.InstanceNorm2d(512)

        self.leaky_learn_1 = LeakyReLUWithLearnable()
        self.leaky_learn_2 = LeakyReLUWithLearnable()

        # Improved attention mechanism
        self.attention1 = SelfAttention(128)
        self.attention2 = SelfAttention(256)
        self.attention3 = SelfAttention(512)

        # Separate fully connected layers for source and test images
        self.fc_source = nn.Linear(576 * 8 * 8, 256)
        self.fc_test = nn.Linear(576 * 8 * 8, 256)

        # Final fully connected layer for style matching prediction
        self.fc_final = nn.Linear(512, 1)

    def forward(self, source_images, test_images):
        # Process source images
        source = F.leaky_relu(self.bn1(self.conv1(source_images)), 0.2)
        source = F.leaky_relu(self.bn2(self.conv2(source)), 0.02)
        source = self.attention1(source)
        source = F.leaky_relu(self.bn3(self.conv3(source)), 0.02)
        source = self.attention2(source)
        source = self.leaky_learn_1(self.bn4(self.conv4(source)))
        source = self.attention3(source)
        source = self.leaky_learn_2(self.conv5(source))
        source = torch.flatten(source, start_dim=1)
        source = self.fc_source(source)

        # Process test images
        test = self.leaky_learn_1(self.bn1(self.conv1(test_images)))
        test = self.leaky_learn_1(self.bn2(self.conv2(test)))
        test = self.attention1(test)
        test = self.leaky_learn_2(self.bn3(self.conv3(test)))
        test = self.attention2(test)
        test = self.leaky_learn_1(self.bn4(self.conv4(test)))
        test = self.attention3(test)
        test = self.leaky_learn_2(self.conv5(test))
        test = torch.flatten(test, start_dim=1)
        test = self.fc_test(test)

        # Concatenate source and test features
        combined = torch.cat((source, test), dim=1)

        # Final prediction
        output = self.fc_final(combined)

        return output

# class StyleDiscriminator(nn.Module):
#     def __init__(self):
#         super(StyleDiscriminator, self).__init__()
#         # Shared layers for both source and test images
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, groups=1),
#             nn.Conv2d(64, 64, kernel_size=1, stride=1),
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, groups=64),
#             nn.Conv2d(64, 128, kernel_size=1, stride=1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
#         self.attention = SelfAttention(256)
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=2, dilation=2)
#         self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
#         self.conv5 = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1)
#         self.bn1 = nn.InstanceNorm2d(64)
#         self.bn2 = nn.InstanceNorm2d(128)
#         self.bn3 = nn.InstanceNorm2d(256)
#         self.bn4 = nn.InstanceNorm2d(512)
#         self.leaky_learn_1 = LeakyReLUWithLearnable()
#         self.leaky_learn_2 = LeakyReLUWithLearnable()
#         # Separate fully connected layers for source and test images
#         self.fc_source = nn.Linear(576 * 8 * 8, 256)
#         self.fc_test = nn.Linear(576 * 8 * 8, 256)
#
#         # Final fully connected layer for style matching prediction
#         self.fc_final = nn.Linear(512, 1)
#
#     def forward(self, source_images, test_images):
#         # print(source_images.shape, test_images.shape)
#         # Process source images
#         source = F.leaky_relu(self.bn1(self.conv1(source_images)), 0.2)
#         source = F.leaky_relu(self.bn2(self.conv2(source)), 0.02)
#         source = F.leaky_relu(self.bn3(self.conv3(source)), 0.02)
#         source = self.attention(source)
#         source = self.leaky_learn_1(self.bn4(self.conv4(source)))
#         source = self.leaky_learn_2(self.conv5(source))
#         source = torch.flatten(source, start_dim=1)
#         source = self.fc_source(source)
#
#         # Process test images
#         test = self.leaky_learn_1(self.bn1(self.conv1(test_images)))
#         test = self.leaky_learn_1(self.bn2(self.conv2(test)))
#         test = self.leaky_learn_2(self.bn3(self.conv3(test)))
#         test = self.attention(test)
#         test = self.leaky_learn_1(self.bn4(self.conv4(test)))
#         test = self.leaky_learn_2(self.conv5(test))
#         test = torch.flatten(test, start_dim=1)
#         test = self.fc_test(test)
#
#         # Concatenate source and test features
#         combined = torch.cat((source, test), dim=1)
#
#         # Final prediction
#         output = self.fc_final(combined)
#         return output

class CharacterDiscriminator(nn.Module):
    def __init__(self, latent_dims, num_characters=37):
        super(CharacterDiscriminator, self).__init__()
        self.fc1 = nn.Linear(latent_dims, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 64)  # Second fully connected layer
        self.fc3 = nn.Linear(64, num_characters)  # Output layer

        # Optional: Dropout layer for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Activation function for first layer
        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.fc2(x))  # Activation function for second layer
        x = self.dropout(x)  # Apply dropout
        x = self.fc3(x)  # No activation function here, raw logits are often used for the final layer in classification
        return x
