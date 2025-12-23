import argparse
import os
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from torch.nn.init import kaiming_normal_, xavier_normal_, constant_

# Assuming diva.model_loss contains MMDLoss and HSICLoss and other necessary components
# If GradientReversalLayer is also in diva.model_loss, the local definition below might override it or be redundant.
from diva.model_loss import *  # Wildcard import, as in original code


# from torch.autograd import Function # Function is already imported via torch.autograd.Function

# ==============================================================================
# Weight Initialization
# ==============================================================================

def initialize_weights(model):
    """
    Initializes the weights and biases of the model.
    - Applies Xavier/He initialization for linear and convolutional layers.
    - Applies standard initialization for BatchNorm layers.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            # He initialization for convolutional layer weights
            kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            # Xavier initialization for fully connected layer weights
            xavier_normal_(m.weight)
            if m.bias is not None:
                constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            # Initialize BatchNorm layer weights to 1 and biases to 0
            constant_(m.weight, 1)
            constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            # Initialize LayerNorm layer weights to 1 and biases to 0
            constant_(m.weight, 1)
            constant_(m.bias, 0)


# ==============================================================================
# Modality Specific Encoders
# ==============================================================================

class VideoEncoder(nn.Module):
    def __init__(self, feature_dim):
        super(VideoEncoder, self).__init__()
        self.feature_dim = feature_dim

        # Defining 3D Convolution + BatchNorm + ReLU blocks
        self.conv1 = self._conv_block(3, 16)
        self.conv2 = self._conv_block(16, 32)
        self.conv3 = self._conv_block(32, 64)
        self.conv4 = self._conv_block(64, 128)

        # Fully connected layer for output, calculated based on conv output dimensions
        # Example: Input (B, 3, 32, 224, 224) -> (B, C, T, H, W) for Conv3D
        # After transpose: (B, 3, 32, 224, 224)
        # Conv1 output: (B, 16, 16, 112, 112)
        # Conv2 output: (B, 32, 8, 56, 56)
        # Conv3 output: (B, 64, 4, 28, 28)
        # Conv4 output: (B, 128, 2, 14, 14)
        self.fc_resnet_output = nn.Linear(128 * 2 * 14 * 14, self.feature_dim)

    def _conv_block(self, in_channels, out_channels):
        """Helper function to define a Conv3d + BatchNorm + ReLU block."""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Input x is expected to be (Batch, Time, Channel, Height, Width)
        # Conv3D expects (Batch, Channel, Time, Height, Width)
        x = x.transpose(1, 2)  # Swap Channel and Time dimensions

        # Apply convolution blocks
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Flatten and pass through the fully connected layer
        x = x.view(x.size(0), -1)  # Flatten the tensor
        feature = self.fc_resnet_output(x)

        return feature


class AudioEncoder(nn.Module):
    def __init__(self, feature_dim):
        super(AudioEncoder, self).__init__()
        self.feature_dim = feature_dim

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Assuming input audio spectrogram (e.g., [B, 1, 1024, 128]) (Channels, Freq, Time)
        # After 4 conv layers with stride (2,2):
        # Freq: 1024 -> 512 -> 256 -> 128 -> 64
        # Time: 128 -> 64 -> 32 -> 16 -> 8
        # Output feature map size: [batch_size, 128, 64, 8]
        self.fc_resnet_output = nn.Linear(128 * 64 * 8, self.feature_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the output: [batch_size, features]
        feature = self.fc_resnet_output(x)
        return feature


# ==============================================================================
# Shared and Private Encoders
# ==============================================================================

class SharedEncoder(nn.Module):
    """Encodes features from different modalities into a common shared space."""

    def __init__(self, input_dim, shared_dim):
        super(SharedEncoder, self).__init__()

        self.shared_fc = nn.Sequential(
            nn.Linear(input_dim, shared_dim),  # Map to shared space
            nn.ReLU(inplace=True),
            nn.LayerNorm(shared_dim)  # Normalize features in shared space
        )

    def forward(self, video_feat, audio_feat):
        # Map video and audio features to the shared space independently
        video_shared = self.shared_fc(video_feat)
        audio_shared = self.shared_fc(audio_feat)
        return video_shared, audio_shared


class VideoPrivateEncoder(nn.Module):
    """Encodes video features into a private space, specific to video modality."""

    def __init__(self, input_dim, private_dim):  # Renamed shared_dim to private_dim for clarity
        super(VideoPrivateEncoder, self).__init__()

        self.private_fc = nn.Sequential(
            nn.Linear(input_dim, private_dim),  # Map to private space
            nn.ReLU(inplace=True),
            nn.LayerNorm(private_dim)  # Normalize features in private space
        )

    def forward(self, x):
        return self.private_fc(x)


class AudioPrivateEncoder(nn.Module):
    """Encodes audio features into a private space, specific to audio modality."""

    def __init__(self, input_dim, private_dim):  # Renamed shared_dim to private_dim for clarity
        super(AudioPrivateEncoder, self).__init__()

        self.private_fc = nn.Sequential(
            nn.Linear(input_dim, private_dim),  # Map to private space
            nn.ReLU(inplace=True),
            nn.LayerNorm(private_dim)  # Normalize features in private space
        )

    def forward(self, x):
        return self.private_fc(x)


# ==============================================================================
# Adversarial Components
# ==============================================================================

class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL).
    Forward pass: identity function.
    Backward pass: reverses the gradient (multiplies by -1).
    """

    @staticmethod
    def forward(ctx, x):
        # ctx.save_for_backward(x) # Save input tensor for backward pass (optional if not needed by backward)
        return x.clone()  # Return a new cloned tensor to ensure original tensor is not modified

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output  # Reverse the gradient direction


class Share_Discriminator(nn.Module):
    """Discriminator for shared features to distinguish modalities (adversarial training)."""

    def __init__(self, input_dim, num_modalities=2):
        super(Share_Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_modalities)  # Output modality classes

    def forward(self, shared_v, shared_a):
        # Concatenate features before passing to discriminator if it processes combined features,
        # or process them individually if it's meant to classify each.
        # Original code concatenates.
        x = torch.cat([shared_v, shared_a], dim=0)  # Concatenates along batch dimension. Check if this is intended.
        # If intended to process features per sample, dim=-1 or a different structure is needed.
        # Assuming it's trying to classify if a feature (v or a) came from modality 0 or 1.
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Logits for modality classification
        return x


class Private_Discriminator(nn.Module):
    """Discriminator for private features to distinguish modalities (adversarial training)."""

    def __init__(self, input_dim, num_modalities=2):
        super(Private_Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_modalities)  # Output modality classes

    def forward(self, private_v, private_a):
        # Similar concatenation as Share_Discriminator
        x = torch.cat([private_v, private_a], dim=0)  # Concatenates along batch dimension.
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Logits for modality classification
        return x


# ==============================================================================
# Attention and Interaction Mechanisms
# ==============================================================================

class CrossAttention(nn.Module):
    """Simple Cross-Attention mechanism."""

    def __init__(self, Q_dim, K_V_dim, d_k):  # K_V_dim for key and value, d_k is hidden dim for Q,K,V
        super(CrossAttention, self).__init__()
        self.query_transform = nn.Linear(Q_dim, d_k)
        self.key_transform = nn.Linear(K_V_dim, d_k)
        self.value_transform = nn.Linear(K_V_dim, d_k)
        self.scale = d_k ** 0.5  # For scaled dot-product attention

    def forward(self, Q_input, K_input, V_input):
        # Transform Query, Key, Value
        Q = self.query_transform(Q_input)  # [batch_size, d_k]
        K = self.key_transform(K_input)  # [batch_size, d_k]
        V = self.value_transform(V_input)  # [batch_size, d_k]

        # Compute attention scores (Q * K.T / sqrt(d_k))
        # Assuming Q, K, V are [batch_size, feature_dim]
        # For typical cross-attention with sequences, dimensions would be [batch_size, seq_len, feature_dim]
        # Here, it seems to be sample-wise attention if batch_size > 1.
        attention_scores = torch.matmul(Q, K.transpose(-2,
                                                       -1)) / self.scale  # [batch_size, batch_size] if K is [batch_size, d_k]
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Compute weighted sum of Values
        attention_output = torch.matmul(attention_weights, V)  # [batch_size, d_k]
        return attention_output


class AdaptiveAttentionMechanism(nn.Module):
    """Computes adaptive weights for a list of input tensors and returns their weighted sum."""

    def __init__(self, input_dim):
        super(AdaptiveAttentionMechanism, self).__init__()
        self.weight_layer = nn.Linear(input_dim, 1)  # Learns a score for each input source

    def forward(self, list_of_inputs):
        """
        Args:
            list_of_inputs: List of tensors [Y1, Y2, ...], each with shape [batch_size, input_dim].
        Returns:
            weighted_sum: Tensor of shape [batch_size, input_dim].
        """
        # Stack inputs along a new dimension to process them together
        stacked_inputs = torch.stack(list_of_inputs, dim=1)  # [batch_size, num_sources, input_dim]

        # Compute attention scores for each source
        attention_scores = self.weight_layer(stacked_inputs).squeeze(-1)  # [batch_size, num_sources]
        attention_weights = F.softmax(attention_scores, dim=-1)  # Normalize weights across sources

        # Weighted sum of inputs
        # attention_weights needs to be [batch_size, num_sources, 1] for broadcasting
        weighted_sum = torch.sum(stacked_inputs * attention_weights.unsqueeze(-1), dim=1)  # [batch_size, input_dim]
        return weighted_sum


class LocalGlobalInteraction(nn.Module):
    def __init__(self, local_dim, global_dim, hidden_dim):
        """
        Local-Global Feature Interaction Module.
        Enhances local features using information from global features.

        Args:
            local_dim (int): Dimension of local features.
            global_dim (int): Dimension of global features.
            hidden_dim (int): Dimension of the intermediate hidden layer.
        """
        super(LocalGlobalInteraction, self).__init__()
        self.fc_global_to_local = nn.Linear(global_dim, local_dim)  # Project global features to local feature space
        self.fc_local_transform = nn.Linear(local_dim, hidden_dim)  # Non-linear transformation for local features
        self.fc_out_transform = nn.Linear(hidden_dim, local_dim)  # Output transformation back to local space

        # Using MultiheadAttention for interaction.
        # Query: local_feat, Key/Value: projected global_feat
        self.attention = nn.MultiheadAttention(embed_dim=local_dim, num_heads=4,
                                               batch_first=False)  # batch_first=False default

    def forward(self, local_feat, global_feat):
        """
        Forward pass for local-global interaction.

        Args:
            local_feat (Tensor): Local features of shape [batch_size, local_dim].
            global_feat (Tensor): Global features of shape [batch_size, global_dim].
        Returns:
            enhanced_local_feat (Tensor): Enhanced local features of shape [batch_size, local_dim].
        """
        # Project global features to the same dimension as local features
        global_proj = self.fc_global_to_local(global_feat)  # [batch_size, local_dim]

        # Prepare features for MultiheadAttention (expects seq_len, batch_size, embed_dim)
        local_feat_attn = local_feat.unsqueeze(0)  # [1, batch_size, local_dim]
        global_proj_attn = global_proj.unsqueeze(0)  # [1, batch_size, local_dim]

        # Attention mechanism for interaction
        # Query=local_feat, Key=global_proj, Value=global_proj
        interaction_output, _ = self.attention(
            query=local_feat_attn,
            key=global_proj_attn,
            value=global_proj_attn
        )
        interaction_output = interaction_output.squeeze(0)  # [batch_size, local_dim]

        # Non-linear transformation and residual connection
        transformed_interaction = self.fc_out_transform(torch.relu(self.fc_local_transform(interaction_output)))
        enhanced_local_feat = local_feat + transformed_interaction  # Residual connection
        return enhanced_local_feat


# ==============================================================================
# Main Model: MISA_FR
# ==============================================================================

class MISA_FR(nn.Module):
    """
    MISA-FR model combining modality-specific encoders, shared/private encoders,
    local-global interaction, and attention for classification.
    """

    def __init__(self, args, feature_dim=512, represent_dim=256):
        super(MISA_FR, self).__init__()
        self.args = args
        self.feature_dim = feature_dim  # Output dimension of modality-specific encoders
        self.represent_dim = represent_dim  # Dimension of shared and private representations
        self.d_k = represent_dim  # Dimension for key/query in attention mechanisms

        # Modality-specific encoders
        self.VideoEncoder = VideoEncoder(feature_dim=self.feature_dim)
        self.AudioEncoder = AudioEncoder(feature_dim=self.feature_dim)

        # Shared and private encoders
        self.SharedEncoder = SharedEncoder(input_dim=self.feature_dim, shared_dim=self.represent_dim)
        self.VideoPrivateEncoder = VideoPrivateEncoder(input_dim=self.feature_dim, private_dim=self.represent_dim)
        self.AudioPrivateEncoder = AudioPrivateEncoder(input_dim=self.feature_dim, private_dim=self.represent_dim)

        # Discriminators (potentially for adversarial training, not explicitly used in this forward pass for loss calculation)
        self.Share_discriminator = Share_Discriminator(self.represent_dim, num_modalities=2)
        self.private_discriminator = Private_Discriminator(self.represent_dim, num_modalities=2)

        # Interaction and Attention
        # Global feature Fso will have dim = 4 * represent_dim (shared_v, shared_a, private_v, private_a)
        self.local_global_interaction = LocalGlobalInteraction(
            local_dim=self.represent_dim,  # Private features are local
            global_dim=4 * self.represent_dim,  # Fso is global
            hidden_dim=self.represent_dim // 2  # Example hidden dim
        )

        # CrossAttention (not directly used in the provided forward pass, but defined)
        # Q_dim = self.represent_dim, k_V_dim = 4*self.represent_dim matches local_global interaction input
        self.cross_attention = CrossAttention(Q_dim=self.represent_dim, k_V_dim=4 * self.represent_dim, d_k=self.d_k)

        # Classifier
        # Input to multihead_attention will be concatenation of enhanced private features: 2 * represent_dim
        # Using 512 as embed_dim based on original code, ensure it matches 2 * represent_dim
        classifier_input_dim = 2 * self.represent_dim  # enhanced_private_v + enhanced_private_a
        self.multihead_attention = nn.MultiheadAttention(embed_dim=classifier_input_dim, num_heads=4, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 512),  # First layer: maps fused features
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Dropout layer
            nn.Linear(512, args.num_classes)  # Output layer for classification
        )

        # Loss functions
        self.sim_loss = MMDLoss()  # For consistency between shared features
        self.diff_loss = HSICLoss()  # For difference between private features

        initialize_weights(self)  # Apply weight initialization

    def forward(self, video, audio):
        # 1. Feature Extraction from raw modalities
        v_feat = self.VideoEncoder(video)  # [batch_size, feature_dim]
        a_feat = self.AudioEncoder(audio)  # [batch_size, feature_dim]

        # 2. Project to Shared and Private Spaces
        shared_v, shared_a = self.SharedEncoder(v_feat, a_feat)  # [batch_size, represent_dim] each
        private_v = self.VideoPrivateEncoder(v_feat)  # [batch_size, represent_dim]
        private_a = self.AudioPrivateEncoder(a_feat)  # [batch_size, represent_dim]

        # 3. Calculate Consistency and Difference Losses
        # These losses are typically used during training, not directly for inference output.
        consistency_loss = self.sim_loss(shared_v, shared_a)
        difference_loss = self.diff_loss(private_v, shared_v) + self.diff_loss(private_a, shared_a)

        # 4. Form Global Context Feature (Fso)
        # Concatenate all shared and private features
        Fso = torch.cat([shared_v, shared_a, private_v, private_a], dim=-1)  # [batch_size, 4 * represent_dim]

        # 5. Local-Global Interaction
        # Enhance private features using the global context Fso
        enhanced_private_v = self.local_global_interaction(private_v, Fso)  # [batch_size, represent_dim]
        enhanced_private_a = self.local_global_interaction(private_a, Fso)  # [batch_size, represent_dim]

        # 6. Concatenate Enhanced Local Features for Classification
        enhanced_features = torch.cat([enhanced_private_v, enhanced_private_a],
                                      dim=-1)  # [batch_size, 2 * represent_dim]

        # 7. Multihead Self-Attention on Enhanced Features
        # Add a sequence dimension (of length 1) for MultiheadAttention
        enhanced_features_attn = enhanced_features.unsqueeze(1)  # [batch_size, 1, 2 * represent_dim]
        attn_output, _ = self.multihead_attention(
            query=enhanced_features_attn,
            key=enhanced_features_attn,
            value=enhanced_features_attn
        )
        attn_output = attn_output.squeeze(1)  # [batch_size, 2 * represent_dim]

        # 8. Classification
        logits = self.classifier(attn_output)  # [batch_size, num_classes]

        return logits, shared_v, private_v, shared_a, private_a, consistency_loss, difference_loss


# ==============================================================================
# Main Execution Block
# ==============================================================================

if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser(description="MISA_FR Model Training and Evaluation")
        parser.add_argument('--num_classes', type=int, default=2, help="Number of output classes for classification.")
        # Add other arguments as needed (e.g., batch_size, epochs, learning_rate)
        return parser.parse_args()


    args = parse_args()

    # Example instantiation and forward pass
    # Ensure device consistency (e.g., .cuda() if using GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dummy input data
    # Video: [Batch, Time, Channels, Height, Width] e.g., for 32 frames of RGB 224x224 video
    video_input = torch.randn([8, 32, 3, 224, 224]).to(device)
    # Audio: [Batch, Channels, Freq_bins, Time_frames] e.g., for mono audio spectrogram
    audio_input = torch.randn([8, 1, 1024, 128]).to(device)

    model = MISA_FR(args).to(device)
    model.eval()  # Set to evaluation mode if not training immediately

    print("Model instantiated on:", device)

    start_time = time()
    with torch.no_grad():  # Disable gradient calculations for inference
        logits, shared_v, private_v, shared_a, private_a, consistency_loss, difference_loss = model(video_input,
                                                                                                    audio_input)
    end_time = time()
    inference_duration = end_time - start_time

    print("\n--- Inference Results ---")
    print("Logits example (first sample):", logits[0])
    print("Logits shape:", logits.shape)
    print(f"Consistency Loss: {consistency_loss.item():.4f}")
    print(f"Difference Loss: {difference_loss.item():.4f}")
    print(f"Inference time: {inference_duration:.4f}s")


    def compute_params_flops(model_instance, video_sample, audio_sample):
        """Computes and prints model parameters and FLOPs."""
        print("\n--- Model Complexity ---")
        # Ensure model and inputs are on the same device for profile
        model_instance = model_instance.to(video_sample.device)

        # profile expects inputs as a tuple
        flops, params = profile(model_instance, inputs=(video_sample, audio_sample), verbose=False)

        flops_g = flops / 1e9  # Convert FLOPs to GigaFLOPs (GFLOPs)
        params_m = params / 1e6  # Convert parameters to Mega Parameters (M)

        print(f'Parameters: {params_m:.4f} M')
        print(f'FLOPs: {flops_g:.4f} GFLOPs')


    # Compute FLOPs and Params using single batch inputs
    video_sample_for_profile = torch.randn([1, 32, 3, 224, 224]).to(device)
    audio_sample_for_profile = torch.randn([1, 1, 1024, 128]).to(device)
    compute_params_flops(model, video_sample_for_profile, audio_sample_for_profile)
