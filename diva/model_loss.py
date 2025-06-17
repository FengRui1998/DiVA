import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# Consistency Losses
# Aims to make representations from different modalities or sources similar.
# ==============================================================================

class CosineSimilarityLoss(nn.Module):
    """
    Computes loss based on cosine similarity.
    Loss = 1 - mean(cosine_similarity), encouraging vectors to be more similar.
    """
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, v, a):
        # Calculate cosine similarity along the last dimension
        cosine_similarity = torch.nn.functional.cosine_similarity(v, a, dim=-1)
        # Loss is 1 minus the mean similarity, so lower loss means higher similarity
        loss = 1 - cosine_similarity.mean()
        return loss


class MSELoss(nn.Module):
    """
    Standard Mean Squared Error loss.
    """
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, v, a):
        loss = torch.nn.functional.mse_loss(v, a)
        return loss


class CMD(nn.Module):
    """
    Central Moment Discrepancy (CMD) loss.
    Measures the difference between distributions by comparing their central moments.
    """
    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments=5):
        # Clamp inputs for numerical stability
        x1 = torch.clamp(x1, min=-1e3, max=1e3)
        x2 = torch.clamp(x2, min=-1e3, max=1e3)

        # Calculate means
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)

        # Calculate centered versions of x1 and x2
        sx1 = x1 - mx1
        sx2 = x2 - mx2

        # Difference between means (1st moment)
        dm = self.matchnorm(mx1, mx2)
        scms = dm

        # Add differences for higher order central moments
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        """Helper function to compute L2 norm of the difference between two vectors."""
        power = torch.pow(x1 - x2, 2)
        summed = torch.sum(power)
        sqrt = summed ** 0.5
        return sqrt

    def scm(self, sx1, sx2, k):
        """Helper function to compute the matchnorm for the k-th central moment."""
        # Calculate k-th power of centered inputs
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)


class ConsistencyLoss(nn.Module):
    """
    Calculates consistency loss based on the similarity of similarity matrices
    from different modalities.
    """
    def __init__(self):
        super(ConsistencyLoss, self).__init__()

    def forward(self, features_a, features_v):
        """
        Args:
        - features_t: Temporal modality features [batch_size, feature_dim] (Note: This parameter is in the original docstring but not used in the function body)
        - features_a: Audio modality features [batch_size, feature_dim]
        - features_v: Video modality features [batch_size, feature_dim]

        Returns:
        - con_loss: Consistency loss
        """
        # 1. Normalize feature representations (L2 norm normalization)
        features_a = F.normalize(features_a, p=2, dim=1)
        features_v = F.normalize(features_v, p=2, dim=1)

        # 2. Calculate similarity matrices S_m
        S_a = features_a @ features_a.T  # Audio modality similarity matrix [batch_size, batch_size]
        S_v = features_v @ features_v.T  # Video modality similarity matrix [batch_size, batch_size]

        # 3. Calculate the square of the Frobenius norm (as MSE) for consistency loss
        loss_av = F.mse_loss(S_a, S_v)  # Audio-video modality loss

        # 4. Total consistency loss
        con_loss = loss_av

        return con_loss


class KLDivergenceLoss(nn.Module):
    """
    Computes Kullback-Leibler Divergence loss between two sets of features,
    treated as probability distributions after softmax.
    """
    def __init__(self):
        super(KLDivergenceLoss, self).__init__()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, features_a, features_v):
        """
        Args:
        - features_a: Feature vector of modality A [batch_size, feature_dim] (P in KL(P||Q))
        - features_v: Feature vector of modality V [batch_size, feature_dim] (Q in KL(P||Q))

        Returns:
        - KL divergence loss
        """
        # KLDivLoss expects log-probabilities for the first argument and probabilities for the second.
        loss = self.kl_loss(F.log_softmax(features_a, dim=1), F.softmax(features_v, dim=1))
        return loss


class JSDivergenceLoss(nn.Module):
    """
    Computes Jensen-Shannon Divergence loss between two sets of features.
    """
    def __init__(self):
        super(JSDivergenceLoss, self).__init__()

    def forward(self, features_a, features_v):
        """
        Args:
        - features_a: Feature vector of modality A [batch_size, feature_dim]
        - features_v: Feature vector of modality V [batch_size, feature_dim]

        Returns:
        - JS divergence loss
        """
        # Mixture distribution
        m = 0.5 * (features_a + features_v)

        # KL(P || M)
        kl_pm = F.kl_div(F.log_softmax(features_a, dim=1), F.softmax(m, dim=1), reduction='batchmean')
        # KL(Q || M)
        kl_qm = F.kl_div(F.log_softmax(features_v, dim=1), F.softmax(m, dim=1), reduction='batchmean')

        return 0.5 * (kl_pm + kl_qm)


class RBF(nn.Module):
    """
    Radial Basis Function (RBF) kernel.
    Can compute a sum of RBF kernels with different bandwidths.
    """
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        # Multipliers for bandwidth, creating a range of kernel scales
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth # Base bandwidth, if None, it's estimated

    def get_bandwidth(self, L2_distances):
        """Estimates bandwidth using the median heuristic if not provided."""
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            # A common heuristic for bandwidth selection (median of pairwise distances)
            # This implementation uses sum / (n^2 - n), which is related to average pairwise distance.
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)
        return self.bandwidth

    def forward(self, X):
        # Get device of input X
        device = X.device
        # Ensure bandwidth_multipliers are on the same device as X
        self.bandwidth_multipliers = self.bandwidth_multipliers.to(device)

        L2_distances = torch.cdist(X, X) ** 2
        L2_distances = L2_distances.to(device) # Ensure L2_distances is on the correct device

        # Calculate RBF kernel values for each bandwidth and sum them up
        # L2_distances shape: [N, N]
        # self.bandwidth_multipliers shape: [n_kernels]
        # Effective bandwidths: self.get_bandwidth(L2_distances) * self.bandwidth_multipliers
        # Resulting kernel K_ij = sum_k exp(-||x_i - x_j||^2 / (2 * sigma_k^2)) where sigma_k is derived from bandwidth
        return torch.exp(
            -L2_distances[None, ...] /
            (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]
        ).sum(dim=0)


class MMDLoss(nn.Module):
    """
    Maximum Mean Discrepancy (MMD) loss.
    Measures the distance between two distributions in a Reproducing Kernel Hilbert Space (RKHS).
    Uses an RBF kernel by default.
    """
    def __init__(self, kernel=None): # Changed default to None to initialize RBF() inside if not provided
        super().__init__()
        if kernel is None:
            self.kernel = RBF()
        else:
            self.kernel = kernel

    def forward(self, X, Y):
        # Stack X and Y to compute the full kernel matrix K_((X,Y), (X,Y))
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        # K_XX: kernel matrix between samples of X
        XX = K[:X_size, :X_size].mean()
        # K_XY: kernel matrix between samples of X and Y
        XY = K[:X_size, X_size:].mean()
        # K_YY: kernel matrix between samples of Y
        YY = K[X_size:, X_size:].mean()

        # MMD^2 = E[k(X,X)] - 2E[k(X,Y)] + E[k(Y,Y)]
        return XX - 2 * XY + YY


# class MMDLoss(nn.Module):
#     def __init__(self, sigma=1.0):
#         super(MMDLoss, self).__init__()
#         self.sigma = sigma  # Gaussian kernel parameter

#     def rbf_kernel(self, X, Y=None):
#         """
#         Compute RBF kernel (Gaussian kernel)
#         :param X: Feature matrix 1 [batch_size, feature_dim]
#         :param Y: Feature matrix 2 [batch_size, feature_dim]
#         :return: Computed kernel matrix
#         """
#         if Y is None:
#             Y = X
#         pairwise_dists = torch.cdist(X, Y, p=2) ** 2  # Compute squared Euclidean distances
#         kernel_matrix = torch.exp(-pairwise_dists / (2 * self.sigma ** 2))  # Gaussian kernel
#         return kernel_matrix

#     def forward(self, features_a, features_v):
#         """
#         Compute MMD loss between two feature distributions
#         :param features_a: Features A (e.g., audio features)
#         :param features_v: Features V (e.g., video features)
#         :return: MMD loss
#         """
#         # Compute RBF kernel matrices for features A and V
#         K_aa = self.rbf_kernel(features_a)
#         K_av = self.rbf_kernel(features_a, features_v)
#         K_vv = self.rbf_kernel(features_v)

#         # Compute MMD loss
#         mmd_loss = torch.mean(K_aa) + torch.mean(K_vv) - 2 * torch.mean(K_av)
#         return mmd_loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Aims to pull similar samples together and push dissimilar samples apart.
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, feature_a, feature_v, label):
        """
        Args:
        - feature_a: Feature vector of modality A [batch_size, feature_dim]
        - feature_v: Feature vector of modality V [batch_size, feature_dim]
        - label: Similarity label, 1 for similar, 0 for dissimilar [batch_size]

        Returns:
        - Contrastive loss
        """
        distances = torch.norm(feature_a - feature_v, p=2, dim=1) # Euclidean distance
        # For similar pairs (label=1), loss is distance^2
        # For dissimilar pairs (label=0), loss is max(0, margin - distance)^2
        loss = 0.5 * (label * distances.pow(2) +
                      (1 - label) * F.relu(self.margin - distances).pow(2))
        return loss.mean()


class WassersteinDistance(nn.Module):
    """
    Computes the 1-Wasserstein distance between two sets of 1D distributions (features).
    This is done by sorting the features and summing the absolute differences.
    Assumes features are 1D or calculates distance per dimension and averages.
    """
    def __init__(self):
        super(WassersteinDistance, self).__init__()

    def forward(self, X1, X2):
        # Sort features along each dimension (dim=0 assuming batch is first dim)
        X1_sorted, _ = torch.sort(X1, dim=0)
        X2_sorted, _ = torch.sort(X2, dim=0)

        # Compute Wasserstein distance as the mean absolute difference of sorted features
        # This is exact for 1D distributions. For multivariate, it's a sum of 1D W-distances.
        wasserstein_dist = torch.mean(torch.abs(X1_sorted - X2_sorted))
        return wasserstein_dist


class KernelCenteredDistance(nn.Module):
    """
    Computes a distance based on centered kernel matrices (Gram matrices).
    Related to HSIC or MMD using centered kernels.
    """
    def __init__(self, sigma=1.0):
        super(KernelCenteredDistance, self).__init__()
        self.sigma = sigma # Bandwidth for RBF kernel

    def _kernel_matrix(self, X):
        """Computes the RBF kernel matrix K for input X."""
        pairwise_dists = torch.cdist(X, X, p=2) ** 2
        K = torch.exp(-pairwise_dists / (2 * self.sigma ** 2))
        return K

    def _center_kernel_matrix(self, K):
        """Centers the kernel matrix K using H K H, where H is the centering matrix."""
        b = K.size(0) # Batch size or number of samples
        H = torch.eye(b).to(K.device) - 1 / b * torch.ones((b, b)).to(K.device)
        K_centered = H @ K @ H
        return K_centered

    def forward(self, X1, X2):
        K1 = self._kernel_matrix(X1)
        K2 = self._kernel_matrix(X2)

        K1_centered = self._center_kernel_matrix(K1)
        K2_centered = self._center_kernel_matrix(K2)

        # Distance is the squared Frobenius norm of the difference between centered kernel matrices,
        # normalized by the square of the number of samples.
        distance = torch.norm(K1_centered - K2_centered, p='fro') ** 2 / (K1.size(0) ** 2)
        return distance


# ==============================================================================
# Difference / Discrepancy Losses
# Aims to make representations from different modalities or sources distinct
# or to measure their statistical independence/difference.
# ==============================================================================

class DiffLoss(nn.Module):
    """
    Difference Loss.
    Encourages orthogonality between two sets of features after normalization.
    Minimizing this loss makes the feature spaces less correlated.
    """
    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1) # Flatten features
        input2 = input2.view(batch_size, -1) # Flatten features

        # Zero mean normalization (centering)
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        # L2 normalization (scaling to unit norm per sample)
        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6) # Add epsilon for numerical stability

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6) # Add epsilon for numerical stability

        # Difference loss is computed as the mean of the squared Frobenius norm
        # of the cross-covariance like matrix (input1_l2.T @ input2_l2).
        # This penalizes correlation between the two sets of normalized features.
        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss


class HSICLoss(nn.Module):
    """
    Hilbert-Schmidt Independence Criterion (HSIC) Loss.
    Measures the statistical independence between two sets of features P1 and P2.
    A smaller HSIC value indicates greater independence.
    """
    def __init__(self, sigma=1.0, eps=1e-6):
        super(HSICLoss, self).__init__()
        self.sigma = sigma  # Gaussian kernel bandwidth parameter
        self.eps = eps      # Numerical stability parameter

    def rbf_kernel(self, X):
        """
        Compute Gaussian RBF kernel matrix K.
        Args:
        - X: Feature matrix [batch_size, feature_dim]
        Returns:
        - Gaussian kernel matrix K [batch_size, batch_size]
        """
        pairwise_dist = torch.cdist(X, X, p=2)  # Compute pairwise Euclidean distances
        K = torch.exp(-pairwise_dist ** 2 / (2 * self.sigma ** 2))  # Gaussian kernel formula
        return K

    def center_kernel(self, K):
        """
        Center the kernel matrix K.
        Args:
        - K: Kernel matrix [batch_size, batch_size]
        Returns:
        - Centered kernel matrix
        """
        n = K.size(0)
        # Centering matrix H = I - (1/n) * 1_n * 1_n^T
        U = torch.eye(n).to(K.device) - (1 / n) * torch.ones(n, n).to(K.device)
        return U @ K @ U

    def forward(self, P1, P2):
        """
        Compute HSIC loss between two modalities.
        Args:
        - P1: Feature representation of modality 1 [batch_size, feature_dim]
        - P2: Feature representation of modality 2 [batch_size, feature_dim]
        Returns:
        - HSIC value (a scalar). Smaller values indicate more independence.
        """
        # Compute Gaussian kernel matrices
        K1 = self.rbf_kernel(P1)
        K2 = self.rbf_kernel(P2)

        # Center kernel matrices
        K1_centered = self.center_kernel(K1)
        K2_centered = self.center_kernel(K2)

        # Compute HSIC
        n = K1.size(0)
        # HSIC = tr(K1_c * K2_c) / (n-1)^2
        hsic_value = torch.trace(K1_centered @ K2_centered) / ((n - 1) ** 2 + self.eps) # Add eps to avoid division by zero if n=1
        return hsic_value


class MSE(nn.Module): # Note: This is a manual MSE implementation, distinct from the earlier MSELoss class.
    """
    Manual implementation of Mean Squared Error.
    """
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data) # Total number of elements
        mse = torch.sum(diffs.pow(2)) / n
        return mse


class OrthogonalityLoss(nn.Module):
    """
    Orthogonality Loss.
    Penalizes non-orthogonality between two sets of features.
    Minimizing this loss encourages features_a.T @ features_v to be close to zero.
    """
    def __init__(self):
        super(OrthogonalityLoss, self).__init__()

    def forward(self, features_a, features_v):
        """
        Args:
        - features_a: Feature vector of modality A [batch_size, feature_dim]
        - features_v: Feature vector of modality V [batch_size, feature_dim]

        Returns:
        - Orthogonality loss (squared Frobenius norm of the product matrix)
        """
        # prod is like a cross-covariance matrix if features were mean-centered.
        # Shape: [feature_dim_a, feature_dim_v]
        prod = torch.mm(features_a.t(), features_v)
        # Squared Frobenius norm of the product matrix.
        loss = torch.norm(prod, p='fro').pow(2)
        return loss


class NegativeCorrelationLoss(nn.Module):
    """
    Negative Correlation Loss.
    Note: Minimizing the current formulation of this loss actually promotes
    *positive* correlation between the corresponding features of A and V.
    The loss is -trace(Cov(A,V))/N. Minimizing this maximizes trace(Cov(A,V))/N.
    """
    def __init__(self):
        super(NegativeCorrelationLoss, self).__init__()

    def forward(self, features_a, features_v):
        """
        Args:
        - features_a: Feature vector of modality A [batch_size, feature_dim]
        - features_v: Feature vector of modality V [batch_size, feature_dim]

        Returns:
        - A loss value. Minimizing it maximizes the trace of the covariance matrix.
        """
        # Center features (subtract mean)
        features_a = features_a - features_a.mean(dim=0, keepdim=True)
        features_v = features_v - features_v.mean(dim=0, keepdim=True)

        # Compute unnormalized covariance matrix (sum of outer products)
        cov = torch.mm(features_a.t(), features_v) # Shape [dim_a, dim_v]

        # Loss is negative trace of covariance, normalized by batch size.
        # Minimizing this loss term maximizes sum_i Cov(a_i, v_i).
        loss = -torch.trace(cov) / features_a.size(0)
        return loss


class MutualInformationLoss(nn.Module):
    """
    Mutual Information Maximization Loss (approximated).
    This loss aims to maximize the mutual information between features_a and features_v
    by minimizing the negative of an MI estimate.
    The specific MI estimation here seems related to contrastive methods.
    """
    def __init__(self):
        super(MutualInformationLoss, self).__init__()

    def forward(self, features_a, features_v):
        """
        Args:
        - features_a: Feature vector of modality A [batch_size, feature_dim]
        - features_v: Feature vector of modality V [batch_size, feature_dim]

        Returns:
        - Negative mutual information (approximation)
        """
        # Normalize input features (L2 norm)
        features_a = torch.nn.functional.normalize(features_a, dim=1)
        features_v = torch.nn.functional.normalize(features_v, dim=1)

        # Approximate joint distribution p(a_i, v_j) or conditional p(v_j | a_i)
        # using softmax over dot products. features_a @ features_v.T gives [batch_size, batch_size] matrix of dot products.
        # Softmax over dim=1 means sum_j p_joint[i,j] = 1 for each i.
        # So, p_joint[i,j] can be interpreted as an estimate of P(v_j | a_i).
        p_joint = torch.softmax(features_a @ features_v.T, dim=1)

        # Marginal distributions
        # p_marginal_a[i] = sum_j p_joint[i,j]. Since p_joint is softmaxed over dim=1, this will be 1.
        p_marginal_a = torch.sum(p_joint, dim=1, keepdim=True)
        # p_marginal_v[j] = sum_i p_joint[i,j]. This is sum_i P(v_j | a_i).
        p_marginal_v = torch.sum(p_joint, dim=0, keepdim=True)

        # Numerical stability: clamp probabilities to avoid log(0)
        p_joint = torch.clamp(p_joint, min=1e-6)
        p_marginal_a = torch.clamp(p_marginal_a, min=1e-6) # Will be clamped at 1.0 typically
        p_marginal_v = torch.clamp(p_marginal_v, min=1e-6)

        # Compute mutual information: MI = sum_{i,j} p_joint[i,j] * log( p_joint[i,j] / (p_marginal_a[i] * p_marginal_v[j]) )
        # Since p_marginal_a[i] is effectively 1, log(p_marginal_a[i]) is 0.
        # The formula becomes sum_{i,j} p_joint[i,j] * (log(p_joint[i,j]) - log(p_marginal_v[j]))
        mi = torch.sum(
            p_joint * (
                torch.log(p_joint) -
                torch.log(p_marginal_a) - # This term will be log(1) = 0
                torch.log(p_marginal_v)
            )
        )

        # Return negative mutual information as loss (to be minimized)
        return -mi


# ==============================================================================
# Adversarial Losses
# Used in generative adversarial networks (GANs) or domain adaptation.
# ==============================================================================

class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL).
    In the forward pass, it acts as an identity function.
    In the backward pass, it multiplies the gradient by -1.
    Used for adversarial training to train a feature extractor that produces
    domain-invariant features, by trying to fool a domain classifier.
    """
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x) # Ensure it's a view, common practice

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse the gradient
        return -grad_output


class AdversarialLoss(nn.Module):
    """
    A common adversarial loss, typically Negative Log Likelihood Loss (NLLLoss)
    used with log-softmax outputs from a discriminator/classifier.
    """
    def forward(self, logits, labels):
        # NLLLoss is often used when the discriminator's output is log-probabilities.
        return F.nll_loss(logits, labels)


# Example usage (commented out)
# input_v = torch.randn([8, 256])
# input_a = torch.randn([8, 256])
# loss_fn = KLDivergenceLoss() # Renamed 'loss' variable to 'loss_fn' to avoid conflict if 'loss' module is imported
# y = loss_fn(input_v, input_a)
# print(y)