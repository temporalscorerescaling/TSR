import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class GaussianMixture1D:
    def __init__(self, mus, variances, weights, device, timestep=0):
        """
        Initialize a 1D Gaussian mixture using PyTorch tensors on the specified device.

        Args:
            mus: List or array-like of shape (N,) or torch Tensor of shape (N,), means of each component.
            variances: List of length N, each element a scalar variance.
            weights: List or array-like of shape (N,) or torch Tensor, mixing weights. Will be normalized to sum to 1.
            device: 'cpu' or 'cuda' device string or torch.device.
            timestep: int diffusion timestep associated with this mixture (optional).
        """
        self.timestep = timestep
        self.device = torch.device(device)
        
        # Process mus
        mus_tensor = torch.tensor(mus, dtype=torch.float32, device=self.device)
        if mus_tensor.ndim != 1:
            raise ValueError("mus must be of shape (N,)")
        self.mus = mus_tensor  # shape (N,)
        self.N = self.mus.shape[0]

        # Process variances
        if len(variances) != self.N:
            raise ValueError("variances must have length N, matching number of mus")
        
        variances_list = []
        for idx, v in enumerate(variances):
            if isinstance(v, torch.Tensor):
                v_tensor = v.to(self.device).to(torch.float32)
            else:
                v_tensor = torch.tensor(v, dtype=torch.float32, device=self.device)
            
            if v_tensor.ndim == 0:
                if v_tensor.item() <= 0:
                    raise ValueError(f"Variance at index {idx} must be positive.")
                variances_list.append(v_tensor)
            else:
                raise ValueError(f"Variance at index {idx} must be a scalar.")
        
        self.variances = torch.stack(variances_list, dim=0)  # (N,)
        
        # Precompute normalization coefficients
        self.norm_coefs = 1.0 / torch.sqrt(2 * torch.pi * self.variances)  # (N,)

        # Process weights
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
        if weights_tensor.ndim != 1 or weights_tensor.shape[0] != self.N:
            raise ValueError("weights must be of shape (N,)")
        if torch.any(weights_tensor < 0):
            raise ValueError("All weights must be non-negative.")
        total_w = torch.sum(weights_tensor)
        if total_w <= 0:
            raise ValueError("Sum of weights must be positive.")
        self.weights = weights_tensor / total_w  # normalized (N,)

    def get_diffused_distribution(self, betas, alphas, alpha_bars, t):
        """
        Given a diffusion schedule and a timestep t, return a new GaussianMixture1D
        that represents the noised distribution at timestep t.

        Args:
            betas: sequence of betas for timesteps 1..T
            alphas: sequence of alphas (1-beta)
            alpha_bars: sequence of cumulative products of alphas
            t: int index (0-based) for the diffusion timestep
        Returns:
            GaussianMixture1D: new instance with means, variances, timestep updated for timestep t
        """
        # Assert original timestep is 0 or None
        if hasattr(self, 'timestep') and self.timestep != 0:
            raise ValueError("get_diffused_distribution should be called on the original distribution (timestep=0)")
        # Store schedule temporarily
        betas_t = torch.tensor(betas, dtype=torch.float32, device=self.device)
        alphas_t = torch.tensor(alphas, dtype=torch.float32, device=self.device)
        alpha_bars_t = torch.tensor(alpha_bars, dtype=torch.float32, device=self.device)
        if not (len(betas_t) == len(alphas_t) == len(alpha_bars_t)):
            raise ValueError("betas, alphas, alpha_bars must have the same length")
        if not (0 <= t < len(alpha_bars_t)):
            raise ValueError(f"t must be in [0, {len(alpha_bars_t)-1}]")
        alpha_bar = alpha_bars_t[t]
        sqrt_ab = torch.sqrt(alpha_bar)
        # New means and variances
        new_mus = self.mus * sqrt_ab
        new_variances = self.variances * alpha_bar + (1.0 - alpha_bar)
        # Weights remain the same
        new_weights = self.weights
        # Create new instance with updated timestep
        new = GaussianMixture1D(new_mus.cpu().numpy(), new_variances.cpu().numpy(), 
                               new_weights.cpu().numpy(), device=self.device, timestep=t)
        return new
    
    def get_flowed_distribution(self, t):
        """
        Given a flow parameter t, return a new GaussianMixture1D that represents the flowed distribution.

        Args:
            t: int index (0-based) for the diffusion timestep
        Returns:
            GaussianMixture1D: new instance with means, variances, timestep updated for timestep t
        """
        # Assert original timestep is 0 or None
        if hasattr(self, 'timestep') and self.timestep != 0:
            raise ValueError("get_flowed_distribution should be called on the original distribution (timestep=0)")
        # New means and variances
        new_mus = self.mus * (1 - t)
        new_variances = self.variances * (1 - t**2) + t**2
        # Weights remain the same
        new_weights = self.weights
        # Create new instance with updated timestep
        new = GaussianMixture1D(new_mus.cpu().numpy(), new_variances.cpu().numpy(),
                               new_weights.cpu().numpy(), device=self.device, timestep=t)
        return new

    def pdf(self, points):
        """
        Compute the density p(x) at given point(s).

        Args:
            points: torch.Tensor of shape (..., 1) or (...,), coordinates to evaluate density.
        Returns:
            Tensor of shape (...) with density values.
        """
        pts = points.to(self.device).to(torch.float32)
        if pts.ndim == 0:
            pts = pts.unsqueeze(0)
        if pts.shape[-1] == 1:
            pts = pts.squeeze(-1)
        
        leading_shape = pts.shape
        pts_flat = pts.reshape(-1)
        M = pts_flat.shape[0]
        
        # Compute diff: (M, N)
        diff = pts_flat.unsqueeze(1) - self.mus.unsqueeze(0)
        
        # Squared distances: (M, N)
        sq_dist = diff ** 2
        
        # Component densities: (M, N)
        comp_density = self.norm_coefs.unsqueeze(0) * torch.exp(-0.5 * sq_dist / self.variances.unsqueeze(0))
        
        # Weighted sum: (M,)
        densities_flat = torch.matmul(comp_density, self.weights)
        return densities_flat.view(leading_shape)

    @torch.no_grad()
    def score(self, points):
        """
        Compute the score ∇_x log p(x) at given points for the current mixture.

        Args:
            points: torch.Tensor of shape (..., 1) or (...,)
        Returns:
            Tensor of same shape, representing the score.
        """
        pts = points.to(self.device).float()
        if pts.ndim == 0:
            pts = pts.unsqueeze(0)
        if pts.shape[-1] == 1:
            pts = pts.squeeze(-1)
        
        leading_shape = pts.shape
        pts_flat = pts.reshape(-1)
        M = pts_flat.shape[0]
        
        # diff: (M, N)
        diff = pts_flat.unsqueeze(1) - self.mus.unsqueeze(0)
        
        # Squared distances
        sq_dist = diff ** 2
        
        # component pdfs: (M, N)
        comp_pdf = self.norm_coefs.unsqueeze(0) * torch.exp(-0.5 * sq_dist / self.variances.unsqueeze(0))
        
        # weighted pdf
        w_comp = comp_pdf * self.weights.unsqueeze(0)
        px = torch.sum(w_comp, dim=1, keepdim=True)  # (M,1)
        
        # component scores: (M,N)
        score_comp = -diff / self.variances.unsqueeze(0)
        weighted_score = score_comp * w_comp  # (M,N)
        score_flat = torch.sum(weighted_score, dim=1) / px.squeeze(-1)  # (M,)
        return score_flat.view(leading_shape)

    def sample(self, n_samples):
        """
        Sample points from the mixture.

        Args:
            n_samples: int, number of samples to draw.
        Returns:
            Tensor of shape (n_samples,) with samples, and component indices.
        """
        indices = torch.multinomial(self.weights, num_samples=n_samples, replacement=True)
        z = torch.randn(n_samples, device=self.device)
        
        chosen_vars = self.variances[indices]
        chosen_means = self.mus[indices]
        
        samples = chosen_means + torch.sqrt(chosen_vars) * z
        return samples, indices

    def evaluate_samples(self, samples):
        """
        Evaluate a batch of samples under the mixture model.

        Args:
            samples: torch.Tensor of shape (M,)
        Returns:
            grouped: list of length N, each element is a tensor of samples assigned to that mode
            mode_weights: torch.Tensor of shape (N,), fractions of samples per mode
            mode_variances: torch.Tensor of shape (N,), variance per mode
            avg_log_likelihood: float, average log p(x) over samples
        """
        pts = samples.to(self.device).float()
        if pts.ndim != 1:
            raise ValueError("samples must have shape (M,)")
        M = pts.shape[0]
        
        # Assign to closest mode
        with torch.no_grad():
            dists = torch.abs(pts.unsqueeze(1) - self.mus.unsqueeze(0))  # (M, N)
            assignments = torch.argmin(dists, dim=1)  # (M,)
        
        # Grouped samples
        grouped = []
        mode_counts = torch.zeros(self.N, device=self.device)
        mode_variances = []
        
        for i in range(self.N):
            group = pts[assignments == i]
            grouped.append(group)
            mode_counts[i] = group.shape[0]
            
            if group.shape[0] > 1:
                mean = group.mean()
                var = torch.var(group, unbiased=True)
            else:
                var = torch.tensor(0.0, device=self.device)
            mode_variances.append(var)
        
        # Mode weights by count
        mode_weights = mode_counts / M
        mode_variances = torch.stack(mode_variances, dim=0)  # (N,)
        
        # Average log likelihood
        pdf_vals = self.pdf(pts)  # (M,)
        eps = 1e-12
        log_lik = torch.log(pdf_vals + eps)
        avg_log_likelihood = log_lik.mean().item()
        
        return grouped, mode_weights, mode_variances, avg_log_likelihood

    def plot_density(self, k=1.0, num_points=1000, save_path=None, ax=None, x_lims=None, **kwargs):
        """
        Plot the 1D density as a line plot with density on y-axis and data values on x-axis.
        """
        # Determine plotting range
        max_std = float(torch.sqrt(torch.max(self.variances)).item())
        
        if x_lims is None:
            x_min = float(torch.min(self.mus).item()) - 3 * max_std
            x_max = float(torch.max(self.mus).item()) + 3 * max_std
        else:
            x_min, x_max = x_lims
        
        # Create grid on device
        x = torch.linspace(x_min, x_max, num_points, device=self.device)
        
        # Compute pdf on grid
        pdf_vals = self.pdf(x)  # (num_points,)
        if k != 1.0:
            orig_pdf_sum = pdf_vals.sum()
            pdf_vals = pdf_vals ** k
            scaled_pdf_sum = pdf_vals.sum()
            pdf_vals = pdf_vals * (orig_pdf_sum / scaled_pdf_sum)
        
        # Assign each point to closest gaussian mode
        with torch.no_grad():
            dists = torch.abs(x.unsqueeze(1) - self.mus.unsqueeze(0))  # (num_points, N)
            assignments = torch.argmin(dists, dim=1)  # (num_points,)
            mode_weights = torch.zeros(self.N, device=self.device)
            mode_vars = torch.zeros(self.N, device=self.device)
            total_pdf = pdf_vals.sum()
            
            for i in range(self.N):
                mask = assignments == i
                mode_weights[i] = pdf_vals[mask].sum()
                if mask.sum() > 0:
                    centered = x[mask] - self.mus[i]
                    sq_dist = centered ** 2
                    weighted_var = (sq_dist * pdf_vals[mask]).sum() / pdf_vals[mask].sum()
                    mode_vars[i] = weighted_var
            
            if total_pdf > 0:
                mode_weights = mode_weights / total_pdf
        
        mode_weights_np = mode_weights.cpu().numpy()
        mode_vars_np = mode_vars.cpu().numpy()
        
        # Prepare for plotting
        pdf_vals_np = pdf_vals.cpu().detach().numpy()
        x_np = x.cpu().numpy()
        
        # Setup axis
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots()
            created_fig = True
        
        ax.plot(x_np, pdf_vals_np, 'b-', linewidth=2)
        ax.fill_between(x_np, pdf_vals_np, alpha=0.3)
        
        # Mark the means
        for i, mu in enumerate(self.mus.cpu().numpy()):
            mu_density = self.pdf(torch.tensor(mu, device=self.device)).item()
            if k != 1.0:
                # Recompute for this point with k scaling
                orig_val = self.pdf(torch.tensor(mu, device=self.device)).item()
                scaled_val = orig_val ** k
                mu_density = scaled_val
            ax.axvline(mu, color='red', linestyle='--', alpha=0.7)
            ax.plot(mu, mu_density, 'ro', markersize=8)
        
        weights_str = ", ".join([f"{w:.2f}" for w in mode_weights_np])
        vars_str = ", ".join([f"{v:.2e}" for v in mode_vars_np])
        ax.set_title(f"t={self.timestep}, weights=[{weights_str}]")
        ax.set_xlabel('x')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        
        if save_path and created_fig:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
        
        return mode_weights_np, mode_vars_np

class GaussianMixture2D:
    def __init__(self, mus, variances, weights, device, timestep=0):
        """
        Initialize a 2D Gaussian mixture, allowing isotropic or full covariance per component,
        using PyTorch tensors on the specified device.

        Args:
            mus: List or array-like of shape (N, 2) or torch Tensor of shape (N,2), means of each component.
            variances: List of length N, each element either:
                       - a scalar (interpreted as isotropic variance)
                       - an array-like of shape (2,2) or torch Tensor (interpreted as full covariance)
            weights: List or array-like of shape (N,) or torch Tensor, mixing weights. Will be normalized to sum to 1.
            device: 'cpu' or 'cuda' device string or torch.device.
            timestep: int diffusion timestep associated with this mixture (optional).
        """
        self.timestep = timestep
        self.device = torch.device(device)
        # Process mus
        mus_tensor = torch.tensor(mus, dtype=torch.float32, device=self.device)
        if mus_tensor.ndim != 2 or mus_tensor.shape[1] != 2:
            raise ValueError("mus must be of shape (N, 2)")
        self.mus = mus_tensor  # shape (N,2)
        self.N = self.mus.shape[0]

        # Process variances / covariances
        covs = []
        if len(variances) != self.N:
            raise ValueError("variances must have length N, matching number of mus")
        for idx, v in enumerate(variances):
            if isinstance(v, torch.Tensor):
                v_tensor = v.to(self.device).to(torch.float32)
            else:
                v_tensor = torch.tensor(v, dtype=torch.float32, device=self.device) if not torch.is_tensor(v) else v.to(self.device).to(torch.float32)

            # Check if scalar (0-d or 1-element tensor)
            if v_tensor.ndim == 0:
                if v_tensor.item() <= 0:
                    raise ValueError(f"Variance at index {idx} must be positive.")
                cov = torch.eye(2, device=self.device, dtype=torch.float32) * v_tensor
            elif v_tensor.ndim == 2 and v_tensor.shape == (2, 2):
                # Check positive-definite via Cholesky
                try:
                    torch.linalg.cholesky(v_tensor)
                except RuntimeError:
                    raise ValueError(f"Covariance matrix at index {idx} is not positive-definite.")
                cov = v_tensor
            else:
                raise ValueError(f"Covariance at index {idx} must be a scalar or 2x2 array/tensor.")
            covs.append(cov)
        # Stack covariances: shape (N,2,2)
        self.covariances = torch.stack(covs, dim=0)  # (N,2,2)

        # Precompute inverses, determinants, normalization coefficients, and Cholesky factors
        inv_covs = []
        det_covs = []
        norm_coefs = []
        chol_covs = []
        for i in range(self.N):
            cov = self.covariances[i]
            inv_cov = torch.linalg.inv(cov)
            det_cov = torch.linalg.det(cov)
            if torch.any(det_cov <= 0):
                raise ValueError(f"Covariance matrix at index {i} must have positive determinant.")
            # Normalization constant: 1 / (2π sqrt(det))
            norm_coef = 1.0 / (2 * torch.pi * torch.sqrt(det_cov))
            # Cholesky for sampling
            chol = torch.linalg.cholesky(cov)
            inv_covs.append(inv_cov)
            det_covs.append(det_cov)
            norm_coefs.append(norm_coef)
            chol_covs.append(chol)

        # Convert lists to tensors
        self.inv_covs = torch.stack(inv_covs, dim=0)       # (N,2,2)
        self.det_covs = torch.stack(det_covs, dim=0)       # (N,)
        self.norm_coefs = torch.stack(norm_coefs, dim=0)   # (N,)
        self.chol_covs = torch.stack(chol_covs, dim=0)     # (N,2,2)

        # Process weights
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
        if weights_tensor.ndim != 1 or weights_tensor.shape[0] != self.N:
            raise ValueError("weights must be of shape (N,)")
        if torch.any(weights_tensor < 0):
            raise ValueError("All weights must be non-negative.")
        total_w = torch.sum(weights_tensor)
        if total_w <= 0:
            raise ValueError("Sum of weights must be positive.")
        self.weights = weights_tensor / total_w  # normalized (N,)

    def get_diffused_distribution(self, betas, alphas, alpha_bars, t):
        """
        Given a diffusion schedule and a timestep t, return a new GaussianMixture2D
        that represents the noised distribution at timestep t.

        Args:
            betas: sequence of betas for timesteps 1..T
            alphas: sequence of alphas (1-beta)
            alpha_bars: sequence of cumulative products of alphas
            t: int index (0-based) for the diffusion timestep
        Returns:
            GaussianMixture2D: new instance with means, covariances, timestep updated for timestep t
        """
        # Assert original timestep is 0 or None
        if hasattr(self, 'timestep') and self.timestep != 0:
            raise ValueError("get_diffused_distribution should be called on the original distribution (timestep=0)")
        # Store schedule temporarily
        betas_t = torch.tensor(betas, dtype=torch.float32, device=self.device)
        alphas_t = torch.tensor(alphas, dtype=torch.float32, device=self.device)
        alpha_bars_t = torch.tensor(alpha_bars, dtype=torch.float32, device=self.device)
        if not (len(betas_t) == len(alphas_t) == len(alpha_bars_t)):
            raise ValueError("betas, alphas, alpha_bars must have the same length")
        if not (0 <= t < len(alpha_bars_t)):
            raise ValueError(f"t must be in [0, {len(alpha_bars_t)-1}]")
        alpha_bar = alpha_bars_t[t]
        sqrt_ab = torch.sqrt(alpha_bar)
        # New means and covariances
        new_mus = (self.mus * sqrt_ab)
        I = torch.eye(2, device=self.device, dtype=torch.float32)
        new_variances = []
        for i in range(self.N):
            cov_t = self.covariances[i] * alpha_bar + I * (1.0 - alpha_bar)
            new_variances.append(cov_t)
        # Weights remain the same
        new_weights = self.weights
        # Create new instance with updated timestep
        new = GaussianMixture2D(new_mus.cpu().numpy(), [v.cpu().numpy() for v in new_variances], new_weights.cpu().numpy(), device=self.device, timestep=t)
        return new
    
    def get_flowed_distribution(self, t):
        """
        Given a diffusion schedule and a timestep t, return a new GaussianMixture2D
        that represents the noised distribution at timestep t.

        Args:
            t: int index (0-based) for the diffusion timestep
        Returns:
            GaussianMixture2D: new instance with means, covariances, timestep updated for timestep t
        """
        # Assert original timestep is 0 or None
        if hasattr(self, 'timestep') and self.timestep != 0:
            raise ValueError("get_diffused_distribution should be called on the original distribution (timestep=0)")
        # New means and covariances
        new_mus = (self.mus * (1 - t))
        I = torch.eye(2, device=self.device, dtype=torch.float32)
        new_variances = []
        for i in range(self.N):
            cov_t = self.covariances[i] * (1 - t**2) + I * t**2
            new_variances.append(cov_t)
        # Weights remain the same
        new_weights = self.weights
        # Create new instance with updated timestep
        new = GaussianMixture2D(new_mus.cpu().numpy(), [v.cpu().numpy() for v in new_variances], new_weights.cpu().numpy(), device=self.device, timestep=t)
        return new


    def pdf(self, points):
        """
        Compute the density p(x) at given point(s) using batched PyTorch operations.

        Args:
            points: torch.Tensor of shape (..., 2), coordinates to evaluate density.
        Returns:
            Tensor of shape (...) with density values.
        """
        pts = points.to(self.device).to(torch.float32)
        if pts.ndim < 1 or pts.shape[-1] != 2:
            raise ValueError("points must have shape (..., 2)")
        leading_shape = pts.shape[:-1]
        pts_flat = pts.reshape(-1, 2)
        M = pts_flat.shape[0]
        # Compute diff: (M, N, 2)
        diff = pts_flat.unsqueeze(1) - self.mus.unsqueeze(0)
        # Mahalanobis distance: (M, N)
        inter = torch.matmul(diff.unsqueeze(-2), self.inv_covs.unsqueeze(0))  # (M,N,1,2)
        mhd = torch.matmul(inter, diff.unsqueeze(-1)).squeeze(-1).squeeze(-1)  # (M,N)
        # Component densities: (M, N)
        comp_density = self.norm_coefs.unsqueeze(0) * torch.exp(-0.5 * mhd)
        # Weighted sum: (M,)
        densities_flat = torch.matmul(comp_density, self.weights)
        return densities_flat.view(leading_shape)

    def score(self, points):
        """
        Compute the score ∇_x log p(x) at given points for the current mixture.

        Args:
            points: torch.Tensor of shape (..., 2)
        Returns:
            Tensor of same leading shape with last dim 2, representing the score.
        """
        pts = points.to(self.device).float()
        if pts.ndim < 1 or pts.shape[-1] != 2:
            raise ValueError("points must have shape (..., 2)")
        lead = pts.shape[:-1]
        flat = pts.reshape(-1, 2)
        M = flat.shape[0]
        # diff: (M, N, 2)
        diff = flat.unsqueeze(1) - self.mus.unsqueeze(0)
        # Mahalanobis distances
        inter = torch.matmul(diff.unsqueeze(-2), self.inv_covs.unsqueeze(0))
        mhd = torch.matmul(inter, diff.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        # component pdfs: (M, N)
        comp_pdf = self.norm_coefs.unsqueeze(0) * torch.exp(-0.5 * mhd)
        # weighted pdf
        w_comp = comp_pdf * self.weights.unsqueeze(0)
        px = torch.sum(w_comp, dim=1, keepdim=True)  # (M,1)
        
        # Fix zero division issue: set zero elements in px to 1 to avoid NaN
        px_safe = torch.where(px == 0, torch.ones_like(px), px)
        
        # component scores: (M,N,2)
        score_comp = -torch.matmul(self.inv_covs.unsqueeze(0), diff.unsqueeze(-1)).squeeze(-1)
        weighted_score = score_comp * w_comp.unsqueeze(-1)  # (M,N,2)
        score_flat = torch.sum(weighted_score, dim=1) / px_safe  # (M,2)
        
        # Set score to zero where original px was zero (points with zero probability)
        # zero_mask = (px == 0).squeeze(-1)  # (M,)
        # score_flat[zero_mask] = 0.0
        
        # breakpoint()
        return score_flat.view(*lead, 2)

    def sample(self, n_samples):
        """
        Sample points from the mixture in parallel using PyTorch.

        Args:
            n_samples: int, number of samples to draw.
        Returns:
            Tensor of shape (n_samples, 2) with samples.
        """
        indices = torch.multinomial(self.weights, num_samples=n_samples, replacement=True)
        z = torch.randn((n_samples, 2), device=self.device)
        chosen_chol = self.chol_covs[indices]
        chosen_means = self.mus[indices]
        offsets = torch.matmul(chosen_chol, z.unsqueeze(-1)).squeeze(-1)
        samples = chosen_means + offsets
        return samples, indices

    def evaluate_samples(self, samples):
        """
        Evaluate a batch of samples under the mixture model.

        Args:
            samples: torch.Tensor of shape (M, 2)
        Returns:
            grouped: list of length N, each element is a tensor of samples assigned to that mode
            mode_weights: torch.Tensor of shape (N,), fractions of samples per mode
            avg_log_likelihood: float, average log p(x) over samples
            mode_variances: torch.Tensor of shape (N,), isotropic variance (average of diag) per mode
        """
        pts = samples.to(self.device).float()
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("samples must have shape (M, 2)")
        M = pts.shape[0]
        # Assign to closest mode
        with torch.no_grad():
            dists = torch.cdist(pts, self.mus)  # (M, N)
            assignments = torch.argmin(dists, dim=1)  # (M,)
        # Grouped samples
        grouped = []
        mode_counts = torch.zeros(self.N, device=self.device)
        mode_variances = []
        for i in range(self.N):
            group = pts[assignments == i]
            grouped.append(group)
            mode_counts[i] = group.shape[0]
            if group.shape[0] > 1:
                mean = group.mean(dim=0, keepdim=True)
                centered = group - mean
                cov = centered.T @ centered / (group.shape[0] - 1)
                var_iso = torch.mean(torch.diag(cov))
            else:
                var_iso = torch.tensor(0.0, device=self.device)
            mode_variances.append(var_iso)
        # Mode weights by count
        mode_weights = mode_counts / M
        mode_variances = torch.stack(mode_variances, dim=0)  # (N,)
        # Average log likelihood
        pdf_vals = self.pdf(pts)  # (M,)
        # Avoid log(0): add small eps
        eps = 1e-12
        log_lik = torch.log(pdf_vals + eps)
        avg_log_likelihood = log_lik.mean().item()
        return grouped, mode_weights, mode_variances, avg_log_likelihood 
    
    def plot_density(self, k=1.0, num_points=100, save_path=None, cmap='viridis', 
                     num_levels=50, ax=None, xy_lims=None, vlims=[None, None]):
        """
        Plot the density as a 2D contourf plot with gradual color changes.
        After computing pdf_vals, assign each cell to the closest mode and compute mode weights
        as the sum of pdf over cells assigned to each mode. Title includes k, mode weights, and timestep.
        If ax is provided, plot into that axis; otherwise create a new figure.
        Also compute approximate scalar variance per mode based on grid cell assignments.
        """
        # Determine plotting range: cover means ± 3 * sqrt(max eigenvalue) for each cov
        max_radius = 0.0
        for i in range(self.N):
            eigvals = torch.linalg.eigvalsh(self.covariances[i])
            max_radius = max(max_radius, float(torch.max(torch.sqrt(eigvals)).item()))

        if xy_lims is None:
            x_min = float(torch.min(self.mus[:, 0]).item()) - 3 * max_radius
            x_max = float(torch.max(self.mus[:, 0]).item()) + 3 * max_radius
            y_min = float(torch.min(self.mus[:, 1]).item()) - 3 * max_radius
            y_max = float(torch.max(self.mus[:, 1]).item()) + 3 * max_radius
        else:
            x_min, x_max, y_min, y_max = xy_lims

        # Create grid on device
        x = torch.linspace(x_min, x_max, num_points, device=self.device)
        y = torch.linspace(y_min, y_max, num_points, device=self.device)
        xx, yy = torch.meshgrid(x, y, indexing='xy')  # (num_points, num_points)
        grid_points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # (num_points^2, 2)

        # Compute pdf on grid
        pdf_vals = self.pdf(grid_points)  # (num_points^2,)
        if k != 1.0:
            orig_pdf_sum = pdf_vals.sum()
            pdf_vals = pdf_vals ** k
            scaled_pdf_sum = pdf_vals.sum()
            pdf_vals = pdf_vals * (orig_pdf_sum / scaled_pdf_sum)  # scale to keep total mass same

        pdf_grid = pdf_vals.reshape(num_points, num_points)

        # Assign each grid cell to closest gaussian mode (Euclidean)
        with torch.no_grad():
            dists = torch.cdist(grid_points, self.mus)  # (num_cells, N)
            assignments = torch.argmin(dists, dim=1)  # (num_cells,)
            mode_weights = torch.zeros(self.N, device=self.device)
            mode_vars = torch.zeros(self.N, device=self.device)
            total_pdf = pdf_vals.sum()
            for i in range(self.N):
                mask = assignments == i
                mode_weights[i] = pdf_vals[mask].sum()
                if mask.sum() > 0:
                    centered = grid_points[mask] - self.mus[i]  # (num_cells_i, 2)
                    sq_dist = (centered ** 2).sum(dim=1)  # (num_cells_i,)
                    weighted_var = (sq_dist * pdf_vals[mask]).sum() / pdf_vals[mask].sum()
                    mode_vars[i] = weighted_var / 2  # isotropic: mean of x^2 and y^2
            if total_pdf > 0:
                mode_weights = mode_weights / total_pdf
        mode_weights_np = mode_weights.cpu().numpy()
        mode_vars_np = mode_vars.cpu().numpy()

        # Prepare for plotting
        pdf_vals_np = pdf_grid.cpu().detach().numpy()
        x_np = x.cpu().numpy()
        y_np = y.cpu().numpy()

        # Setup axis
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots()
            created_fig = True
        cf = ax.contourf(x_np, y_np, pdf_vals_np, levels=num_levels, cmap=cmap, vmin=vlims[0], vmax=vlims[1])
        cs = ax.contour(x_np, y_np, pdf_vals_np, levels=int(num_levels/5), colors='black', linewidths=0.5, alpha=0.5, vmin=vlims[0], vmax=vlims[1])
        if created_fig:
            cbar = plt.colorbar(cf, ax=ax)
            cbar.set_label('Density')

        weights_str = ", ".join([f"{w:.2f}" for w in mode_weights_np])
        vars_str = ", ".join([f"{v:.2e}" for v in mode_vars_np])
        ax.set_title(f"t={self.timestep}, weights=[{weights_str}]")

        if save_path and created_fig:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)

        return mode_weights_np, mode_vars_np




# Example usage:
import math
if __name__ == "__main__":
    def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999):
        """
        Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
        (1-beta) over time from t = [0,1].

        Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
        to that part of the diffusion process.


        Args:
            num_diffusion_timesteps (`int`): the number of betas to produce.
            max_beta (`float`): the maximum beta to use; use values lower than 1 to
                        prevent singularities.

        Returns:
            betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
        """
        def alpha_bar(time_step):
            return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return torch.tensor(betas, dtype=torch.float32)


    # Define mixture: some isotropic, some full-covariance components
    mus = [[-1.0, 0.0], [1.0, 0.0]]
    variances = [0.1, 0.1]  # mix scalar and 2x2
    weights = [2, 1]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gm = GaussianMixture2D(mus, variances, weights, device=device)

    # ====================================
    # k = 2.0
    # # Plot density contour
    # gm.plot_density(num_points=200, save_path=f"./gm_density_plots_test/t0_k1.png")
    # gm.plot_density(k=k, num_points=200, save_path=f"./gm_density_plots_test/t0_k{k}.png")
    # # noise scheduling
    

    # # Define the noise schedule parameters
    # T = 1000
    # # cosine schedule
    # beta = betas_for_alpha_bar(T)
    # alpha = 1.0 - beta
    # alpha_bar = torch.cumprod(alpha, dim=0)
    # t = 200
    # gm_t = gm.get_diffused_distribution(beta, alpha, alpha_bar, t=t)
    # gm_t.plot_density(num_points=200, save_path=f"./gm_density_plots_test/t{t}_k1.png")
    # gm_t.plot_density(k=k, num_points=200, save_path=f"./gm_density_plots_test/t{t}_k{k}.png")
    # ====================================


    # diffusion schedule example
    k = 2.0
    T = 1000
    beta = betas_for_alpha_bar(T)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    # timesteps 0 to 1000 step 100 -> indices 0,100,...,900, but 1000 out of range so upto 900
    ts = list(range(0, T, 100))
    n = len(ts)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    for ax, t in zip(axes, ts):
        gm_t = gm.get_diffused_distribution(beta, alpha, alpha_bar, t=t)
        gm_t.plot_density(k=k, num_points=100, cmap='viridis', num_levels=30, ax=ax)
    plt.tight_layout()
    plt.savefig(f"./gm_density_plots_test/k{k}_all_timesteps_density.png")
    plt.show()

