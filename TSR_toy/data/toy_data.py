# ### Evaluation code

# Given a batch of samples x_t, we want to:

# 1. compute weights of each mode under gm_model_t
# 2. compute the density plot
# 3. (optional) compute the per-mode variance
# 4. (optional) compute KL-divergence wtih GT x_t distribution
import torch
import matplotlib.pyplot as plt
from .gaussian_mixtures import GaussianMixture2D, GaussianMixture1D

def get_3gm_data(num_data_points, var=0.05, weights=[1, 1, 1], device=None, **kwargs):
    mode_distance = 1.0
    mus = [[-mode_distance/2, mode_distance/2], [-mode_distance/2, -mode_distance/2], [mode_distance/2, -mode_distance/2]]
    variances = [var for _ in range(len(mus))]  # Variances for each mode

    initweight_str = "".join([f"{w:.0f}" for w in weights])
    print(f"initial weights: {initweight_str}")
    gm = GaussianMixture2D(mus, variances, weights, device=device)
    data, labels = gm.sample(num_data_points)

    x_min = -1.5
    x_max = 1.5
    y_min = -1.5
    y_max = 1.5
    xy_lims = [x_min, x_max, y_min, y_max]

    return data, labels, xy_lims, initweight_str, gm

def get_thin_3gm_data(num_data_points, var=0.02, weights=[1, 1, 1], device=None, **kwargs):
    mus = [[-1, 0],[0, 0], [1, 0]]
    variances = [var for _ in range(len(mus))] 

    initweight_str = "".join([f"{w:.0f}" for w in weights])
    print(f"initial weights: {initweight_str}")
    gm = GaussianMixture2D(mus, variances, weights, device=device)
    data_x, labels = gm.sample(num_data_points)[:, 0]

    y_min, y_max = -0.01, 0.01
    data_y = torch.rand((num_data_points,), device=device) * (y_max - y_min) + y_min  # Uniformly sample y in [y_min, y_max]
    data = torch.stack([data_x, data_y], dim=1)


    x_min = -2.0
    x_max = 2.0
    y_min = -0.1
    y_max = 0.1
    xy_lims = [x_min, x_max, y_min, y_max]

    return data, labels, xy_lims, initweight_str, gm

def get_any_gm_data(num_data_points, mus, variances, weights, limit_margin=5, device=None, **kwargs):
    if isinstance(variances, float):
        variances = [variances for _ in range(len(mus))]

    initweight_str = "".join([f"{w:.0f}" for w in weights])
    print(f"initial weights: {initweight_str}")
    
    gm = GaussianMixture2D(mus, variances, weights, device=device)
    data, labels = gm.sample(num_data_points)

    # Compute xy limits based on mus and variances (3 std dev range)
    stds = [var**0.5 for var in variances]
    x_min = min(mu[0] - limit_margin*std for mu, std in zip(mus, stds))
    x_max = max(mu[0] + limit_margin*std for mu, std in zip(mus, stds))
    y_min = min(mu[1] - limit_margin*std for mu, std in zip(mus, stds))
    y_max = max(mu[1] + limit_margin*std for mu, std in zip(mus, stds))
    xy_lims = [x_min, x_max, y_min, y_max]

    return data, labels, xy_lims, initweight_str, gm


def get_single_point_data(num_data_points, point=[1.0, 1.0], device=None, **kwargs):
    data = torch.tensor(point, device=device).expand(num_data_points, -1)


    x_min = point[0] - 0.5
    x_max = point[0] + 0.5
    y_min = point[1] - 0.5
    y_max = point[1] + 0.5
    xy_lims = [x_min, x_max, y_min, y_max]

    initweight_str = ""
    gm = None

    return data, xy_lims, initweight_str, gm

def get_checkerboard_data(num_data_points, num_squares=4, oversample_factor=1.0, device=None, **kwargs):
    """
    Generate checkerboard data within [-1, 1] x [-1, 1], sampled in batch.
    Points are uniformly sampled from 'black' squares in a checkerboard pattern.
    """
    side_len = 2.0 / num_squares  # Length of one square
    total_needed = num_data_points
    data_list = []

    while total_needed > 0:
        # Oversample in batch
        batch_size = int(total_needed * oversample_factor)
        xy = torch.rand((batch_size, 2), device=device) * 2 - 1  # uniform in [-1, 1]^2
        x, y = xy[:, 0], xy[:, 1]

        # Compute (i, j) indices of each square
        i = ((x + 1.0) / side_len).floor().long()
        j = ((y + 1.0) / side_len).floor().long()

        # Keep points where (i + j) % 2 == 0 (black squares)
        mask = ((i + j) % 2 == 0)
        accepted = xy[mask]

        # Collect as much as we need
        if accepted.size(0) > total_needed:
            accepted = accepted[:total_needed]

        data_list.append(accepted)
        total_needed -= accepted.size(0)

    data = torch.cat(data_list, dim=0)
    data = data * 2.0  # Scale to [-2, 2]^2

    x_min, x_max = -2.0, 2.0
    y_min, y_max = -2.0, 2.0
    xy_lims = [x_min, x_max, y_min, y_max]

    initweight_str = f"{num_squares}x{num_squares}"
    gm = None

    return data, xy_lims, initweight_str, gm

def get_swissroll_data(num_data_points, noise=0.0, curly_factor=1.0, device=None, **kwargs):
    """
    Generate Swiss roll data in 2D (2D projection of a 3D Swiss roll).
    - 'curly_factor' controls how curly the roll is (number of rotations).
    - Sampling is adjusted so density is more uniform as t increases.
    """
    total_needed = num_data_points
    data_list = []

    base_min_t = torch.pi
    base_max_t = 4 * torch.pi
    min_t = base_min_t
    max_t = base_min_t + (base_max_t - base_min_t) * curly_factor

    while total_needed > 0:
        batch_size = int(total_needed * 1.2)

        # Sample t with increasing density: t ∝ sqrt(u)
        u = torch.rand(batch_size, device=device)
        t = torch.sqrt(u) * (max_t - min_t) + min_t

        x = t * torch.cos(t)
        y = t * torch.sin(t)
        xy = torch.stack([x, y], dim=1)

        if noise > 0:
            xy += torch.randn_like(xy) * noise

        if xy.size(0) > total_needed:
            xy = xy[:total_needed]

        data_list.append(xy)
        total_needed -= xy.size(0)

    data = torch.cat(data_list, dim=0)

    # Normalize to [-1, 1]^2
    data_max = data.abs().max()
    data = data * 2 / data_max

    xy_lims = [-2.0, 2.0, -2.0, 2.0]
    initweight_str = f"swissroll_curl={curly_factor:.2f}"
    gm = None

    return data, xy_lims, initweight_str, gm

def get_any_1d_gm_data(num_data_points, mus, variances, weights, limit_margin=5, device=None, **kwargs):
    """
    Generate 1D Gaussian mixture data with specified parameters.
    
    Args:
        num_data_points: int, number of samples to generate
        mus: List of scalars, means of each component
        variances: List of scalars or single scalar, variances of each component
        weights: List of scalars, mixing weights for each component
        limit_margin: float, multiplier for extending the plot range beyond std dev
        device: torch device for computation
        
    Returns:
        data: torch.Tensor of shape (num_data_points,), 1D samples
        labels: torch.Tensor of component assignments
        x_lims: List [x_min, x_max] for plotting limits
        initweight_str: String representation of weights
        gm: GaussianMixture1D instance
    """
    if isinstance(variances, (int, float)):
        variances = [variances for _ in range(len(mus))]
    
    if len(variances) != len(mus):
        raise ValueError("variances must have same length as mus or be a single scalar")
    
    initweight_str = "".join([f"{w:.0f}" for w in weights])
    print(f"initial weights: {initweight_str}")
    
    gm = GaussianMixture1D(mus, variances, weights, device=device)
    data, labels = gm.sample(num_data_points)
    
    # Compute x limits based on mus and variances (limit_margin * std dev range)
    stds = [var**0.5 for var in variances]
    x_min = min(mu - limit_margin*std for mu, std in zip(mus, stds))
    x_max = max(mu + limit_margin*std for mu, std in zip(mus, stds))
    x_lims = [x_min, x_max]
    
    return data, labels, x_lims, initweight_str, gm

def plot_1d_sample_density(samples, ax=None, num_points=1000, bandwidth=None,
                          margin=3.0, x_lims=None, y_lims=None, **kwargs):
    """
    Estimate and plot the 1D density of given samples via Gaussian KDE.

    Args:
        samples: torch.Tensor or array-like of shape (M,) or (M, 1). The 1D points.
        ax:      matplotlib Axes or None. If provided, plot into this axis; 
                 otherwise, a new figure+axis is created.
        num_points: int. Resolution of grid along x-axis.
        bandwidth: float or None. KDE bandwidth h. If None, use a rule-of-thumb:
            h = std * M^(-1/5) for 1D data.
        margin:  float. Multiplier for extending the plot range beyond sample std.
        x_lims:  Optional (x_min, x_max) tuple.
        y_lims:  Optional (y_min, y_max) tuple for y-axis limits.

    Returns:
        ax: the matplotlib Axes containing the plot.
    """
    # Convert samples to torch.Tensor on CPU (or GPU if desired)
    if not torch.is_tensor(samples):
        pts = torch.tensor(samples, dtype=torch.float32)
    else:
        pts = samples.detach().cpu().float()

    # Handle both (M,) and (M, 1) shapes
    if pts.ndim == 2 and pts.shape[1] == 1:
        pts = pts.squeeze(-1)
    elif pts.ndim != 1:
        raise ValueError("samples must have shape (M,) or (M, 1)")
    
    M = pts.shape[0]
    if M == 0:
        raise ValueError("samples must contain at least one point")

    # Compute bounding box: mean ± margin * std
    mean = torch.mean(pts)
    std = torch.std(pts)
    # If std is zero (all samples identical), set small positive
    if std < 1e-6:
        std = torch.tensor(1e-6)

    if x_lims is None:
        x_min = (mean - margin * std).item()
        x_max = (mean + margin * std).item()
    else:
        x_min, x_max = x_lims

    # Determine device for computation
    device = pts.device

    # Bandwidth selection if None: rule-of-thumb for 1D
    if bandwidth is None:
        # Silverman-like: h ~ std * M^{-1/(d+4)}, here d=1 → exponent = -1/5
        h = float(std.item()) * (M ** (-1/5))
        # Avoid extremely small h
        if h <= 0:
            h = 1e-3
    else:
        h = float(bandwidth)
    h_tensor = torch.tensor(h, device=device, dtype=torch.float32)

    # Create grid
    x = torch.linspace(x_min, x_max, num_points, device=device)
    
    # Compute pairwise squared distances between grid points and samples
    # For 1D: |x_i - x_j|^2
    dists = torch.abs(x.unsqueeze(1) - pts.unsqueeze(0))  # (num_points, M)
    sq_dists = dists.pow(2)  # (num_points, M)

    # Compute KDE: Gaussian kernel
    # K(x) = (1/(sqrt(2π) h)) * exp(-||x - xi||^2 / (2 h^2))
    # density at grid point j: (1/M) * sum_{i=1}^M K(x_j, x_i)
    exp_term = torch.exp(-0.5 * sq_dists / (h_tensor ** 2))  # (num_points, M)
    coef = 1.0 / (torch.sqrt(torch.tensor(2 * torch.pi)) * h_tensor)  # scalar
    # Sum over samples:
    density_vals = coef * exp_term.mean(dim=1)  # (num_points,)

    # Prepare plotting
    x_np = x.cpu().numpy()
    density_np = density_vals.cpu().detach().numpy()

    # Setup axis
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        created_fig = True

    # Plot density as line and fill
    ax.plot(x_np, density_np, 'b-', linewidth=2, **kwargs)
    ax.fill_between(x_np, density_np, alpha=0.3, **kwargs)
    
    ax.set_xlabel('x')
    ax.set_ylabel('Density')
    ax.set_title(f'1D Sample KDE density (M={M}, h={h:.3f})')
    ax.grid(True, alpha=0.3)
    
    # Set y-axis limits if provided
    if y_lims is not None:
        ax.set_ylim(y_lims)

    if created_fig:
        plt.tight_layout()

    return ax


def create_dataset(config, device):
    """Create dataset based on configuration."""
    dataset_name = config['dataset_name']
    num_data_points = config['num_data_points']
    
    if dataset_name == 'checkerboard':
        params = config['checkerboard_params']
        data, xy_lims, initweight_str, gm = get_checkerboard_data(
            num_data_points=num_data_points,
            num_squares=params['num_squares'],
            device=device
        )
        labels = torch.zeros(data.shape[0], dtype=torch.long, device=device)
        config['data_dim'] = 2
        return data, labels, xy_lims, initweight_str, gm
    elif dataset_name == 'swissroll':
        params = config['swissroll_params']
        data, xy_lims, initweight_str, gm = get_swissroll_data(
            num_data_points=num_data_points,
            curly_factor=params['curly_factor'],
            noise=params['noise'],
            device=device
        )
        labels = torch.zeros(data.shape[0], dtype=torch.long, device=device)
        config['data_dim'] = 2
        return data, labels, xy_lims, initweight_str, gm
    elif dataset_name == '3gm':
        data, labels, xy_lims, initweight_str, gm = get_3gm_data(
            num_data_points=num_data_points,
            device=device
        )
        config['data_dim'] = 2
        return data, labels, xy_lims, initweight_str, gm
    elif dataset_name == 'thin_3gm':
        data, labels, xy_lims, initweight_str, gm = get_thin_3gm_data(
            num_data_points=num_data_points,
            device=device
        )
        config['data_dim'] = 2
        return data, labels, xy_lims, initweight_str, gm
    elif dataset_name == '6gm':
        params = config['6gm_params']
        data, labels, xy_lims, initweight_str, gm = get_any_gm_data(
            num_data_points=num_data_points,
            n_components=params['n_components'],
            mus=params['means'],
            variances=params['variances'],
            weights=params['weights'],
            device=device
        )
        labels = torch.where(labels <= 2, 0, labels)
        labels = torch.where(labels >= 3, 1, labels)
        config['data_dim'] = 2
        return data, labels, xy_lims, initweight_str, gm
    elif dataset_name == '6gm_horizontal':
        params = config['6gm_horizontal_params']
        data, labels, xy_lims, initweight_str, gm = get_any_gm_data(
            num_data_points=num_data_points,
            n_components=params['n_components'],
            mus=params['means'],
            variances=params['variances'],
            weights=params['weights'],
            device=device
        )
        labels = torch.where(labels <= 2, 0, labels)
        labels = torch.where(labels >= 3, 1, labels)
        config['data_dim'] = 2
        return data, labels, xy_lims, initweight_str, gm
    elif dataset_name == '1d_gm':
        params = config['1d_gm_params']
        data, labels, x_lims, initweight_str, gm = get_any_1d_gm_data(
            num_data_points=num_data_points,
            mus=params['mus'],
            variances=params['variances'],
            weights=params['weights'],
            device=device
        )
        # Convert 1D data to have shape (N, 1) for compatibility
        data = data.unsqueeze(-1)
        config['data_dim'] = 1
        # For 1D data, use x_lims as xy_lims with dummy y limits
        xy_lims = [x_lims[0], x_lims[1], -0.1, 0.1]
        return data, labels, xy_lims, initweight_str, gm
    elif dataset_name == 'single_point':
        params = config['single_point_params']
        data, xy_lims, initweight_str, gm = get_single_point_data(
            num_data_points=num_data_points,
            point=params['point'],
            device=device
        )
        labels = torch.zeros(data.shape[0], dtype=torch.long, device=device)
        config['data_dim'] = 2
        return data, labels, xy_lims, initweight_str, gm
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")