import torch
import numpy as np
from einops import rearrange, repeat


def feature_pca(feature: torch.Tensor, dim: int = 3):
  """Converts feature of shape (N, f) into a lower-dim feature of shape (N, dim) using PCA.

  Args:
      feature (torch.Tensor): shape (N, f)
      dim (int, optional): The number of principal components to keep. Defaults to 3.
  Returns:
      torch.Tensor: shape (N, dim)
  """

  assert feature.dim() == 2, "Input feature must have shape (N, f)"
  N, f = feature.shape

  # Step 1: Center the data by subtracting the mean
  data_mean = torch.mean(feature, dim=0)
  data_centered = feature - data_mean

  # Step 2: Compute the covariance matrix
  # Since data is large, it's more efficient to compute using the covariance formula
  covariance_matrix = torch.matmul(data_centered.T,
                                   data_centered) / (data_centered.shape[0] - 1)

  # Ensure the covariance matrix is symmetric
  covariance_matrix = (covariance_matrix + covariance_matrix.T) / 2

  # Step 3: Perform PCA using torch.linalg.eigh (since covariance matrix is symmetric)
  eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)

  # Extract real parts in case of numerical issues
  eigenvalues = eigenvalues.real
  eigenvectors = eigenvectors.real

  # Sort the eigenvalues and eigenvectors in descending order
  idx = torch.argsort(eigenvalues, descending=True)
  eigenvalues = eigenvalues[idx]
  eigenvectors = eigenvectors[:, idx]

  # Select the top `dim` principal components
  principal_components = eigenvectors[:, :dim]  # Shape: (f, dim)

  # Step 4: Project the data onto the principal components
  projected_data = torch.matmul(data_centered,
                                principal_components)  # Shape: (N, dim)

  return projected_data


def feature_map_to_rgb(feature_map, dim=3, use_normalize=True):
  """
    Converts a feature map of shape (f, h, w) into an RGB map of shape (3, h, w) using PCA.

    Args:
        feature_map (torch.Tensor): The input feature map of shape (f, h, w).

    Returns:
        rgb_map (torch.Tensor): The output RGB map of shape (3, h, w).
    """

  # Ensure the feature map is a 3D tensor
  assert feature_map.dim() == 3, "Input feature map must have shape (f, h, w)"

  f, h, w = feature_map.shape

  # Step 1: Reshape the feature map to (h*w, f)
  data = feature_map.view(f, -1).T  # Shape: (h*w, f)

  projected_data = feature_pca(data, dim=dim)

  # Step 2: Reshape back to (3, h, w)
  rgb_map = projected_data.T.view(dim, h, w)

  if use_normalize:
    # Normalize the RGB map to [0, 1] for visualization
    rgb_min = torch.amin(rgb_map, dim=(1, 2), keepdim=True)
    rgb_max = torch.amax(rgb_map, dim=(1, 2), keepdim=True)
    rgb_map = (rgb_map - rgb_min) / (
        rgb_max - rgb_min + 1e-7)  # Adding epsilon to avoid division by zero

  return rgb_map


# def knn_index(a, b, k):
#   """Computes the k-nearest neighbors of each point in a with respect to b.

#   Args:
#       a (torch.Tensor): Tensor of shape (B, N, D) containing N points in D dimensions.
#       b (torch.Tensor): Tensor of shape (B, M, D) containing M points in D dimensions.
#       k (int): Number of nearest neighbors to return.

#   Returns:
#       torch.Tensor: Tensor of shape (B, N, k) containing the indices of the k-nearest neighbors in b.
#   """
#   a_expanded = a.unsqueeze(2)  # (B, N, 1, D)
#   b_expanded = b.unsqueeze(1)  # (B, 1, M, D)

#   # compute the squared Euclidean distance between each pair of points
#   dist_squared = torch.sum((a_expanded - b_expanded)**2, dim=3)  # (B, N, M)

#   # get the indices of the k-nearest neighbors
#   _, indices = torch.topk(dist_squared, k=k, dim=2, largest=False)

#   return indices


def knn_index(a, b, k):
  """Computes the k-nearest neighbors of each point in a with respect to b.

    Args:
        a (torch.Tensor): Tensor of shape (B, N, D) containing N points in D dimensions.
        b (torch.Tensor): Tensor of shape (B, M, D) containing M points in D dimensions.
        k (int): Number of nearest neighbors to return.

    Returns:
        torch.Tensor: Tensor of shape (B, N, k) containing the indices of the k-nearest neighbors in b.
    """
  # Squared norms of points in a and b
  a_squared = torch.sum(a**2, dim=2, keepdim=True)  # (B, N, 1)
  b_squared = torch.sum(b**2, dim=2, keepdim=True).transpose(1, 2)  # (B, 1, M)

  # Compute squared Euclidean distances using the expanded form (a-b)^2 = a^2 + b^2 - 2 * a * b
  dist_squared = a_squared + b_squared - 2 * torch.bmm(a, b.transpose(
      1, 2))  # (B, N, M)

  # Get the indices of the k-nearest neighbors
  _, indices = torch.topk(dist_squared, k=k, dim=2, largest=False)

  return indices


def knn_gather(b, indices):
  """Gathers the values of the k-nearest neighbors in b.

  Args:
      b (torch.Tensor): Tensor of shape (B, M, D) containing M points in D dimensions.
      indices (torch.Tensor): Tensor of shape (B, N, k) containing the indices of the k-nearest neighbors in b.

  Returns:
      torch.Tensor: Tensor of shape (B, N, k, D) containing the values of the k-nearest neighbors in b.
  """
  b_expanded = b.unsqueeze(1)  # (B, 1, M, D)
  indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1,
                                                  b.size(2))  # (B, N, k, D)

  values = torch.gather(b_expanded.expand(-1, indices.size(1), -1, -1), 2,
                        indices_expanded)

  return values


def knn(a, b, k):
  """
  Args:
      a (torch.Tensor): Tensor of shape (B, N, D) containing N points in D dimensions.
      b (torch.Tensor): Tensor of shape (B, M, D) containing M points in D dimensions.
      k (int): Number of nearest neighbors to return.
  Returns:
      torch.Tensor: Tensor of shape (B, N, k, D) containing the values of the k-nearest neighbors in b.
  """
  indices = knn_index(a, b, k)
  values = knn_gather(b, indices)
  return values


def random_downsample_pc(pc, m):
  """Randomly downsamples a point cloud to m points.

  Args:
      pc (torch.Tensor): Tensor of shape (N, 3) containing N points in 3D.
      m (int): Number of points to sample.

  Returns:
      torch.Tensor: Tensor of shape (m, 3) containing the downsampled points.
  """
  N = pc.size(0)
  indices = torch.randperm(N)[:m]
  return pc[indices]


def value2HSV2RGB(value):
  """Converts a value to HSV and then to RGB.

    Args:
        value (torch.Tensor): Tensor of shape (N,) containing N values between 0 and 1.

    Returns:
        torch.Tensor: Tensor of shape (N, 3) containing the RGB values.
    """
  N = value.shape[0]
  device = value.device

  # HSV values
  h = value * 360  # Convert to degrees (0 to 360)
  s = torch.ones(N, device=device)  # Saturation set to 1
  v = torch.ones(N, device=device)  # Value set to 1

  # Normalize hue to [0, 360)
  h = h % 360

  # Convert HSV to RGB
  c = v * s  # Chroma
  h_prime = h / 60  # Sector of the color wheel
  x = c * (1 - torch.abs(h_prime % 2 - 1))
  m = v - c

  r = torch.zeros(N, device=device)
  g = torch.zeros(N, device=device)
  b = torch.zeros(N, device=device)

  # Conditions for different sectors
  cond = (h_prime >= 0) & (h_prime < 1)
  r[cond] = c[cond]
  g[cond] = x[cond]
  b[cond] = 0

  cond = (h_prime >= 1) & (h_prime < 2)
  r[cond] = x[cond]
  g[cond] = c[cond]
  b[cond] = 0

  cond = (h_prime >= 2) & (h_prime < 3)
  r[cond] = 0
  g[cond] = c[cond]
  b[cond] = x[cond]

  cond = (h_prime >= 3) & (h_prime < 4)
  r[cond] = 0
  g[cond] = x[cond]
  b[cond] = c[cond]

  cond = (h_prime >= 4) & (h_prime < 5)
  r[cond] = x[cond]
  g[cond] = 0
  b[cond] = c[cond]

  cond = (h_prime >= 5) & (h_prime < 6)
  r[cond] = c[cond]
  g[cond] = 0
  b[cond] = x[cond]

  # Add m to match the value
  r = r + m
  g = g + m
  b = b + m

  rgb = torch.stack([r, g, b], dim=1)  # Shape: (N, 3)

  return rgb


def to_sparse(x: torch.BoolTensor) -> torch.sparse.FloatTensor:
  """
    Converts a BoolTensor of arbitrary shape to a sparse tensor representation.

    Parameters:
        x (torch.BoolTensor): A BoolTensor of arbitrary shape.

    Returns:
        sparse_tensor (torch.sparse.FloatTensor): The sparse tensor representation.
    """
  # Get indices of elements that are True
  indices = torch.nonzero(x, as_tuple=False).T

  indices_dtype = torch.uint64
  max_shape = max(x.shape)
  if max_shape <= torch.iinfo(torch.uint8).max:
    indices_dtype = torch.uint8
  elif max_shape <= torch.iinfo(torch.uint16).max:
    indices_dtype = torch.uint16
  elif max_shape <= torch.iinfo(torch.uint32).max:
    indices_dtype = torch.uint32
  else:
    indices_dtype = torch.uint64
  indices = indices.to(dtype=indices_dtype)
  # Create a tensor with all values set to 1
  values = torch.ones(indices.shape[1], dtype=torch.bool, device=x.device)
  # Create the sparse tensor
  return torch.sparse_coo_tensor(indices,
                                 values,
                                 size=x.shape,
                                 dtype=torch.bool,
                                 device=x.device)


def from_sparse(sparse_tensor: torch.sparse.FloatTensor) -> torch.BoolTensor:
  """
    Restores the original BoolTensor from its sparse tensor representation.
    [NOTE] Recommended to use sparse_tensor.to_dense() instead. This function is for demonstration purposes only.

    Parameters:
        sparse_tensor (torch.sparse.FloatTensor): The sparse tensor representation.

    Returns:
        x (torch.BoolTensor): The recovered BoolTensor.
    """
  # Convert the sparse tensor to a dense tensor
  return sparse_tensor.to_dense()
