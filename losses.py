import torch 
from torch.nn import functional as F

import commons


def feature_loss(fmap_r, fmap_g):
  loss = 0
  for dr, dg in zip(fmap_r, fmap_g):
    for rl, gl in zip(dr, dg):
      rl = rl.float().detach()
      gl = gl.float()
      loss += torch.mean(torch.abs(rl - gl))

  return loss * 2 


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
  loss = 0
  r_losses = []
  g_losses = []
  for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
    dr = dr.float()
    dg = dg.float()
    r_loss = torch.mean((1-dr)**2)
    g_loss = torch.mean(dg**2)
    loss += (r_loss + g_loss)
    r_losses.append(r_loss.item())
    g_losses.append(g_loss.item())

  return loss, r_losses, g_losses


def generator_loss(disc_outputs):
  loss = 0
  gen_losses = []
  for dg in disc_outputs:
    dg = dg.float()
    l = torch.mean((1-dg)**2)
    gen_losses.append(l)
    loss += l

  return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask, multi_samples = False):
  """
  z_p, logs_q: [b, h, t_t]
  m_p, logs_p: [b, h, t_t]
  """
  z_p = z_p.float()
  logs_q = logs_q.float()
  m_p = m_p.float()
  logs_p = logs_p.float()
  z_mask = z_mask.float()
  
  print("logs_p.shape",logs_p.shape)
  print("logs_q.shape",logs_q.shape)
  

  if not multi_samples:
    kl = logs_p - logs_q - 0.5
    # Here, z_p should be the mean, why not use m_q?
    # It's measuring the difference bewteen z' and mu_p, logs_p
    kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)
  else:
    # Assuming z is your tensor with size [N, a, b, c]
    N = z_p.size(0)  # Get the number of samples dynamically
    
    #  Reshape z to have samples in the first dimension
    z_reshaped = z_p.view(N, -1)  # Shape: [N, a*b*c]

    # Compute mean along the first dimension (samples)
    m_estimate = torch.mean(z_reshaped, dim=0)

    # Compute centered data
    z_centered = z_reshaped - m_estimate.unsqueeze(0)

    # Compute covariance matrix
    covariance_matrix = torch.matmul(z_centered.transpose(0, 1), z_centered) / N  # Normalize by number of samples
    
    # Rreshape it back to the original shape
    dim_product = z_reshaped.size(1)
    covariance_matrix = covariance_matrix.view(dim_product, dim_product)
    std_dev = torch.sqrt(torch.diag(covariance_matrix))
    logs_estimate = torch.log(std_dev)

    print("cov.shape",covariance_matrix.shape)
    print("logs_estimate.shape",logs_estimate.shape)
    kl = logs_p - logs_estimate - 0.5
    kl += 0.5 * ((m_estimate - m_p)**2) * torch.exp(-2. * logs_p)

  
  kl = torch.sum(kl * z_mask)
  l = kl / torch.sum(z_mask)
  return l
