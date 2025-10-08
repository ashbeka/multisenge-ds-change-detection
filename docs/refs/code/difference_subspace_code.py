import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict


class SubspaceAnalysis(nn.Module):
    """
    Compute canonical angles, difference subspace, and principal component subspace 
    between two subspaces extracted from data matrices using PCA.
    """
    
    def __init__(self, n_components: Optional[int] = None, center: bool = True):
        """
        Args:
            n_components: Number of PCA components. If None, determined from data.
            center: Whether to center data before PCA.
        """
        super().__init__()
        self.n_components = n_components
        self.center = center
        
    def extract_subspace(self, X: torch.Tensor, n_components: Optional[int] = None) -> torch.Tensor:
        """Extract orthonormal subspace basis using PCA."""
        n_components = n_components or self.n_components or min(X.shape)
        
        # Use torch.pca_lowrank for efficient PCA
        U, S, V = torch.pca_lowrank(X, q=n_components, center=self.center, niter=2)
        
        # V contains the principal components (orthonormal basis)
        return V
    
    def forward(self, X1: torch.Tensor, X2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute subspace analysis between two data matrices.
        
        Args:
            X1: First data matrix (n_samples1, n_features)
            X2: Second data matrix (n_samples2, n_features)
            
        Returns:
            Dictionary with keys:
                - 'Phi', 'Psi': Orthonormal bases of subspaces S1 and S2
                - 'theta': Canonical angles
                - 'D': Orthonormal basis of difference subspace
                - 'M': Orthonormal basis of principal component subspace (Karcher mean)
                - 'U', 'Sigma', 'V': SVD components
        """
        # Extract PCA subspaces with same dimensionality
        Phi = self.extract_subspace(X1)
        Psi = self.extract_subspace(X2)
        
        # Compute canonical angles via SVD: Phi^T @ Psi = U @ Sigma @ V^T
        U, Sigma, Vt = torch.linalg.svd(Phi.T @ Psi, full_matrices=False)
        V = Vt.T
        
        # Clamp for numerical stability and compute angles
        Sigma = Sigma.clamp(-1, 1)
        theta = torch.acos(Sigma)
        
        # Compute difference subspace: D = (Phi@U - Psi@V) @ {2(I - Sigma)}^(-1/2)
        diff = Phi @ U - Psi @ V
        d = Sigma.shape[0]
        I = torch.eye(d, device=Sigma.device, dtype=Sigma.dtype)
        D_scale = torch.linalg.matrix_power(2 * (I - torch.diag(Sigma)), -1/2)
        D = diff @ D_scale
        
        # Compute principal component subspace: M = (Phi@U + Psi@V) @ {2(I + Sigma)}^(-1/2}
        sum_vectors = Phi @ U + Psi @ V
        M_scale = torch.linalg.matrix_power(2 * (I + torch.diag(Sigma)), -1/2)
        M = sum_vectors @ M_scale
        
        return {
            'Phi': Phi,
            'Psi': Psi,
            'theta': theta,
            'D': D,
            'M': M,
            'U': U,
            'Sigma': Sigma,
            'V': V
        }


# Example usage
if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Generate sample data
    X1 = torch.randn(100, 20)
    X2 = torch.randn(100, 20)
    
    # Analyze subspaces
    analyzer = SubspaceAnalysis(n_components=5)
    results = analyzer(X1, X2)
    
    # Display results
    print(f"Canonical angles (degrees): {results['theta'].rad2deg()}")
    print(f"Difference subspace D shape: {results['D'].shape}")
    print(f"Principal component subspace M shape: {results['M'].shape}")
    
    # Verify orthonormality
    d = results['D'].shape[1]
    print(f"D orthonormal: {torch.allclose(results['D'].T @ results['D'], torch.eye(d), atol=1e-5)}")
    print(f"M orthonormal: {torch.allclose(results['M'].T @ results['M'], torch.eye(d), atol=1e-5)}")