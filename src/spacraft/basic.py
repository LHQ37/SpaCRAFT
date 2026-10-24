"""
Feature selection functions adapted from SnapATAC2 for SpaCRAFT
"""

import numpy as np
from pathlib import Path
from typing import Union, Optional, List
import anndata
from scipy.sparse import issparse
import warnings


def _find_most_accessible_features(
    feature_count: np.ndarray,
    filter_lower_quantile: float,
    filter_upper_quantile: float,
    total_features: int,
) -> np.ndarray:
    """
    Find the most accessible features based on their count.
    
    Parameters
    ----------
    feature_count : np.ndarray
        Count of each feature across all cells
    filter_lower_quantile : float
        Lower quantile to filter out (e.g., 0.005 = bottom 0.5%)
    filter_upper_quantile : float
        Upper quantile to filter out (e.g., 0.005 = top 0.5%)
    total_features : int
        Number of features to select
        
    Returns
    -------
    np.ndarray
        Indices of selected features
    """
    # Sort features by count
    idx = np.argsort(feature_count)
    
    # Remove zero-count features
    for i in range(idx.size):
        if feature_count[idx[i]] > 0:
            break
    idx = idx[i:]
    
    # Apply quantile filtering
    n = idx.size
    n_lower = int(filter_lower_quantile * n)
    n_upper = int(filter_upper_quantile * n)
    idx = idx[n_lower:n-n_upper]
    
    # Return top features (descending order)
    return idx[::-1][:total_features]


def select_features(
    adata: Union[anndata.AnnData, List[anndata.AnnData]],
    n_features: int = 500000,
    filter_lower_quantile: float = 0.005,
    filter_upper_quantile: float = 0.005,
    inplace: bool = True,
    verbose: bool = True,
) -> Optional[np.ndarray]:
    """
    Perform feature selection by selecting the most accessible features across all cells.
    
    This function is adapted from SnapATAC2 for use with standard AnnData objects.
    It selects features based on their total count (accessibility) across all cells.

    Parameters
    ----------
    adata : anndata.AnnData or list of anndata.AnnData
        The (annotated) data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to regions/features.
        Can also be a list of AnnData objects (processed sequentially).
    n_features : int, default=500000
        Number of features to keep. Note that the final number of features
        may be smaller if there are not enough features passing the filtering criteria.
    filter_lower_quantile : float, default=0.005
        Lower quantile of the feature count distribution to filter out.
        For example, 0.005 means the bottom 0.5% features with the lowest counts will be removed.
    filter_upper_quantile : float, default=0.005
        Upper quantile of the feature count distribution to filter out.
        For example, 0.005 means the top 0.5% features with the highest counts will be removed.
    inplace : bool, default=True
        If True, add the feature mask to `.var['selected']`.
        If False, return the boolean mask.
    verbose : bool, default=True
        Whether to print progress messages.
    
    Returns
    -------
    np.ndarray or None
        If `inplace = False`, return a boolean index mask where `True` means the feature is kept.
        If `inplace = True`, return None and store the mask in `.var['selected']`.
        
    Examples
    --------
    >>> import anndata as ad
    >>> from spacraft.basic import select_features
    >>> adata = ad.read_h5ad('data.h5ad')
    >>> select_features(adata, n_features=50000)
    >>> # Selected features are marked in adata.var['selected']
    """
    # Handle list of AnnData objects
    if isinstance(adata, list):
        if verbose:
            print(f"Processing {len(adata)} AnnData objects...")
        
        results = []
        for i, ad_obj in enumerate(adata):
            if verbose:
                print(f"  Processing object {i+1}/{len(adata)}...")
            result = select_features(
                ad_obj, n_features, filter_lower_quantile,
                filter_upper_quantile, inplace, verbose=False
            )
            if not inplace:
                results.append(result)
        
        if inplace:
            return None
        else:
            return results
    
    # Calculate feature counts
    if verbose:
        print(f"Calculating feature counts for {adata.shape[1]} features...")
    
    if issparse(adata.X):
        # For sparse matrices
        count = np.array(adata.X.sum(axis=0)).flatten()
    else:
        # For dense matrices
        count = np.sum(adata.X, axis=0)
    
    if inplace:
        adata.var['count'] = count
    
    # Select most accessible features
    selected_features = _find_most_accessible_features(
        count, filter_lower_quantile, filter_upper_quantile, n_features
    )
    
    # Create boolean mask
    result = np.zeros(adata.shape[1], dtype=bool)
    result[selected_features] = True
    
    if verbose:
        print(f"Selected {result.sum()} features out of {adata.shape[1]} total features.")
        print(f"  Mean count of selected features: {count[result].mean():.2f}")
        print(f"  Min count of selected features: {count[result].min():.2f}")
        print(f"  Max count of selected features: {count[result].max():.2f}")
    
    if inplace:
        adata.var["selected"] = result
        return None
    else:
        return result


def apply_feature_selection(
    adata: anndata.AnnData,
    inplace: bool = True,
    verbose: bool = True,
) -> Optional[anndata.AnnData]:
    """
    Apply previously computed feature selection mask to subset the data.
    
    This function subsets the AnnData object to only include features where
    `.var['selected']` is True.
    
    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object with `.var['selected']` column
    inplace : bool, default=True
        If True, subset the data in place.
        If False, return a new subset AnnData object.
    verbose : bool, default=True
        Whether to print progress messages.
        
    Returns
    -------
    anndata.AnnData or None
        If `inplace = False`, return the subset AnnData.
        If `inplace = True`, return None.
        
    Examples
    --------
    >>> import anndata as ad
    >>> from spacraft.basic import select_features, apply_feature_selection
    >>> adata = ad.read_h5ad('data.h5ad')
    >>> select_features(adata, n_features=50000)
    >>> apply_feature_selection(adata)  # Now adata only has 50000 features
    """
    if 'selected' not in adata.var.columns:
        raise ValueError("No feature selection found. Please run `select_features()` first.")
    
    n_before = adata.shape[1]
    selected_mask = adata.var['selected'].values
    n_after = selected_mask.sum()
    
    if verbose:
        print(f"Applying feature selection: {n_before} → {n_after} features")
    
    if inplace:
        adata._inplace_subset_var(selected_mask)
        return None
    else:
        return adata[:, selected_mask].copy()
