import scipy
import numpy as np
import episcanpy as epi
import scanpy as sc
import pandas as pd

from typing import Optional
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


def preprocess_CAS(adata_list, adata_concat, binarize=False, use_fragment_count=False, tfidf=True,
                   min_cells_rate=0.03, min_features=1, tfidf_type=2):

    epi.pp.filter_features(adata_concat, min_cells=int(min_cells_rate * adata_concat.shape[0]))
    epi.pp.filter_cells(adata_concat, min_features=min_features)

    if binarize and use_fragment_count:
        raise ValueError("'binarize' and 'use_fragment_count' cannot be set to True at the same time !")

    elif binarize:
        epi.pp.binarize(adata_concat)

    elif use_fragment_count:
        adata_concat.X = scipy.sparse.csr_matrix(np.ceil((adata_concat.X / 2).toarray()))

    if tfidf:
        adata_concat.X = TFIDF(adata_concat.X.T, type_=tfidf_type).T.copy()
    else:
        epi.pp.normalize_total(adata_concat, target_sum=10000)
        epi.pp.log1p(adata_concat)

    for i in range(len(adata_list)):
        obs_list = [item for item in adata_list[i].obs_names if item in adata_concat.obs_names]
        var_names = adata_concat.var_names
        adata_list[i] = adata_list[i][obs_list, var_names]


def preprocess_SRT(adata_list, adata_concat, n_top_genes=5000, min_cells=1, min_genes=1):

    sc.pp.filter_genes(adata_concat, min_cells=min_cells)
    sc.pp.filter_cells(adata_concat, min_genes=min_genes)

    sc.pp.highly_variable_genes(adata_concat, flavor="seurat_v3", n_top_genes=n_top_genes)
    sc.pp.normalize_total(adata_concat, target_sum=1e4)
    sc.pp.log1p(adata_concat)
    adata_concat = adata_concat[:, adata_concat.var['highly_variable']]

    for i in range(len(adata_list)):
        obs_list = [item for item in adata_list[i].obs_names if item in adata_concat.obs_names]
        var_names = adata_concat.var_names
        adata_list[i] = adata_list[i][obs_list, var_names]

    return adata_list, adata_concat


def TFIDF(count_mat, type_=2):
    # Perform TF-IDF (count_mat: peak*cell)
    def tfidf1(count_mat):
        if not scipy.sparse.issparse(count_mat):
            count_mat = scipy.sparse.coo_matrix(count_mat)

        nfreqs = count_mat.multiply(1.0 / count_mat.sum(axis=0))
        tfidf_mat = nfreqs.multiply(np.log(1 + 1.0 * count_mat.shape[1] / count_mat.sum(axis=1)).reshape(-1, 1)).tocoo()

        return scipy.sparse.csr_matrix(tfidf_mat)

    # Perform Signac TF-IDF (count_mat: peak*cell) [selected]
    def tfidf2(count_mat):
        if not scipy.sparse.issparse(count_mat):
            count_mat = scipy.sparse.coo_matrix(count_mat)

        tf_mat = count_mat.multiply(1.0 / count_mat.sum(axis=0))
        signac_mat = (1e4 * tf_mat).multiply(1.0 * count_mat.shape[1] / count_mat.sum(axis=1).reshape(-1, 1))
        signac_mat = signac_mat.log1p()

        return scipy.sparse.csr_matrix(signac_mat)

    # Perform TF-IDF (count_mat: ?)
    from sklearn.feature_extraction.text import TfidfTransformer
    def tfidf3(count_mat):
        model = TfidfTransformer(smooth_idf=False, norm="l2")
        model = model.fit(np.transpose(count_mat))
        model.idf_ -= 1
        tf_idf = np.transpose(model.transform(np.transpose(count_mat)))

        return scipy.sparse.csr_matrix(tf_idf)

    if type_ == 1:
        return tfidf1(count_mat)
    elif type_ == 2:
        return tfidf2(count_mat)
    else:
        return tfidf3(count_mat)


def find_peak_overlaps(query, key):
    q_seqname = np.array(query.get_seqnames())
    k_seqname = np.array(key.get_seqnames())
    q_start = np.array(query.get_start())
    k_start = np.array(key.get_start())
    q_width = np.array(query.get_width())
    k_width = np.array(key.get_width())
    q_end = q_start + q_width
    k_end = k_start + k_width

    q_index = 0
    k_index = 0
    overlap_index = [[] for i in range(len(query))]
    overlap_count = [0 for i in range(len(query))]

    while True:
        if q_index == len(query) or k_index == len(key):
            return overlap_index, overlap_count

        if q_seqname[q_index] == k_seqname[k_index]:
            if k_start[k_index] >= q_start[q_index] and k_end[k_index] <= q_end[q_index]:
                overlap_index[q_index].append(k_index)
                overlap_count[q_index] += 1
                k_index += 1
            elif k_start[k_index] < q_start[q_index]:
                k_index += 1
            else:
                q_index += 1
        elif q_seqname[q_index] < k_seqname[k_index]:
            q_index += 1
        else:
            k_index += 1


def _parse_peak_region(region):
    """Parse peak names from common ATAC formats.

    Supported examples:
    - chr1:100-200
    - chr1-100-200
    - chr1_100_200
    - chr1:100:200
    """
    region = str(region).strip()
    if not region:
        raise ValueError('Peak 名称为空')

    if ':' in region and '-' in region:
        chrom, rest = region.split(':', 1)
        start, end = rest.split('-', 1)
    elif region.count(':') >= 2:
        chrom, start, end = region.rsplit(':', 2)
    elif region.count('-') >= 2:
        chrom, start, end = region.rsplit('-', 2)
    elif region.count('_') >= 2:
        chrom, start, end = region.rsplit('_', 2)
    else:
        raise ValueError(
            f'无法解析 peak 名称: {region}。支持格式示例: chr1:100-200, chr1-100-200, chr1_100_200, chr1:100:200'
        )

    start = int(float(str(start).replace(',', '')))
    end = int(float(str(end).replace(',', '')))
    if end <= start:
        raise ValueError(f'peak 区间非法: {region}')
    return chrom, start, end


def peak_sets_alignment(adata_list, sep=(":", "-"), min_width=20, max_width=10000, min_gap_width=1,
                        peak_region: Optional[str] = None):
    from genomicranges import GenomicRanges
    from iranges import IRanges
    from biocutils.combine import combine

    def _peak_key(chrom, start, end):
        return (str(chrom), int(start), int(end))

    ## Peak merging
    gr_list = []
    for i in range(len(adata_list)):
        seq_names = []
        starts = []
        widths = []
        regions = adata_list[i].var_names if peak_region is None else adata_list[i].obs[peak_region]
        peak_lookup = {}
        for idx, region in enumerate(regions):
            chrom, start, end = _parse_peak_region(region)
            seq_names.append(chrom)
            width = int(end) - int(start)
            starts.append(int(start))
            widths.append(width)
            peak_lookup.setdefault(_peak_key(chrom, start, end), idx)
        gr = GenomicRanges(seqnames=seq_names, ranges=IRanges(starts, widths)).sort()
        peak_keys = [_peak_key(seqname, start, end) for seqname, start, end in
                     zip(gr.get_seqnames(), gr.get_start(), gr.get_end())]
        peak_indices = []
        missing = []
        for key in peak_keys:
            if key in peak_lookup:
                peak_indices.append(peak_lookup[key])
            else:
                missing.append(key)
        if missing:
            raise ValueError(
                f'Peak 对齐失败，找不到以下 peak 的原始索引: {missing[:5]}... 共 {len(missing)} 个。'
                '请检查 var_names 是否在同一数据集中保持一致。'
            )
        adata_list[i] = adata_list[i][:, peak_indices]
        gr_list.append(gr)

    gr_combined = combine(*gr_list)
    gr_merged = gr_combined.reduce(min_gap_width=min_gap_width).sort()
    print("Peak merged")

    ## Peak filtering
    # filter by intesect
    overlap_index_list = []
    index = np.ones(len(gr_merged)).astype(bool)
    for gr in gr_list:
        overlap_index, overlap_count = find_peak_overlaps(gr_merged, gr)
        index = (np.array(overlap_count) > 0) * index
        overlap_index_list.append(overlap_index)
    # filter by width
    index = index * (gr_merged.get_width() > min_width) * (gr_merged.get_width() < max_width)
    gr_merged = gr_merged.get_subset(index)
    common_peak = [seqname + ":" + str(start) + "-" + str(end) for seqname, start, end in
                   zip(gr_merged.get_seqnames(), gr_merged.get_start(), gr_merged.get_end())]
    print("Peak filtered")

    ## Merge count matrix
    adata_merged_list = []
    for adata, overlap_index in zip(adata_list, overlap_index_list):
        overlap_index = [overlap_index[i] for i in range(len(index)) if index[i]]
        X = adata.X.tocsc()
        X_merged = scipy.sparse.hstack([scipy.sparse.csr_matrix(X[:, cur].sum(axis=1)) for cur in overlap_index])
        adata_merged_list.append(
            sc.AnnData(X_merged, obs=adata.obs, var=pd.DataFrame(index=common_peak), obsm=adata.obsm))
    print("Matrix merged")

    return adata_merged_list


def create_neighbor_graph(adata_list, adata_concat, rad_cutoff=None, rad_coef=1.5, coor_key='spatial',
                         use_feature=True, feature_key=None, k=20, similarity_metric='cosine',
                         spatial_weight=0.4, sp_dist='recip', spmethod='threshold'):
    """
    Create neighbor graph considering both spatial proximity and feature similarity.
    
    This function constructs a hybrid graph that combines:
    1. Feature similarity (from LSI/PCA embeddings): captures chromatin accessibility patterns
    2. Spatial proximity: captures spatial relationships between cells
    
    The final graph is computed as: Graph = (1-r) * Feature_Graph + r * Spatial_Graph
    where r is the spatial_weight parameter.
    
    Parameters:
    -----------
    adata_list : list
        List of AnnData objects for each slice
    adata_concat : AnnData
        Concatenated AnnData object
    rad_cutoff : list, optional
        List of radius cutoffs for each slice. If not provided, will be automatically 
        calculated as rad_coef * minimum_distance
    rad_coef : float, default=1.5
        Coefficient to multiply minimum distance for radius cutoff
    coor_key : str, default='spatial'
        Key in obsm for spatial coordinates
    use_feature : bool, default=True
        Whether to use feature similarity in addition to spatial proximity.
        If False, only spatial graph will be constructed (original SpaCRAFT behavior)
    feature_key : str, optional, default=None
        Key in obsm for feature embeddings (e.g., 'X_lsi', 'X_pca', 'X_spectral').
        - For ATAC-seq data with LSI: use 'X_lsi' (recommended, auto-detected by default)
        - For other data with PCA: use 'X_pca'
        - If None, will auto-detect in priority order: X_lsi -> X_pca -> X_spectral
    k : int, default=20
        Number of nearest neighbors for graph construction:
        - For 'cosine': keeps top-k most similar neighbors
        - For 'Jaccard': uses k neighbors for computing shared nearest neighbors
    similarity_metric : str, default='cosine'
        Metric for feature similarity:
        - 'cosine': Cosine similarity (recommended for LSI/PCA embeddings)
        - 'Jaccard': Jaccard coefficient based on shared nearest neighbors
        - 'euclidean': Inverse of Euclidean distance
    spatial_weight : float, default=0.4
        Weight for spatial similarity (corresponds to 'r' parameter in Descart).
        - spatial_weight=0: pure feature-based graph
        - spatial_weight=1: pure spatial graph (original SpaCRAFT)
        - spatial_weight=0.4: balanced combination (Descart default)
        Final graph = (1-spatial_weight)*feature + spatial_weight*spatial
    sp_dist : str, default='recip'
        Type of spatial distance weighting:
        - 'const': Binary adjacency (1 if within threshold, 0 otherwise)
        - 'recip': Reciprocal distance weighting (1/distance)
        - 'square': Squared reciprocal distance weighting (1/distance^2)
    spmethod : str, default='threshold'
        Method for spatial graph construction:
        - 'threshold': Distance threshold-based (recommended)
        - 'SNN': Shared nearest neighbors based
    
    Returns:
    --------
    None
        Modifies adata_concat.uns['graph_list'] in place with the constructed graphs
    
    Examples:
    ---------
    # Default usage (auto-detect LSI, balanced combination)
    >>> create_neighbor_graph(adata_list, adata_concat)
    
    # Pure spatial graph (original SpaCRAFT)
    >>> create_neighbor_graph(adata_list, adata_concat, use_feature=False)
    
    # More emphasis on features
    >>> create_neighbor_graph(adata_list, adata_concat, spatial_weight=0.2)
    
    # Explicitly specify LSI features
    >>> create_neighbor_graph(adata_list, adata_concat, feature_key='X_lsi')
    
    Notes:
    ------
    - This implementation follows Descart's graph construction strategy
    - For ATAC-seq data, LSI (Latent Semantic Indexing) is the recommended embedding
    - The function automatically handles missing feature embeddings by falling back to spatial-only graph
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Auto-detect feature key if not specified
    if feature_key is None and use_feature:
        # Priority: X_lsi (for ATAC-seq/LSI) -> X_pca -> X_spectral
        candidate_keys = ['X_lsi', 'X_pca', 'X_spectral']
        feature_key = None
        # Check in adata_concat first (most common case)
        for key in candidate_keys:
            if key in adata_concat.obsm:
                feature_key = key
                print(f"Auto-detected feature key in adata_concat: {feature_key}")
                break
        # If not found in adata_concat, check in adata_list
        if feature_key is None:
            for key in candidate_keys:
                if len(adata_list) > 0 and key in adata_list[0].obsm:
                    feature_key = key
                    print(f"Auto-detected feature key in adata_list: {feature_key}")
                    break
        
        if feature_key is None:
            print("Warning: No feature embeddings found (X_lsi, X_pca, X_spectral). Using spatial graph only.")
            use_feature = False
    
    if not rad_cutoff:
        rad_cutoff = []

    G_list = []

    # calculate the rad_cutoff based on the min pair distance if it is not provided
    if len(rad_cutoff) == 0:
        for i in range(len(adata_list)):
            adata_copy = adata_list[i].copy()
            loc_copy = np.array(adata_copy.obsm[coor_key])
            pair_dist_ref = pairwise_distances(loc_copy)
            min_dist = np.sort(np.unique(pair_dist_ref), axis=None)[1]

            # Construct spatial similarity matrix
            if spmethod == 'threshold':
                if sp_dist == 'const':
                    graph_spatial = (pair_dist_ref < rad_coef * min_dist).astype(float)
                elif sp_dist == 'recip':
                    graph_spatial = min_dist / (pair_dist_ref + 1e-10)
                    graph_spatial = graph_spatial * (pair_dist_ref < rad_coef * min_dist)
                else:  # square
                    graph_spatial = (min_dist / (pair_dist_ref + 1e-10)) ** 2
                    graph_spatial = graph_spatial * (pair_dist_ref < rad_coef * min_dist)
            elif spmethod == 'SNN':
                neighbor_index = np.argsort(pair_dist_ref, axis=1)[:, 1:k+1]  # exclude self
                graph_spatial = np.zeros_like(pair_dist_ref)
                if similarity_metric == "Jaccard":
                    for ii in range(loc_copy.shape[0]):
                        for jj in range(ii):
                            intersect_num = len(np.intersect1d(neighbor_index[ii, :], neighbor_index[jj, :]))
                            graph_spatial[ii][jj] = 1.0 * intersect_num / (2 * k - intersect_num)
                            graph_spatial[jj][ii] = graph_spatial[ii][jj]
            
            # Construct feature similarity matrix if requested
            if use_feature:
                # Try to get features from adata_copy first, then from adata_concat
                features = None
                if feature_key in adata_copy.obsm:
                    features = adata_copy.obsm[feature_key]
                elif feature_key in adata_concat.obsm:
                    # Extract features for current slice from adata_concat
                    slice_obs_names = adata_copy.obs_names
                    slice_indices = [idx for idx, name in enumerate(adata_concat.obs_names) if name in slice_obs_names]
                    features = adata_concat.obsm[feature_key][slice_indices, :]
                    print(f"Slice {i}: Extracted {features.shape[0]} cells' features from adata_concat")
                else:
                    print(f"Warning: Feature key '{feature_key}' not found. Skipping feature graph for slice {i}.")
                
                if features is not None:
                    if similarity_metric == 'cosine':
                        graph_feature = cosine_similarity(features)
                        # Keep top-k neighbors (most similar), excluding self
                        n_cells = features.shape[0]
                        graph_feature_sparse = np.zeros_like(graph_feature)
                        for ii in range(n_cells):
                            # Get similarities for current cell
                            row = graph_feature[ii, :].copy()
                            # Exclude self by setting diagonal to -inf
                            row[ii] = -np.inf
                            # Find top-k neighbors
                            if k < n_cells:
                                top_k_indices = np.argpartition(row, -k)[-k:]
                                # Keep only top-k values
                                graph_feature_sparse[ii, top_k_indices] = graph_feature[ii, top_k_indices]
                            else:
                                # If k >= n_cells, keep all except diagonal
                                graph_feature_sparse[ii, :] = graph_feature[ii, :]
                                graph_feature_sparse[ii, ii] = 0
                        graph_feature = graph_feature_sparse
                        
                    elif similarity_metric == 'Jaccard':
                        feature_dist = pairwise_distances(features, metric='euclidean')
                        np.fill_diagonal(feature_dist, np.inf)
                        neighbor_index = np.argsort(feature_dist, axis=1)[:, :k]
                        graph_feature = np.zeros((features.shape[0], features.shape[0]))
                        for ii in range(features.shape[0]):
                            for jj in range(ii):
                                intersect_num = len(np.intersect1d(neighbor_index[ii, :], neighbor_index[jj, :]))
                                graph_feature[ii][jj] = 1.0 * intersect_num / (2 * k - intersect_num)
                                graph_feature[jj][ii] = graph_feature[ii][jj]
                                
                    elif similarity_metric == 'euclidean':
                        feature_dist = pairwise_distances(features, metric='euclidean')
                        graph_feature = 1.0 / (feature_dist + 1e-10)
                        np.fill_diagonal(graph_feature, 0)
                    else:
                        raise ValueError(f"Unsupported similarity metric: {similarity_metric}")
                    
                    # Combine spatial and feature graphs
                    graph = (1 - spatial_weight) * graph_feature + spatial_weight * graph_spatial
                    print(f"Slice {i}: Combined graph with spatial_weight={spatial_weight}")
                    # Debug: Print statistics
                    print(f"  Feature graph - mean: {graph_feature.mean():.4f}, std: {graph_feature.std():.4f}, nonzeros: {np.sum(graph_feature > 0)}")
                    print(f"  Spatial graph - mean: {graph_spatial.mean():.4f}, std: {graph_spatial.std():.4f}, nonzeros: {np.sum(graph_spatial > 0)}")
                    print(f"  Combined graph - mean: {graph.mean():.4f}, std: {graph.std():.4f}, nonzeros: {np.sum(graph > 0)}")
                else:
                    graph = graph_spatial
                    print(f"Slice {i}: Using spatial graph only (features not available)")
            else:
                graph = graph_spatial
                print(f"Slice {i}: Using spatial graph only")
            
            G_list.append(graph)
            print(f"Radius for graph connection of slice {i} is {rad_coef * min_dist:.4f}.")
            print('%.4f neighbors per cell on average including itself.' % (np.mean(np.sum(graph > 0, axis=1))))

    elif len(rad_cutoff) != len(adata_list):
        raise ValueError("The length of 'rad_cutoff' should be the number of slices !")

    else:
        for i in range(len(adata_list)):
            adata_copy = adata_list[i].copy()
            loc_copy = np.array(adata_copy.obsm[coor_key])
            pair_dist_ref = pairwise_distances(loc_copy)

            # Construct spatial similarity matrix
            if spmethod == 'threshold':
                if sp_dist == 'const':
                    graph_spatial = (pair_dist_ref < rad_cutoff[i]).astype(float)
                elif sp_dist == 'recip':
                    min_dist = np.sort(np.unique(pair_dist_ref), axis=None)[1]
                    graph_spatial = min_dist / (pair_dist_ref + 1e-10)
                    graph_spatial = graph_spatial * (pair_dist_ref < rad_cutoff[i])
                else:  # square
                    min_dist = np.sort(np.unique(pair_dist_ref), axis=None)[1]
                    graph_spatial = (min_dist / (pair_dist_ref + 1e-10)) ** 2
                    graph_spatial = graph_spatial * (pair_dist_ref < rad_cutoff[i])
            elif spmethod == 'SNN':
                neighbor_index = np.argsort(pair_dist_ref, axis=1)[:, 1:k+1]
                graph_spatial = np.zeros_like(pair_dist_ref)
                if similarity_metric == "Jaccard":
                    for ii in range(loc_copy.shape[0]):
                        for jj in range(ii):
                            intersect_num = len(np.intersect1d(neighbor_index[ii, :], neighbor_index[jj, :]))
                            graph_spatial[ii][jj] = 1.0 * intersect_num / (2 * k - intersect_num)
                            graph_spatial[jj][ii] = graph_spatial[ii][jj]
            
            # Construct feature similarity matrix if requested
            if use_feature:
                # Try to get features from adata_copy first, then from adata_concat
                features = None
                if feature_key in adata_copy.obsm:
                    features = adata_copy.obsm[feature_key]
                elif feature_key in adata_concat.obsm:
                    # Extract features for current slice from adata_concat
                    slice_obs_names = adata_copy.obs_names
                    slice_indices = [idx for idx, name in enumerate(adata_concat.obs_names) if name in slice_obs_names]
                    features = adata_concat.obsm[feature_key][slice_indices, :]
                    print(f"Slice {i}: Extracted {features.shape[0]} cells' features from adata_concat")
                else:
                    print(f"Warning: Feature key '{feature_key}' not found. Skipping feature graph for slice {i}.")
                
                if features is not None:
                    if similarity_metric == 'cosine':
                        graph_feature = cosine_similarity(features)
                        # Keep top-k neighbors (most similar), excluding self
                        n_cells = features.shape[0]
                        graph_feature_sparse = np.zeros_like(graph_feature)
                        for ii in range(n_cells):
                            # Get similarities for current cell
                            row = graph_feature[ii, :].copy()
                            # Exclude self by setting diagonal to -inf
                            row[ii] = -np.inf
                            # Find top-k neighbors
                            if k < n_cells:
                                top_k_indices = np.argpartition(row, -k)[-k:]
                                # Keep only top-k values
                                graph_feature_sparse[ii, top_k_indices] = graph_feature[ii, top_k_indices]
                            else:
                                # If k >= n_cells, keep all except diagonal
                                graph_feature_sparse[ii, :] = graph_feature[ii, :]
                                graph_feature_sparse[ii, ii] = 0
                        graph_feature = graph_feature_sparse
                        
                    elif similarity_metric == 'Jaccard':
                        feature_dist = pairwise_distances(features, metric='euclidean')
                        np.fill_diagonal(feature_dist, np.inf)
                        neighbor_index = np.argsort(feature_dist, axis=1)[:, :k]
                        graph_feature = np.zeros((features.shape[0], features.shape[0]))
                        for ii in range(features.shape[0]):
                            for jj in range(ii):
                                intersect_num = len(np.intersect1d(neighbor_index[ii, :], neighbor_index[jj, :]))
                                graph_feature[ii][jj] = 1.0 * intersect_num / (2 * k - intersect_num)
                                graph_feature[jj][ii] = graph_feature[ii][jj]
                                
                    elif similarity_metric == 'euclidean':
                        feature_dist = pairwise_distances(features, metric='euclidean')
                        graph_feature = 1.0 / (feature_dist + 1e-10)
                        np.fill_diagonal(graph_feature, 0)
                    else:
                        raise ValueError(f"Unsupported similarity metric: {similarity_metric}")
                    
                    # Combine spatial and feature graphs
                    graph = (1 - spatial_weight) * graph_feature + spatial_weight * graph_spatial
                    print(f"Slice {i}: Combined graph with spatial_weight={spatial_weight}")
                    # Debug: Print statistics
                    print(f"  Feature graph - mean: {graph_feature.mean():.4f}, std: {graph_feature.std():.4f}, nonzeros: {np.sum(graph_feature > 0)}")
                    print(f"  Spatial graph - mean: {graph_spatial.mean():.4f}, std: {graph_spatial.std():.4f}, nonzeros: {np.sum(graph_spatial > 0)}")
                    print(f"  Combined graph - mean: {graph.mean():.4f}, std: {graph.std():.4f}, nonzeros: {np.sum(graph > 0)}")
                else:
                    graph = graph_spatial
                    print(f"Slice {i}: Using spatial graph only (features not available)")
            else:
                graph = graph_spatial
                print(f"Slice {i}: Using spatial graph only")
            
            G_list.append(graph)
            print(f"Radius for graph connection of slice {i} is {rad_cutoff[i]:.4f}.")
            print('%.4f neighbors per cell on average including itself.' % (np.mean(np.sum(graph > 0, axis=1))))

    adata_concat.uns['graph_list'] = G_list


def apply_spectral_embedding(adata_concat, n_comps=100, use_snapatac2=True, 
                              distance_metric="cosine", random_state=0, 
                              weighted_by_sd=True, **kwargs):
    """
    使用 Spectral Embedding (Laplacian Eigenmaps) 进行降维，替代 PCA
    
    参数:
    --------
    adata_concat : AnnData
        拼接后的 AnnData 对象
    n_comps : int, default=100
        降维后的维度数
    use_snapatac2 : bool, default=True
        是否使用 SnapATAC2 的 spectral embedding 实现
        如果为 False，将使用简化的实现（需要安装 SnapATAC2）
    distance_metric : str, default="cosine"
        距离度量方式："cosine" 或 "jaccard"
    random_state : int, default=0
        随机种子
    weighted_by_sd : bool, default=True
        是否使用特征值平方根加权
    **kwargs
        传递给 spectral embedding 的其他参数
    
    返回:
    --------
    np.ndarray
        降维后的特征矩阵，形状为 (n_cells, n_comps)
    """
    if use_snapatac2:
        try:
            import snapatac2 as snap
            from snapatac2.tools import spectral
            
            # 创建临时 AnnData 对象用于 spectral embedding
            # 注意：SnapATAC2 的 spectral 函数需要特定的 AnnData 格式
            adata_temp = adata_concat.copy()
            
            # 确保数据是稀疏矩阵格式
            if not scipy.sparse.issparse(adata_temp.X):
                adata_temp.X = scipy.sparse.csr_matrix(adata_temp.X)
            
            print(f'Applying Spectral Embedding (SnapATAC2) to reduce the feature dimension to {n_comps}...')
            
            # 调用 SnapATAC2 的 spectral embedding
            spectral(
                adata_temp,
                n_comps=n_comps,
                features=None,  # 使用所有特征
                random_state=random_state,
                distance_metric=distance_metric,
                weighted_by_sd=weighted_by_sd,
                inplace=True,
                **kwargs
            )
            
            # 提取降维后的特征
            if 'X_spectral' in adata_temp.obsm:
                embedding = adata_temp.obsm['X_spectral']
                print(f'Spectral embedding completed. Shape: {embedding.shape}')
                return embedding
            else:
                raise ValueError("Spectral embedding failed. 'X_spectral' not found in adata.obsm")
                
        except ImportError:
            print("Warning: SnapATAC2 not found. Falling back to simplified spectral embedding.")
            use_snapatac2 = False
        except Exception as e:
            print(f"Warning: SnapATAC2 spectral embedding failed: {e}")
            print("Falling back to simplified spectral embedding.")
            use_snapatac2 = False
    
    # 简化的 spectral embedding 实现（基于 scikit-learn）
    if not use_snapatac2:
        try:
            from sklearn.manifold import SpectralEmbedding
            from sklearn.preprocessing import StandardScaler
            
            print(f'Applying Spectral Embedding (scikit-learn) to reduce the feature dimension to {n_comps}...')
            
            # 转换为密集矩阵（如果数据太大可能需要采样）
            if scipy.sparse.issparse(adata_concat.X):
                X = adata_concat.X.toarray()
            else:
                X = adata_concat.X
            
            # 标准化（可选）
            # scaler = StandardScaler()
            # X = scaler.fit_transform(X)
            
            # 使用 scikit-learn 的 SpectralEmbedding
            # 注意：这需要计算相似度矩阵，对于大数据集可能很慢
            embedding_model = SpectralEmbedding(
                n_components=n_comps,
                affinity='nearest_neighbors',
                n_neighbors=15,
                random_state=random_state
            )
            
            embedding = embedding_model.fit_transform(X)
            print(f'Spectral embedding completed. Shape: {embedding.shape}')
            return embedding
            
        except ImportError:
            raise ImportError("Neither SnapATAC2 nor scikit-learn is available. "
                            "Please install one of them: pip install snapatac2 or pip install scikit-learn")
        except Exception as e:
            raise RuntimeError(f"Spectral embedding failed: {e}")


def apply_dimension_reduction(adata_concat, method='pca', n_comps=100, 
                              spectral_distance_metric="cosine", 
                              spectral_random_state=0, **kwargs):
    """
    统一的降维接口，支持 PCA 和 Spectral Embedding
    
    参数:
    --------
    adata_concat : AnnData
        拼接后的 AnnData 对象
    method : str, default='pca'
        降维方法：'pca' 或 'spectral'
    n_comps : int, default=100
        降维后的维度数
    spectral_distance_metric : str, default="cosine"
        当 method='spectral' 时使用的距离度量
    spectral_random_state : int, default=0
        当 method='spectral' 时使用的随机种子
    **kwargs
        传递给降维方法的其他参数
    
    返回:
    --------
    np.ndarray
        降维后的特征矩阵
    """
    if method.lower() == 'pca':
        from sklearn.decomposition import PCA
        print(f'Applying PCA to reduce the feature dimension to {n_comps}...')
        
        if scipy.sparse.issparse(adata_concat.X):
            X = adata_concat.X.toarray()
        else:
            X = adata_concat.X
        
        pca = PCA(n_components=n_comps, random_state=kwargs.get('random_state', 1234))
        embedding = pca.fit_transform(X)
        print(f'PCA completed. Shape: {embedding.shape}')
        return embedding
        
    elif method.lower() == 'spectral':
        return apply_spectral_embedding(
            adata_concat,
            n_comps=n_comps,
            distance_metric=spectral_distance_metric,
            random_state=spectral_random_state,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported dimension reduction method: {method}. "
                        f"Supported methods: 'pca', 'spectral'")

def mclust_R(
    adata,
    num_cluster=None,
    modelNames="EEE",
    used_obsm="emb_pca",
    random_seed=2020,
):
    """
    使用 R 包 mclust 进行聚类。
    - 如果 num_cluster 为 None，则在 1~20 中自动搜索最佳簇数；
    - 否则使用指定的簇数。
    """
    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri, default_converter
    from rpy2.robjects.conversion import localconverter

    r = robjects.r
    r("library(mclust)")
    r["set.seed"](random_seed)

    # 取出需要聚类的 embedding
    X = adata.obsm[used_obsm]
    X = np.asarray(X, dtype=float)  # 保证是 numpy array

    # 在 localconverter 作用域里启用 numpy <-> R 的自动转换
    with localconverter(default_converter + numpy2ri.converter):
        # 把数据和参数塞到 R 的环境里
        r.assign("X", X)
        if num_cluster is None:
            r.assign("G", robjects.IntVector(range(1, 21)))   # 自动搜索 1..20
        else:
            r.assign("G", robjects.IntVector([int(num_cluster)]))
        r.assign("modelNames", modelNames)

        # 直接在 R 里运行 Mclust，并提取所需字段
        r("res <- Mclust(X, G=G, modelNames=modelNames)")
        cls = np.asarray(r("res$classification"), dtype=int)
        k = int(np.asarray(r("res$G"))[0])

    # 写回到 adata
    adata.obs["mclust"] = cls
    adata.obs["mclust"] = adata.obs["mclust"].astype("int").astype("category")
    adata.uns["mclust_k"] = k

    if num_cluster is None:
        print(f"Mclust 自动选择的最优聚类数: {k}")

    return adata


