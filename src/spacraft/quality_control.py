import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # 仅用于绘图，可选
from anndata import AnnData
from scipy.sparse import issparse

def cal_var(adata, show=True, color=['b', 'r'], save=None):
    """
    Show distribution plots of cells sharing features and variability score.

    Parameters
    ----------

    adata

    show

    color

    save

    """

    if issparse(adata.X):
        adata.var['n_cells'] = adata.X.sum(axis=0).tolist()[0] ## double check this
        adata.var['prop_shared_cells'] = adata.var['n_cells']/len(adata.obs_names.tolist())
        adata.var['variability_score'] = [1-abs(n-0.5) for n in adata.var['prop_shared_cells']]
    else:
        adata.var['n_cells'] = adata.X.sum(axis=0)
        adata.var['prop_shared_cells'] = adata.var['n_cells']/len(adata.obs_names.tolist())
        adata.var['variability_score'] = [1-abs(n-0.5) for n in adata.var['prop_shared_cells']]

    if save!= None:
        fig, axs = plt.subplots(ncols=2)
        plt.subplots_adjust(wspace=0.7)

        ax0 = sns.distplot(adata.var['prop_shared_cells'], bins=40, ax=axs[0], color=color[0])
        ax0.set(xlabel='cells sharing a feature', ylabel='density')
        ax1 = sns.distplot(adata.var['variability_score'], bins=40, ax=axs[1], color=color[1])
        ax1.set(xlabel='variability score', ylabel='density')
        plt.savefig(save, bbox_inches="tight")
        show = False

    if show: # plotting
        fig, axs = plt.subplots(ncols=2)
        plt.subplots_adjust(wspace=0.7)

        ax0 = sns.distplot(adata.var['prop_shared_cells'], bins=40, ax=axs[0], color=color[0])
        ax0.set(xlabel='cells sharing a feature', ylabel='density')
        ax1 = sns.distplot(adata.var['variability_score'], bins=40, ax=axs[1], color=color[1])
        ax1.set(xlabel='variability score', ylabel='density')
        plt.show()
        
def select_var_feature(adata, min_score=0.5, nb_features=None, show=True, copy=False, save=None):
    """
    This function computes a variability score to rank the most variable features across all cells.
    Then it selects the most variable features according to either a specified number of features (nb_features) or a maximum variance score (max_score).

    Parameters
    ----------

    adata: adata object

    min_score: minimum threshold variability score to retain features,
    where 1 is the score of the most variable features and 0.5 is the score of the least variable features.

    nb_features: default value is None, if specify it will select a the top most variable features.
    if the nb_features is larger than the total number of feature, it filters based on the max_score argument

    show: default value True, it will plot the distribution of var.

    copy: return a new adata object if copy == True.

    Returns
    -------
    Depending on ``copy``, returns a new AnnData object or overwrite the input


    """
    if copy:
        inplace=False
    else:
        inplace=True

    adata = adata.copy() if not inplace else adata

    # calculate variability score
    cal_var(adata, show=show, save=save) # adds variability score for each feature
    # adata.var['variablility_score'] = abs(adata.var['prop_shared_cells']-0.5)
    var_annot = adata.var.sort_values(ascending=False, by ='variability_score')

    # calculate the max score to get a specific number of feature
    if nb_features != None and nb_features < len(adata.var_names):
            min_score = var_annot['variability_score'][nb_features]


    adata_tmp = adata[:,adata.var['variability_score']>=min_score].copy()

    ## return the filtered AnnData objet.
    if not inplace:
        adata_tmp = adata[:,adata.var['variability_score']>=min_score]
        return(adata_tmp)
    else:
        adata._inplace_subset_var(adata.var['variability_score']>=min_score)

def highly_variable(adata,
                    min_score=None,
                    n_features=None,
                    figsize=None,
                    save=None):
    """
    Computes variability scores, plots the results, and marks highly variable features. Intended to use with binary
    features.

    Args:
        adata: AnnData
        min_score: variability score threshold
        n_features: number of features to keep
        figsize: size of the figure
        save: if True or str, save the figure. str represents entire path. filetype is inferred.

    Returns:
        None
    """

    if not min_score and not n_features:
        raise ValueError("Either min_score or n_features must not be None")
    elif min_score and n_features:
        raise ValueError("Use either min_score OR n_features")

    adata.var["prop_shared_cells"] = adata.var.n_cells / adata.n_obs
    adata.var["variability_score"] = 1 - np.abs(adata.var.prop_shared_cells - 0.5)

    n_bins = 500
    min_score = min_score
    n_features_selected = n_features

    if min_score is None:
        tmp = adata.var.nlargest(n_features_selected, columns="variability_score")
    else:
        tmp = adata.var[adata.var.variability_score >= min_score]
        n_features_selected = tmp.shape[0]

    min_var_score = tmp.variability_score.min()
    max_var_score = tmp.variability_score.max()

    min_prop_shared = tmp.prop_shared_cells.min()
    max_prop_shared = tmp.prop_shared_cells.max()

    adata.var["highly_variable"] = adata.var.variability_score >= min_var_score

    nrows = 3
    ncols = 1

    if figsize is None:
        figsize = (ncols*6, nrows*3)

    fig, axs = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols, squeeze=False, sharey=False)
    axs = axs.flatten()

    fig.suptitle("Feature Selection")

    axs[0].plot(np.arange(1, adata.n_vars + 1), adata.var.variability_score.sort_values(ascending=False))
    if min_score is None:
        axs[0].axvline(x=n_features_selected, color="red", linestyle="--", linewidth=1, alpha=0.75)
        lims = axs[0].get_xlim()
        axs[0].hlines(y=min_var_score, xmin=lims[0], xmax=n_features_selected, color="black", linestyle="--", linewidth=1, alpha=0.75)
        axs[0].set_xlim(lims)
    else:
        axs[0].axhline(y=min_score, color="red", linestyle="--", linewidth=1, alpha=0.75)
        lims = axs[0].get_ylim()
        axs[0].vlines(x=n_features_selected, ymin=lims[0], ymax=min_var_score, color="black", linestyle="--", linewidth=1, alpha=0.75)
        axs[0].set_ylim(lims)
    axs[0].set_xlabel("Rank")
    axs[0].set_ylabel("Variability Score")

    axs[1].hist(adata.var.variability_score, bins=n_bins, linewidth=0)
    if min_score is None:
        axs[1].axvline(x=min_var_score, color="black", linestyle="--", linewidth=1, alpha=0.75)
    else:
        axs[1].axvline(x=min_var_score, color="red", linestyle="--", linewidth=1, alpha=0.75)
    axs[1].set_xlabel("Variability Score")
    axs[1].set_ylabel("Counts")
    axs[1].set_xlim((0.5, 1))

    axs[2].hist(adata.var.prop_shared_cells, bins=n_bins, linewidth=0)
    if min_score is None:
        axs[2].axvline(x=min_prop_shared, color="black", linestyle="--", linewidth=1, alpha=0.75)
        axs[2].axvline(x=max_prop_shared, color="black", linestyle="--", linewidth=1, alpha=0.75)
    else:
        axs[2].axvline(x=min_score - 0.5, color="red", linestyle="--", linewidth=1, alpha=0.75)
        axs[2].axvline(x=1 - (min_score - 0.5), color="red", linestyle="--", linewidth=1, alpha=0.75)
    axs[2].set_xlabel("Proportion Share Cells")
    axs[2].set_ylabel("Counts")
    axs[2].set_xlim((0, 1))

    plt.tight_layout()

    if not save:
        plt.show()

    else:
        if isinstance(save, str):
            filename = save
        else:
            filename = "feature_selection.png"

        plt.savefig(filename, dpi=300)


def select_highly_variable(adata,
                           verbose=True):
    """
    Returns an AnnData object that only contains the highly variable features.

    Args:
        adata: AnnData
        verbose: some feedback

    Returns:
        AnnData object
    """

    vars_prev = adata.n_vars
    adata = adata[:, adata.var.highly_variable].copy()

    if verbose:
        print("{} of {} features remain ({})".format(adata.n_vars, vars_prev, adata.n_vars - vars_prev))

    return adata
