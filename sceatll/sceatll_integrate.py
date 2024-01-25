import scanpy as sc
import matplotlib.pyplot as plt
from song.umap_song import SONG
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, vstack
import scanpy.external as sce
from sklearn.linear_model import LinearRegression
import pandas as pd
import pickle
from sklearn.metrics import homogeneity_score
import time as t
import os
from umap import UMAP

def get_common_genes(sources):
    """
    This function homegenizes the variables in the disparate h5ad files. Current implementation takes the intersection\
     of the genes across all RNA-Seq data

    Args:
        sources (List[str]): list of path for each cellxgene path. 

    Returns:
        Union(dict, numpy array): Returns a tuple of two values. The first value is a dictionary, keys = data source, values = variable index for each gene.\
        the second value is a numpy array of strings, which contains the cumulative cell-type list for each of the data source. 
    """
    
    cell_types = []  # common = None
    genes = []
    shared_gene_dict = {}

    for i in range(len(sources)):

        ad_read = sc.read_h5ad(f'{sources[i]}/local.h5ad')

        cell_types.extend(list(np.unique(np.array(ad_read.obs['cell_type']))))
        genes.append(list(ad_read.var_names))
        if not i:
            common = ad_read.var_names
        else:
            common = np.intersect1d(ad_read.var_names, common)
    del ad_read

    print(len(common))
    for i in range(len(sources)):
        series = pd.Series(np.arange(len(genes[i])), index=genes[i])
        shared_gene_dict[sources[i]] = (np.array(series[common]))

    for i in range(len(sources)):
        print(len(shared_gene_dict[sources[i]]))

    # np.save('./extended_integration/common_genes.npy', shared_gene_dict)
    #
    # # %%
    #
    # np.savetxt('./extended_integration/cell_types_union.txt', cell_types, fmt='%s')
    print(shared_gene_dict)
    return shared_gene_dict, np.unique(cell_types)
    # %%



def read_preprocess_sample(fpath, sample_size = -1, varinds = None, normalise = True, logarithmize = True):
    """Preprocess each of the AnnData files to be processed by the SCEATLL Workflow

    Args:
        fpath (str): path to the h5ad file
        sample_size (int, optional): Number of samples to consider. Defaults to -1.
        varinds (List[int], optional): List of indices of the genes. See get_common_genes(). Defaults to None.
        normalise (bool, optional): Should the file be normalised?. Defaults to True.
        logarithmize (bool, optional): Should the file be logarithmized?. Defaults to True.

    Returns:
        ad(scanpy.AnnData): The preprocessed data ready to be handled by the integration.
    """

    ad = sc.read_h5ad(fpath)
    print(ad.raw)
    var_strings = ad.var_names[varinds]
    if sample_size == -1:
        sample_size = ad.shape[0]
    try:
        ad.X = ad.raw.X
    except:
        print(f'no raw found in {fpath}. using input data')
    ad = ad[:sample_size]
    ad_n = ad[:, varinds]
    ad_n.obs = ad.obs
    ad = ad_n
    ad.var_names = var_strings

    # print(ad.X[:10])
    if normalise:
        sc.pp.normalize_per_cell(ad)
    if logarithmize:
        sc.pp.log1p(ad)
    order = np.random.permutation(ad.shape[0])[:sample_size]
    return ad[order]

def visualize_ad(ad, Y_u, field, savepath = 'figure.png', labelencoder = LabelEncoder(), n_cell_types = 0):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    le = labelencoder
    if not n_cell_types:
        n_cell_types = len(np.unique(ad.obs[field].tolist()))
    for u in np.unique(ad.obs[field].tolist()):
        c = plt.cm.Spectral((le.transform([u]).astype(float)) / n_cell_types)[0]
        ixs = np.where(np.array(ad.obs[field].tolist()) == u)[0]

        ax.scatter(Y_u[ixs].T[0], Y_u[ixs].T[1], color=c, label=u, s=0.5, alpha=0.2)
        # Y_pos = Y_u[ixs].mean(axis=0)
        # ax.text(Y_pos[0], Y_pos[1], u)

    # ax.legend()
    plt.savefig(savepath)


def fit_first(model, ad):
    """Fit the first dataset with no batch correction. 

    Args:
        model (SONG): The SONG model to be used in the evolving integration
        ad (scanpy.AnnData): Data to fit the model

    Returns:
        Union(SONG, scanpy.AnnData): Will return the fitted SONG model and the AnnData Object
    """

    model.fit(ad.X)

    return model, ad

def fit_with_correction(model, ad_old, ad_new):
    """ Fit an evolving model with a new dataset. also needs a subset of the old data.

    Args:
        model (SONG): Evolving SONG model
        ad_old (scanpy.AnnData): ad object; sample of the old data to retain the distribution.
        ad_new (scanpy.AnnData): ad object; the new cells

    Returns:
        Union(SONG, scanpy.AnnData): returns the fitted SONG model and the final data used to train the model.
    """

    rand_choice = np.random.choice(np.arange(ad_old.shape[0]), 100000)

    ad_tr = sc.concat([ad_old[rand_choice], ad_new])
    model_ad = sc.AnnData(model.W)
    model_ad.obs['batch'] = ['0'] * model_ad.shape[0]
    temp_ad = sc.AnnData(ad_new.X[:2000])#(temp_model.W)
    temp_ad.obs['batch'] = ['1'] * temp_ad.shape[0]
    har_ad = sc.concat([model_ad, temp_ad])#sc.concat([new_sample, old_sample])#sc.AnnData(X=(vstack([old_sample, new_sample])))
    har_ad.obsm['X_pca'] = model._get_XPCA(har_ad.X.toarray())
    print(f'\n\nBATCHES DETECTED: {np.unique(har_ad.obs["batch"])}')
    # har_ad.obs['batch']
    # sc.pp.pca(har_ad)
    # har_ad.obs['batch'] = np.asarray(['a'] * model_ad.shape[0] + ['b'] * temp_ad.shape[0])
    sce.pp.harmony_integrate(har_ad, key='batch', max_iter_harmony=40)

    print('harmony correction done')
    lreg = LinearRegression(fit_intercept=False, n_jobs = -1)

    rand_choice = np.random.choice(np.arange(har_ad.shape[0]), 5000)

    lreg.fit(har_ad.X[rand_choice], har_ad.obsm['X_pca_harmony'][rand_choice])
    print('mvr fit done')

    y=UMAP().fit_transform(har_ad.obsm['X_pca_harmony'])

    plt.scatter(y.T[0], y.T[1], c= LabelEncoder().fit_transform(har_ad.obs['batch']), )
    plt.savefig(f'harmony_umap_{len(np.unique(har_ad.obs["batch"]))}')
    reconstruction = lreg.predict(har_ad.X)
    # print('reconstruction')
    # print(reconstruction)
    rmse = np.linalg.norm(reconstruction - har_ad.obsm['X_pca_harmony'])

    print(f'discrepancy = {rmse}')
    model.pca.components_ = lreg.coef_
    model.linear_transform = True
    # plt.show()
    # print('prediction')
    # print(model._get_XPCA(har_ad.X))
    # print('original')
    # print(har_ad.obsm['X_pca_harmony'])

    model.fit(ad_tr.X)

    return model, ad_tr

def get_purity_for_onek1k(onekad, song):
    clustering, _, _, _ = song.get_representatives(onekad.X)
    true_labels = LabelEncoder().fit_transform(onekad.obs['cell_type'].values)
    purity_score = homogeneity_score(true_labels, clustering)

    print(purity_score)



def SCEATLL_INTEGRATE(model, sources, input_path, output_path):
    """Integrates a list of data sources into an evolving atlas, one dataset at a time.

    Args:
        model (SONG): a SONG model to be used for the evolving Atlas
        sources (List(string)): list of names of the data sources.
        input_path(string): root folder for where the data sources can be found. 
        output_path(string): root folder to store the outputs and intermediate files.

    Returns:
        model (SONG): the fitted final SONG model after integrating all the datasets
    """

    shared_gene_dict, _ = get_common_genes(sources)
    
    combine_ads = []
    # onek1ksample = read_preprocess_sample('onek1k_full/local.h5ad', 10000, shared_gene_dict['onek1k_full'], True, True)
    for i, s in enumerate(sources):

        fpath = os.path.join(input_path,f'{s}/local.h5ad')
        ad_read = read_preprocess_sample(fpath, -1, shared_gene_dict[s], logarithmize=logar[i], normalise=normar[i])
        ad_read.write_h5ad(os.path.join(output_path, f'csvs/incremental_add_{sources[i]}.h5ad'))
        ad_read.obs['batch'] = np.asarray([str(i)] * ad_read.shape[0])
        combine_ads.append(ad_read)
        st_s = t.time()
        if not i:
            model, ad_tr = fit_first(model, ad_read)
        else:
            model, ad_tr = fit_with_correction(model, ad_tr, ad_read)
        et_s = t.time()

        print(f'for the dataset { sources[i] }: {et_s-st_s} seconds')

        # ad_tot = sc.concat(combine_ads)
        with open(os.path.join(output_path,f'integrated_files/song_int_{i}.obj'), 'wb') as picfile:
            pickle.dump(model, picfile)
        # Y = model.transform(ad_tot.X)
        # visualize_ad(ad_tot, Y, 'cell_type', f'pngs/new_intermediate_{i}_{s}.png', labelencoder=le, n_cell_types=n_ct)
        # get_purity_for_onek1k(onek1ksample, model)
    print('incremental training done')
    return model

