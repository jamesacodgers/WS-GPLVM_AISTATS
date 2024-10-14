# Results for spectroscopy classification experiment
#%%

import numpy as np
import torch
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns

#%%


work_dir = Path.cwd()

wsgplvm_results = pd.read_csv(work_dir / "results/WSGPLVM/csvs/spectroscopy_classification_incl_wavelength_latent_inducing_16_wl_inducing_20.csv")
wsgplvm_results_no_wl = pd.read_csv(work_dir / "results/WSGPLVM/csvs/spectroscopy_classification.csv")
ilmc_results = pd.read_csv(work_dir / "results/ILMC/csvs/spectroscopy_classification.csv")
pls_results = pd.read_csv(work_dir / "results/PLS/csvs/spectroscopy_classification.csv")
cls_results= pd.read_csv(work_dir / "results/CLS/csvs/spectroscopy_classification.csv")

#%%
# get best restarts for PLS, we want the smallest cross validation error
idx_min = pls_results.groupby('fold_number').idxmin()['cross_val_error']
pls_best_restarts = pls_results.loc[idx_min]
pls_best_restarts

#%%

print(f"PLS aggregated results. accuracy mean: {pls_best_restarts['accuracy'].mean()}, std: {pls_best_restarts['accuracy'].std()}")
# %%

# get best restarts for GP, we want the largest marginal log likelihood
idx_max = ilmc_results.groupby('fold_number').idxmax()['mll']
ilmc_best_restarts = ilmc_results.loc[idx_max]
ilmc_best_restarts
# %%

for metric in ['accuracy', 'log_prob']:
    print(f"ILMC aggregated results. {metric} mean: {ilmc_best_restarts[metric].mean()}, std: {ilmc_best_restarts[metric].std()}")

#%% 

# get best restarts for WSGPLVM, we want the largest elbo

wsgplvm_results['elbo'] = - np.abs(wsgplvm_results['elbo'])
idx_max = wsgplvm_results.groupby('fold_number').idxmax()['elbo']
wsgplvm_best_restarts = wsgplvm_results.loc[idx_max]

#%%
for metric in ['accuracy', 'log_prob']:
    print(f"WSGPLVM aggregated results. {metric} mean: {wsgplvm_best_restarts[metric].mean()}, std: {wsgplvm_best_restarts[metric].std()}")

#%%

# get best restarts for WSGPLVM, we want the largest elbo

wsgplvm_results_no_wl['elbo'] = - np.abs(wsgplvm_results_no_wl['elbo'])
idx_max = wsgplvm_results_no_wl.groupby('fold_number').idxmax()['elbo']
wsgplvm_best_restarts_no_wl = wsgplvm_results_no_wl.loc[idx_max]

#%%
for metric in ['accuracy', 'log_prob']:
    print(f"WSGPLVM no wl aggregated results. {metric} mean: {wsgplvm_best_restarts_no_wl[metric].mean()}, std: {wsgplvm_best_restarts_no_wl[metric].std()}")

#%%

idx_max = cls_results.groupby('data_idx').idxmax()['elbo test']
cls_best_restarts = cls_results.loc[idx_max]

#%%
for metric in ['accuracy', 'log_prob']:
    print(f"CLS aggregated results. {metric} mean: {wsgplvm_best_restarts[metric].mean()}, std: {wsgplvm_best_restarts[metric].std()}")

# %%

# make table of the results 

final_results = pd.DataFrame(index=['WSGPLVM', 'GP', 'PLS'], columns=['accuracy', 'accuracy std', 'log_prob', 'log_prob std'])

for name, df in {'WSGPLVM': wsgplvm_best_restarts, 'wsgplvm_no_wl': wsgplvm_best_restarts_no_wl, 'GP': ilmc_best_restarts, 'CLS':cls_best_restarts, 'PLS': pls_best_restarts}.items(): #'WSGPLVM': wsgplvm_best_restarts,
    final_results.loc[name, 'accuracy'] = df['accuracy'].mean()
    final_results.loc[name, 'accuracy std'] = df['accuracy'].std()
    if 'log_prob' in df.columns:
        final_results.loc[name, 'log_prob'] = df['log_prob'].mean()
        final_results.loc[name, 'log_prob std'] = df['log_prob'].std()
        final_results.loc[name, 'prob'] = np.mean(np.exp(df['log_prob']))

final_results.sort_index()

#%%

# Calculate ROC AUC

def roc(n_classes, y_true, y_scores):

    n_classes = 4
    # Binarize the output labels for multi-class
    y_true_bin = y_true

    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return roc_auc


results = {}


for name, df in {'WSGPLVM':wsgplvm_best_restarts, 'CLS':cls_best_restarts, 'ILMC': ilmc_best_restarts, 'wsgplvm_no_wl':wsgplvm_best_restarts_no_wl,}.items():
    actual = []
    predicted = []
    actual_argmax = []
    predicted_argmax = []

    for i, row in df.iterrows():
        seed = int(row['seed'])
        if name == 'WSGPLVM':
            fold = int(row['fold_number'])
            params = torch.load(work_dir / f'results/{name}/parameters/spectroscopy_classification_incl_wavelength_latent_inducing_16_wl_inducing_20/test_dataset_fold_{fold}_seed_{seed}.pt')
            test_probs =params['components_distribution._log_p']

        elif name == 'wsgplvm_no_wl':
            fold = int(row['fold_number'])
            params = torch.load(work_dir / f'results/WSGPLVM/parameters/spectroscopy_classification/test_dataset_fold_{fold}_seed_{seed}.pt')
            test_probs =params['components_distribution._log_p']

        elif name == 'ILMC':
            fold = int(row['fold_number'])
            test_probs = torch.load(work_dir / f'results/ILMC/parameters/spectroscopy_classification_with_saved_logprob/test_log_probs_fold_{fold}_seed_{seed}.pt')

        else:
            fold = int(row['data_idx'])
            params = torch.load(work_dir / f'results/CLS/parameters/spectroscopy_classification/test_dataset_data_{fold}_seed_{seed}.pt')
            test_probs =params['components_distribution._log_p']


        training_df = pd.read_csv(work_dir / f"examples/data/UCRArchive_2018/Rock/cross_val/Rock_TRAIN_diff_fold_{fold}.tsv", delimiter="\t", header  = None)
        test_df = pd.read_csv(work_dir / f"examples/data/UCRArchive_2018/Rock/cross_val/Rock_TEST_diff_fold_{fold}.tsv", delimiter="\t", header  = None)

        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(training_df[0].to_numpy().reshape(-1, 1))

        test_components_one_hot = enc.transform(test_df[0].to_numpy().reshape(-1, 1)).toarray()

        actual.append(test_components_one_hot)
        predicted.append(test_probs)
        actual_argmax.append(test_components_one_hot.argmax(axis=1).flatten())
        predicted_argmax.append(test_probs.argmax(dim=1).flatten())
        roc_auc = roc(4, test_components_one_hot, test_probs)
        print(roc_auc.values())
        # print(np.mean(roc_auc.values()))
        for key, value in roc_auc.items():
            df.loc[i, f'roc_auc_{key}'] = value
        df.loc[i, 'roc_auc_mean'] = np.nanmean(list(roc_auc.values()))


# %%


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",  # You can change this to "sans-serif" if needed
    "font.serif": ["Computer Modern Roman"],  # You can change this to your preferred serif font
    "font.size": 28  # Adjust the font size as needed
})


df_dict = {'ILMC': ilmc_best_restarts, 'CLS-GP': cls_best_restarts, 'WS-GPLVM': wsgplvm_best_restarts,  'PLS': pls_best_restarts, } #'WS-GPLVM-ind.': wsgplvm_best_restarts_no_wl,
dfs = []
for name, df in df_dict.items(): 
    df['name'] = name
    dfs.append(df)

all_dfs = pd.concat(dfs)

palette = sns.color_palette("husl", n_colors=all_dfs['name'].nunique())

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12,3))

metrics = {'accuracy':'Accuracy \u2191', 'log_prob':'LPP \u2191', 'roc_auc_mean':'ROC AUC \u2191'}



i = 0
for metric, title in metrics.items():

    df = all_dfs[all_dfs[metric].notna()]

    _ = sns.boxplot(x='name', y=metric, hue='name', data=df, palette=palette, linewidth=2, ax=axs[i], medianprops={"linewidth": 3})
    axs[i].set_title(title, fontsize=28)

    formatted_labels = df_dict.keys()
    axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=45, ha='right')
    axs[0].set_yticks([0, 0.5, 1.])
    axs[1].set_yticks([-150, -75, 0])
    axs[2].set_yticks([0.2, 0.6, 1.])
    # axs[i].set_yticklabels(, rotation=45, ha='right')
    axs[i].set_xlabel('', fontsize=22)
    axs[i].set_ylabel("", fontsize=22)

    i+=1
plt.subplots_adjust(wspace=0.4)

plt.savefig( work_dir / "results/figures/spectroscopy_classification_results_horizontal_incl_roc_short.pdf", bbox_inches='tight')


# %%
