# Code to analyse the results of the oil flow regression experiment
#%%

import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

#%%

work_dir = Path.cwd()

# Load the results 
experiment_name = "oil_flow_regression"
moilmc_results = pd.read_csv(work_dir / f"results/ILMC/csvs/{experiment_name}.csv")
pls_results = pd.read_csv(work_dir / f"results/PLS/csvs/{experiment_name}.csv")
cls_results = pd.read_csv(work_dir / f"results/CLS/csvs/{experiment_name}.csv")
wsgplvm_results = pd.read_csv(work_dir / f"results/WSGPLVM/csvs/{experiment_name}_50inducing.csv")

#%%

for data_idx in range(10):
    print(f'fold {data_idx}, {wsgplvm_results[wsgplvm_results['data_idx']==data_idx]['seed'].unique()} seeds')
#%%
# get best restarts for PLS, we want the smallest cross validation error
idx_min = pls_results.groupby('data_idx').idxmin()['cross_val_error']
pls_best_restarts = pls_results.loc[idx_min]
pls_best_restarts

#%%

print(f"PLS aggregated results. msep mean: {pls_best_restarts['msep'].mean()}, std: {pls_best_restarts['msep'].std()}")
# %%

# get best restarts for GP, we want the largest marginal log likelihood

# the mll is saved the wrong way round, so fix it 
moilmc_results['mll'] = - moilmc_results['mll']
idx_max = moilmc_results.groupby('data_idx').idxmax()['mll']
ilmc_best_restarts = moilmc_results.loc[idx_max]
ilmc_best_restarts
# %%

for metric in ['msep', 'log_prob']:
    print(f"GP aggregated results. {metric} mean: {ilmc_best_restarts[metric].mean()}, std: {ilmc_best_restarts[metric].std()}")

#%%
wsgplvm_results
#%%
# get best restarts for wsgplvm, we want the largest elbo

idx_max = wsgplvm_results.groupby('data_idx').idxmax()['elbo']
wsgplvm_best_restarts = wsgplvm_results.loc[idx_max]

idx_max = cls_results.groupby('data_idx').idxmax()['elbo']
cls_best_restarts = cls_results.loc[idx_max]

#%%
for metric in ['msep', 'log_prob']:
    print(f"wsgplvm aggregated results. {metric} mean: {wsgplvm_best_restarts[metric].mean()}, std: {wsgplvm_best_restarts[metric].std()}")
for metric in ['msep', 'log_prob']:
    print(f"CLS aggregated results. {metric} mean: {cls_best_restarts[metric].mean()}, std: {cls_best_restarts[metric].std()}")



#%%

print(wsgplvm_results.sort_values(by=['data_idx', 'seed']).to_string())
#%% 

# make table of the results 

final_results = pd.DataFrame(index=['wsgplvm', 'GP', 'PLS', "CLS"], columns=['msep', 'msep std', 'log_prob', 'log_prob std'])

for name, df in {'wsgplvm': wsgplvm_best_restarts, 'GP': ilmc_best_restarts, 'PLS': pls_best_restarts, "CLS":cls_best_restarts}.items(): #'wsgplvm': wsgplvm_best_restarts,
    final_results.loc[name, 'msep'] = df['msep'].mean()
    final_results.loc[name, 'msep std'] = df['msep'].std()
    if 'log_prob' in df.columns:
        final_results.loc[name, 'log_prob'] = df['log_prob'].mean()
        final_results.loc[name, 'log_prob std'] = df['log_prob'].std()


final_results.sort_index()

# %%

# plot the results as a boxplot 

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",  # You can change this to "sans-serif" if needed
    "font.serif": ["Computer Modern Roman"],  # You can change this to your preferred serif font
    "font.size": 28  # Adjust the font size as needed
})

dfs = []
classification = False

df_dict = {'ILMC': ilmc_best_restarts, 'CLS': cls_best_restarts, 'WS-GPLVM-ind': wsgplvm_best_restarts,  'PLS': pls_best_restarts, }

for name, df in df_dict.items(): 
    df['name'] = name
    dfs.append(df)

all_dfs = pd.concat(dfs)

palette = sns.color_palette("husl", n_colors=all_dfs['name'].nunique())


fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6,10.2), sharex=True)


all_dfs['neg_log_prob'] = -all_dfs['log_prob']  
metrics = {'msep': 'MSE \u2193', 'neg_log_prob':'NLPD \u2193'}

i = 0
for metric, title in metrics.items():

    df = all_dfs[all_dfs[metric].notna()]

    # all_dfs.boxplot(column=[metric], by='name', ax=ax)
    # if metric == 'accuracy':
    #     _ = sns.violinplot(x='name', y=metric, hue='name', data=df, palette=palette, linewidth=1, ax=axs[i], cut=0)
    # else:
    _ = sns.boxplot(x='name', y=metric, hue='name', data=df, palette=palette, linewidth=2, ax=axs[i])
    axs[i].set_title(title, fontsize=28)

    formatted_labels = df_dict.keys()
    axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=45, ha='right')
    # axs[0].set_yticks([0, 0.5, 1.])
    # axs[1].set_yticks([300, 500, 700])
    # axs[i].set_yticklabels(, rotation=45, ha='right')
    axs[i].set_xlabel('', fontsize=22)
    axs[i].set_ylabel("", fontsize=22)
    axs[0].ticklabel_format(axis='y', style='sci', scilimits=(-4,-4))
    axs[1].ticklabel_format(axis='y', style='sci', scilimits=(3,3))

    # axs[i].tick_params(axis='x', labelsize=22)
    # axs[i].tick_params(axis='y', labelsize=22)
    i+=1
plt.suptitle("b) Oil Flow Regression")
# plt.tight_layout()

plt.savefig("results/figures/oil_flow_regression_results_vertical_short.pdf", bbox_inches='tight')


# %%
