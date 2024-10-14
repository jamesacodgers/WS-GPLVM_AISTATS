#%%

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

#%%


work_dir = Path.cwd()

experiment_name = "spectroscopy_regression"
ilmc_results = pd.read_csv(work_dir / f"results/ILMC/csvs/{experiment_name}.csv")
pls_results = pd.read_csv(work_dir / f"results/PLS/csvs/{experiment_name}.csv")
wsgplvm_results = pd.read_csv(work_dir / f"results/WSGPLVM/csvs/{experiment_name}.csv")
wsgplvm_results_no_wl = pd.read_csv(work_dir / f"results/WSGPLVM/csvs/{experiment_name}_no_wl.csv")
cls_results = pd.read_csv(work_dir / f"results/CLS/csvs/{experiment_name}.csv")


#%%
# get best restarts for PLS, we want the smallest cross validation error
idx_min = pls_results.groupby('data_idx').idxmin()['cross_val_error']
pls_best_restarts = pls_results.loc[idx_min]
pls_best_restarts

#%%

print(f"PLS aggregated results. msep mean: {pls_best_restarts['msep'].mean()}, std: {pls_best_restarts['msep'].std()}")


# the mll is saved the wrong way round, so fix it 
ilmc_results['mll'] =  - ilmc_results['mll']
idx_max = ilmc_results.groupby('data_idx').idxmax()['mll']
ilmc_best_restarts = ilmc_results.loc[idx_max]
ilmc_best_restarts
# %%

for metric in ['msep', 'log_prob']:
    print(f"ILMC aggregated results. {metric} mean: {ilmc_best_restarts[metric].mean()}, std: {ilmc_best_restarts[metric].std()}")

#%%

wsgplvm_results
#%%
# get best restarts for WS-GPLVM, we want the largest elbo

# wsgplvm_results['elbo'] = wsgplvm_results['elbo'].apply(lambda x: float(x[7:17]))
idx_max = wsgplvm_results.groupby('data_idx').idxmax()['elbo test']
wsgplvm_best_restarts = wsgplvm_results.loc[idx_max]

#%%
idx_max = wsgplvm_results_no_wl.groupby('data_idx').idxmax()['elbo test']
wsgplvm_best_restarts_no_wl = wsgplvm_results_no_wl.loc[idx_max]

#%%


idx_max = cls_results.groupby('data_idx').idxmax()['elbo test']
cls_best_restarts = cls_results.loc[idx_max]

#%%
for metric in ['msep', 'log_prob']:
    print(f"WS-GPLVM aggregated results. {metric} mean: {wsgplvm_best_restarts[metric].mean()}, std: {wsgplvm_best_restarts[metric].std()}")

#%%
print(wsgplvm_best_restarts.to_string())

#%%

print(wsgplvm_results.sort_values(by=['data_idx', 'seed']).to_string())
#%% 

# wsgplvm_results['elbo'] = wsgplvm_results['elbo'].apply(lambda x: float(x[7:17]))
idx_max = cls_results.groupby('data_idx').idxmax()['elbo test']
cls_best_restarts = cls_results.loc[idx_max]

# %%

# make table of the results 

final_results = pd.DataFrame(index=['WS-GPLVM', 'GP', 'PLS'], columns=['msep', 'msep std', 'log_prob', 'log_prob std'])

for name, df in {'WS-GPLVM': wsgplvm_best_restarts, 'WS-GPLVM no wl': wsgplvm_best_restarts_no_wl, 'GP': ilmc_best_restarts, 'PLS': pls_best_restarts, 'CLS': cls_best_restarts}.items(): #'WS-GPLVM': wsgplvm_best_restarts,
    final_results.loc[name, 'msep'] = df['msep'].mean()
    final_results.loc[name, 'msep std'] = df['msep'].std()
    if 'log_prob' in df.columns:
        final_results.loc[name, 'log_prob'] = df['log_prob'].mean()
        final_results.loc[name, 'log_prob std'] = df['log_prob'].std()

final_results.sort_index()

# %%

# plot results 

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",  # You can change this to "sans-serif" if needed
    "font.serif": ["Computer Modern Roman"],  # You can change this to your preferred serif font
    "font.size": 28  # Adjust the font size as needed
})

df_dict = {'ILMC': ilmc_best_restarts, 'CLS-GP': cls_best_restarts, 'WS-GPLVM': wsgplvm_best_restarts, 'PLS': pls_best_restarts, }

dfs = []
for name, df in df_dict.items(): 
    df['name'] = name
    dfs.append(df)

all_dfs = pd.concat(dfs)

all_dfs['neg_log_prob'] = - all_dfs['log_prob']

palette = sns.color_palette("husl", n_colors=all_dfs['name'].nunique())

fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6,10.2), sharex=True) #7.5

metrics = {'msep': 'MSE \u2193', 'neg_log_prob':'NLPD \u2193'}


i = 0
for metric, title in metrics.items():

    df = all_dfs[all_dfs[metric].notna()]

    # all_dfs.boxplot(column=[metric], by='name', ax=ax)
    _ = sns.boxplot(x='name', y=metric, hue='name', data=df, palette=palette, linewidth=2, ax=axs[i])
    axs[i].set_title(title, fontsize=28)

    formatted_labels = df_dict.keys()
    axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=45, ha='right')
    axs[0].set_yticks([0, 0.02, 0.04, 0.06])
    axs[1].set_yticks([-300, -500, -700])
    # axs[i].set_yticklabels(, rotation=45, ha='right')
    axs[i].set_xlabel('', fontsize=22)
    axs[i].set_ylabel("", fontsize=22)
    axs[0].ticklabel_format(axis='y', style='sci', scilimits=(-2,-2))
    axs[1].ticklabel_format(axis='y', style='sci', scilimits=(2,2))

    # axs[i].tick_params(axis='x', labelsize=22)
    # axs[i].tick_params(axis='y', labelsize=22)
    i+=1
plt.suptitle("a) Spectroscopy Regression")

# plt.tight_layout()


plt.savefig(work_dir / "results/figures/spectroscopy_regression_results_vertical_short.pdf", bbox_inches='tight')


# %%
