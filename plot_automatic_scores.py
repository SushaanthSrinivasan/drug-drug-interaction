import json
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import warnings
import re
import numpy as np
from matplotlib.lines import Line2D
import scipy.stats as stats
import pandas as pd

warnings.filterwarnings('ignore', message='FixedFormatter should only be used together with FixedLocator')

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'

plt.rcParams.update({'font.size': 6, 'font.family': 'sans-serif', 'font.sans-serif': 'Arial' })
matplotlib.rcParams['axes.linewidth'] = 0.3


# List of chemical questions and score types
# chemical_questions = ['description', 'indication', 'pharmacodynamics', 'mechanism of action']
chemical_questions = ['mechanism of interaction', 'management', ]
# score_types = ['Human evaluation', 'cos sim', 'METEOR', 'Precision']
# score_types = ['Human evaluation', 'cos sim', 'Precision', 'METEOR']

score_types = ['cos sim', 'BLEU-1', 'METEOR']

# Initialize figure
fig, axes = plt.subplots(len(score_types), len(chemical_questions), figsize=(6.5, 5))
# fig.suptitle('Comparison of Scores for Chemical Questions', fontsize=16)

# Customize appearance
flierprops = dict(marker='o', color='b', markersize=0.3)
medianprops = dict(color="blue", linewidth=0.5)
meanprops = dict(color="purple", linewidth=0.5)
boxprops = dict(
    linecolor='black', linewidth=0.4, width=.5,
    showmeans=True, meanline=True, showcaps=True, 
    showbox=True,
    # showfliers=True,
    # whis=[0, 100],
)

my_colors = {'GPT-4': (142/255.,153/255.,171/255.), 'DrugChat': (183/255.,142/255.,114/255.)}

# Create custom legend handles
legend_elements = [
    Line2D([0], [0], color=my_colors['GPT-4'], lw=5, label='GPT-4'),
    Line2D([0], [0], color=my_colors['DrugChat'], lw=5, label='DrugChat')
]


def cohen_d_from_samples(sample1, sample2, sided='two-sided'):
    """
    Computes Cohen's d effect size for a t-test from two sample lists.
    
    Parameters:
    - sample1, sample2: Lists or arrays of sample values for two groups.
    - sided: 'one-sided' or 'two-sided', affects the interpretation but not the calculation.
    
    Returns:
    - Cohen's d value.
    """
    
    # Calculate means, standard deviations, and sample sizes for both samples
    mean1 = np.mean(sample1)
    mean2 = np.mean(sample2)
    
    std1 = np.std(sample1, ddof=1)  # Standard deviation with Bessel's correction
    std2 = np.std(sample2, ddof=1)
    
    n1 = len(sample1)
    n2 = len(sample2)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    # Compute Cohen's d
    d = (mean1 - mean2) / pooled_std

    # print("Note: Effect size computation (Cohen's d) does not differ for one-sided vs two-sided tests.")
    return d


def get_stats(scores1, scores2, sided='two-sided'):
    ttest_result = stats.ttest_ind(scores1, scores2, equal_var=False, alternative=sided)
    tukey = stats.tukey_hsd(scores1, scores2)
    print(ttest_result, tukey)
    # import ipdb
    # ipdb.set_trace()
    t_statistic, p_value = ttest_result
    df = ttest_result.df
    ci95_low, ci95_high = ttest_result.confidence_interval()
    effect_size = cohen_d_from_samples(scores1, scores2)
    return t_statistic, p_value, df, ci95_low, ci95_high, effect_size


def read_scores_and_preprocess(file_path):

    remove_keys = {"management", "mechanism_interaction", "mean_scores", "advisory_terms", "classification_interaction"}
    
    new_categories = {
        "management": "management",
        "mechanism_interaction": "mechanism of interaction"
    }

    with open(file_path, 'r') as f:
        results = json.load(f)

    scores = {chem_q: {st: [] for st in score_types} for chem_q in chemical_questions}
    
    for smiles, result in results.items():
        if smiles in remove_keys:
            continue
        
        for category in ["management", "mechanism_interaction"]:
            category_data = result.get(category, {})
            for key, value in category_data.items():
                if "similarity" in key:
                    metric = 'cos sim'
                elif "meteor" in key:
                    metric = 'METEOR'
                elif "bleu_1" in key:
                    metric = 'BLEU-1'
                else:
                    continue
                
                scores[new_categories[category]][metric].append(value)
    
    return scores


    

ttest_results = []
all_scores = {t: {'GPT-4': [], 'DrugChat': []} for t in score_types}
alternative = 'two-sided'  # 'less' #
gpt4_scores = read_scores_and_preprocess('/home/zhaoyang/project/drug-drug-interaction/eval_results/gpt_results.json')
ours_scores = read_scores_and_preprocess('/home/zhaoyang/project/drug-drug-interaction/eval_results/ours.json')

# Create box plots
for i, question in enumerate(chemical_questions):
    for j, score in enumerate(score_types):
        # scores1 = get score of 'GPT-4'
        # scores2 = get score of 'DrugChat'
        
        scores1 = gpt4_scores[question][score]
        scores2 = ours_scores[question][score]
        
        all_scores[score]['GPT-4'].extend(scores1)
        all_scores[score]['DrugChat'].extend(scores2)

        ax = axes[j, i]
        sns.boxplot(
            data={'GPT-4': scores1, 'DrugChat': scores2}, ax=ax, 
            medianprops=medianprops, meanprops=meanprops, flierprops=flierprops, 
            palette=my_colors,
            # labels=['GPT-4', 'DrugChat'] if i == j == 0 else [],
            **boxprops)
        # ax.set_title(f'{question} - {score}')
        # ax.set_xticklabels(['GPT-4', 'DrugChat'])

        ax.xaxis.set_visible(False)
        ax.yaxis.set_tick_params(width=0.3)
        ax.spines[['right', 'top']].set_visible(False)

        if score == 'cos sim':
            score = 'Semantic similarity'
        if score == 'Precision':
            score = 'BLEU'
        if score == 'Human evaluation':
            score = 'Human evaluation score'
        if i == 0:
            ax.set_ylabel(f'{score}')

        # t test
        if 'description' in question:
            qq = 'overview'.capitalize()
        else:
            qq = question.capitalize()

        t_statistic, p_value, df, ci95_low, ci95_high, effect_size = get_stats(scores1, scores2, alternative)
        ttest_results.append([f'{qq}_{score}'.replace(' ', '_'), t_statistic, p_value, df, ci95_low, ci95_high, effect_size])

        if j == 0:
            if 'description' in question:
                ax.set_title('overview'.capitalize())
            else:
                ax.set_title(question.capitalize())

        if i == 0 and j == 0:
            ax.legend(handles=legend_elements, bbox_to_anchor=(0, 1.2, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2)

# t test
for score, res in all_scores.items():
    scores1, scores2 = res['GPT-4'], res['DrugChat']
    if score == 'cos sim':
        score = 'Semantic similarity'
    if score == 'Precision':
        score = 'BLEU'
    if score == 'Human evaluation':
        score = 'Human evaluation score'
    t_statistic, p_value, df, ci95_low, ci95_high, effect_size = get_stats(scores1, scores2, alternative)
    ttest_results.append([f'all_{score}'.replace(' ', '_'), t_statistic, p_value, df, ci95_low, ci95_high, effect_size])

tt = pd.DataFrame(ttest_results, columns=['metric', 't_statistic', 'p_value', 'degree_of_freedom', 'ci95_low', 'ci95_high', 'Cohens_d_effect_size'])
tt.to_csv('eval_results/similarity_scores/t_test_single_sided.csv', index=False)


plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('eval_results/similarity_scores/scores_box_plot.png', dpi=400)
plt.savefig('eval_results/similarity_scores/scores_box_plot.svg')
plt.savefig('eval_results/similarity_scores/scores_box_plot.pdf')