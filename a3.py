import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, ttest_ind
import statsmodels.formula.api as smf
import nltk
from nltk.stem import WordNetLemmatizer
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# ==========================================
# Setup
# ==========================================

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')

# ==========================================
# 0. Load and Prepare Data
# ==========================================

df_rt = pd.read_csv('processed_RTs.tsv', sep='\t')

df_rt['word'] = df_rt['word'].astype(str)
df_rt['word_length'] = df_rt['word'].apply(len)

# ==========================================
# Part I: Preliminary Data Analysis
# ==========================================

mean_rt_per_word = (
    df_rt
    .groupby(['word', 'item', 'zone'])['RT']
    .mean()
    .reset_index()
    .rename(columns={'RT': 'mean_rt'})
)

# REQUIRED predictors for regressions
mean_rt_per_word['word_length'] = mean_rt_per_word['word'].apply(len)
word_freqs = mean_rt_per_word['word'].value_counts().to_dict()
mean_rt_per_word['word_freq'] = mean_rt_per_word['word'].map(word_freqs)

# Aggregate for plotting
unique_words = (
    mean_rt_per_word
    .groupby('word')
    .agg(mean_rt=('mean_rt', 'mean'),
         frequency=('word', 'count'))
    .reset_index()
)

unique_words['word_length'] = unique_words['word'].apply(len)

# ------------------------------------------
# Plot 1: Word Length vs Mean RT
# ------------------------------------------

plt.figure(figsize=(10, 6))
sns.lineplot(data=unique_words, x='word_length', y='mean_rt')
plt.title('Word Length (characters) vs Mean RT')
plt.xlabel('Word Length')
plt.ylabel('Mean RT (ms)')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "length_vs_rt.png"), dpi=300)
plt.close()

# ------------------------------------------
# Plot 2: Word Frequency vs Mean RT
# ------------------------------------------

plt.figure(figsize=(10, 6))
sns.scatterplot(data=unique_words, x='frequency', y='mean_rt', alpha=0.5)
plt.xscale('log')
plt.title('Word Frequency vs Mean RT')
plt.xlabel('Word Frequency (log)')
plt.ylabel('Mean RT (ms)')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "frequency_vs_rt.png"), dpi=300)
plt.close()

# ------------------------------------------
# Correlations
# ------------------------------------------

corr_len_freq, _ = pearsonr(unique_words['word_length'], unique_words['frequency'])
corr_len_rt, _ = pearsonr(unique_words['word_length'], unique_words['mean_rt'])
corr_freq_rt, _ = pearsonr(unique_words['frequency'], unique_words['mean_rt'])

print("\n--- Part I Analysis ---")
print(f"Pearson (Length vs Frequency): {corr_len_freq:.4f}")
print(f"Pearson (Length vs Mean RT): {corr_len_rt:.4f}")
print(f"Pearson (Frequency vs Mean RT): {corr_freq_rt:.4f}")

# ==========================================
# Part II: Hypothesis Testing
# ==========================================

def get_gpt_surprisal(words):
    tokenizer = GPT2Tokenizer.from_pretrained(
        'gpt2',
        clean_up_tokenization_spaces=True
    )
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()

    surprisals = []
    with torch.no_grad():
        for w in words:
            inputs = tokenizer(w, return_tensors='pt')
            outputs = model(**inputs, labels=inputs['input_ids'])

            logits = outputs.logits[:, :-1, :]
            labels = inputs['input_ids'][:, 1:]

            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            token_log_probs = torch.gather(
                log_probs, 2, labels.unsqueeze(-1)
            ).squeeze(-1)

            surprisals.append(-token_log_probs.mean().item())

    return surprisals

mean_rt_per_word['gpt_surprisal'] = get_gpt_surprisal(
    mean_rt_per_word['word'].tolist()
)

# ------------------------------------------
# Hypothesis 1
# ------------------------------------------

model1 = smf.ols(
    'mean_rt ~ word_freq + word_length',
    data=mean_rt_per_word
).fit()

model2 = smf.ols(
    'mean_rt ~ gpt_surprisal + word_length',
    data=mean_rt_per_word
).fit()

print("\n--- Part II: Hypothesis 1 ---")
print(f"Model 1 (Frequency) AIC: {model1.aic:.2f}")
print(f"Model 2 (GPT) AIC: {model2.aic:.2f}")

# ------------------------------------------
# Hypothesis 2: Content vs Function
# ------------------------------------------

function_word_tags = {'DT', 'IN', 'CC', 'PRP', 'PRP$', 'MD', 'TO'}

def classify_word(w):
    tag = nltk.pos_tag([w])[0][1]
    return 'function' if tag in function_word_tags else 'content'

mean_rt_per_word['type'] = mean_rt_per_word['word'].apply(classify_word)

content_df = mean_rt_per_word[mean_rt_per_word['type'] == 'content']
function_df = mean_rt_per_word[mean_rt_per_word['type'] == 'function']

m1c = smf.ols('mean_rt ~ word_freq + word_length', content_df).fit()
m2c = smf.ols('mean_rt ~ gpt_surprisal + word_length', content_df).fit()
m1f = smf.ols('mean_rt ~ word_freq + word_length', function_df).fit()
m2f = smf.ols('mean_rt ~ gpt_surprisal + word_length', function_df).fit()

print("\n--- Part II: Hypothesis 2 ---")
print(f"Content: Freq AIC {m1c.aic:.2f} | GPT AIC {m2c.aic:.2f}")
print(f"Function: Freq AIC {m1f.aic:.2f} | GPT AIC {m2f.aic:.2f}")

# ==========================================
# Part III: FOBS Model
# ==========================================

lemmatizer = WordNetLemmatizer()

mean_rt_per_word['lemma'] = mean_rt_per_word['word'].str.lower().apply(
    lemmatizer.lemmatize
)
mean_rt_per_word['lemma_length'] = mean_rt_per_word['lemma'].apply(len)

lemma_freqs = mean_rt_per_word['lemma'].value_counts().to_dict()
mean_rt_per_word['lemma_freq'] = mean_rt_per_word['lemma'].map(lemma_freqs)

model_surface = smf.ols(
    'mean_rt ~ word_freq + word_length',
    mean_rt_per_word
).fit()

model_lemma = smf.ols(
    'mean_rt ~ lemma_freq + lemma_length',
    mean_rt_per_word
).fit()

print("\n--- Part III: Hypothesis 1 ---")
print(f"Surface AIC: {model_surface.aic:.2f}")
print(f"Lemma AIC: {model_lemma.aic:.2f}")

# ------------------------------------------
# Hypothesis 2: Pseudo-affixes
# ------------------------------------------

pseudo_words = ['finger', 'corner', 'hammer']
real_words = ['driver', 'teacher']

subset = mean_rt_per_word[
    mean_rt_per_word['word'].isin(pseudo_words + real_words)
].copy()

subset['is_pseudo'] = subset['word'].isin(pseudo_words)

plt.figure(figsize=(8, 5))
sns.boxplot(data=subset, x='is_pseudo', y='mean_rt')
plt.xticks([0, 1], ['Regular (-er)', 'Pseudo (-er)'])
plt.title('RT: Pseudo-affixes vs Regular Affixes')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "pseudo_vs_real.png"), dpi=300)
plt.close()

t_stat, p_val = ttest_ind(
    subset[subset['is_pseudo']]['mean_rt'],
    subset[~subset['is_pseudo']]['mean_rt']
)

print("\n--- Part III: Hypothesis 2 ---")
print(f"T-statistic: {t_stat:.4f}, p-value: {p_val:.4f}")
