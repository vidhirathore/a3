# Assignment Report

## Computational Psycholinguistics Assignment-3: Word Processing

**Author:** Vidhi Rathore (2023121002)  
**Institution:** IIIT Hyderabad


https://github.com/vidhirathore/a3
---

## Contents
1. **Part I: Preliminary Data Analysis**
   - 1.1. Objective
   - 1.2. Data and Aggregation
   - 1.3. Results
   - 1.4. Interpretation
   - 1.5. Reproducibility
2. **Part II: Hypothesis Testing**
   - 2.1. Data Preparation
   - 2.2. Hypothesis 1
   - 2.3. Hypothesis 2
   - 2.4. Discussion
   - 2.5. Reproducibility
3. **Part III: Frequency Ordered Bin Search (FOBS)**
   - 3.1. FOBS Setup
   - 3.2. Hypothesis 1
   - 3.3. Hypothesis 2
   - 3.4. Part III Note
   - 3.5. Reproducibility
4. **Submission Details**
   - 4.1. Repository Link
   - 4.2. Deliverables Checklist
   - 4.3. Reproducibility Note

---

## 1. Part I: Preliminary Data Analysis
This part studies the relationship between word length, word frequency, and mean reading time per word in the Natural Stories corpus.

### 1.1. Objective
The required analyses for Part I are:
* Compute mean RT per word across subjects,
* Plot word length vs mean RT,
* Plot word frequency vs mean RT,
* Compute Pearson correlations for the three required variable pairs,
* Summarize the observed relationships.

### 1.2. Data and Aggregation
**Input files used:**
* `naturalstories/naturalstories_RTS/processed_RTs.tsv`
* `naturalstories/freqs/freqs-1.tsv`

For each token identified by `(item, zone, word)` in the RT data, mean RT was computed across all available subjects. Frequency values were joined using token IDs from `freqs-1.tsv` with code format `item.zone.word`. After joining, 10,246 tokens were used for analysis (10,256 RT tokens before join; 10 unmatched).

### 1.3. Results
#### 1.3.1. Word Length vs Mean RT

![Figure 1: Scatter plot showing Word Length (characters) on the X-axis vs Mean RT per word (ms) on the Y-axis with a positive regression line.](figure1_word_length_vs_rt.png)

The scatter in Figure 1 shows a positive tendency: longer words are associated with slower reading times.

#### 1.3.2. Word Frequency vs Mean RT

![Figure 2: Scatter plot showing Word Frequency (log scale) on the X-axis vs Mean RT per word (ms) on the Y-axis with a negative regression line.](figure2_word_freq_vs_rt.png)

The trend in Figure 2 is negative: more frequent words tend to have lower mean RT.

#### 1.3.3. Pearson Correlation Coefficients

| Variable Pair | Pearson r | p-value | N |
| :--- | :--- | :--- | :--- |
| Word length vs frequency | -0.3990 | 0 (underflow) | 10246 |
| Word length vs mean RT | 0.2462 | 2.61e-141 | 10246 |
| Word frequency vs mean RT | -0.1318 | 6.43e-41 | 10246 |

### 1.4. Interpretation
The analyses indicate three consistent effects:
* **Word length and frequency** are inversely related ($r = -0.3990$): longer words are generally less frequent.
* **Word length and mean RT** are positively related ($r = 0.2462$): longer words tend to be read more slowly.
* **Word frequency and mean RT** are negatively related ($r = -0.1318$): frequent words tend to be read faster.

These directions match standard psycholinguistic expectations for self-paced reading data.

Here is your text cleanly converted into **proper Markdown format** (with fixed tables, headings, code blocks, and image embedding).

You can paste this directly into a `.md` file.

---

# 2. Part II: Hypothesis Testing

This part compares frequency-based and language-model-based predictors for reading time, using linear regression at the word level.

---

## 2.1. Data Preparation

The Part II dataset extends Part I features with:

* **GPT-3 surprisal per word**, computed as
  `-sum(logprob)` over GPT tokens aligned to each story word using character offsets.
* **Word class label** (content or function) derived from aligned UD parse POS tags.

### Rows Used

* Full merged dataset: **10246**
* Hypothesis 1 modeling rows: **10238**
* Hypothesis 2 content rows: **5910**
* Hypothesis 2 function rows: **4328**

---

## 2.2. Hypothesis 1

**Hypothesis:** Language-model probability is a better predictor of reading time than word frequency.

### Models Compared

* **Model 1:**
  `mean_rt ~ frequency + word_length`

* **Model 2:**
  `mean_rt ~ gpt3_surprisal + word_length`

---

### 2.2.1. Fit Comparison

| Model                           | Adjusted R² | AIC       | RMSE  | N     |
| ------------------------------- | ----------- | --------- | ----- | ----- |
| Model 1 (frequency + length)    | 0.0651      | 104673.24 | 40.16 | 10238 |
| Model 2 (LM surprisal + length) | 0.0772      | 104539.93 | 39.90 | 10238 |

**Winner:** Model 2 (LM surprisal + word length), based on lower AIC and higher adjusted R².

![Hypothesis 1 Comparison](figure3_h1_comparison.png)

---

## 2.3. Hypothesis 2

**Hypothesis:** Content words are processed differently than function words.

### Models Compared

* **Model 1:**
  `mean_rt (content) ~ frequency + word_length`

* **Model 2:**
  `mean_rt (content) ~ gpt3_surprisal + word_length`

* **Model 3:**
  `mean_rt (function) ~ frequency + word_length`

* **Model 4:**
  `mean_rt (function) ~ gpt3_surprisal + word_length`

---

### 2.3.1. Fit Comparison

| Model                                     | Adjusted R² | AIC      | N    |
| ----------------------------------------- | ----------- | -------- | ---- |
| Model 1: content (frequency + length)     | 0.0830      | 61289.16 | 5910 |
| Model 2: content (LM surprisal + length)  | 0.0938      | 61218.94 | 5910 |
| Model 3: function (frequency + length)    | 0.0287      | 43062.64 | 4328 |
| Model 4: function (LM surprisal + length) | 0.0348      | 43035.60 | 4328 |

### Winners by Subset

* **Content words:** Model 2 (LM surprisal + length)
* **Function words:** Model 4 (LM surprisal + length)

![Hypothesis 2 Comparison](figure4_h2_comparison.png)

---

## 2.4. Discussion

The results support both hypotheses:

* For all words, LM surprisal improves fit over raw frequency (Hypothesis 1).
* Content and function subsets show different fit magnitudes, and both are better explained by LM surprisal than by frequency (Hypothesis 2).

This indicates that contextual predictability captured by LM probabilities explains reading-time variance beyond corpus frequency alone.

---


---

# 3. Part III: Frequency Ordered Bin Search (FOBS)

This part evaluates whether lemma-level frequency organization improves reading-time prediction over surface-form frequency, and whether pseudo-affixed words incur extra processing cost.

---

## 3.1. FOBS Setup

For each token, we used:

* Mean RT from `processed_RTs.tsv`
* Surface frequency from `freqs-1.tsv`
* POS from aligned UD parses (`stories-aligned.conllx`) for POS-aware lemmatization

Lemmas were produced with WordNet lemmatization.

For both surface forms and lemmas, we built:

* Frequency-ordered ranks
* **FOBS depth**:
  `FOBS depth = ceil(log2(rank + 1))`

---

## 3.2. Hypothesis 1

**Hypothesis:** Root (lemma) frequency predicts reading time better than surface frequency.

### Models Compared

* **Model 1:**
  `mean_rt ~ frequency + word_length`

* **Model 2:**
  `mean_rt ~ lemma_frequency + lemma_length`

---

### 3.2.1. Fit Comparison

| Model                               | Adjusted R² | AIC       | RMSE  | N     |
| ----------------------------------- | ----------- | --------- | ----- | ----- |
| Model 1: surface freq + word length | 0.0434      | 105242.09 | 41.12 | 10246 |
| Model 2: lemma freq + lemma length  | 0.0416      | 105261.29 | 41.16 | 10246 |

**Result:** Model 1 performs slightly better (higher adjusted R², lower AIC, lower RMSE).
In this dataset, surface frequency predicts mean RT marginally better than lemma frequency.

![FOBS Hypothesis 1](figure5_h1_fobs_metrics.png)

---

## 3.3. Hypothesis 2

**Hypothesis:** Pseudo-affixed words (e.g., *finger*) are processed slower than regularly affixed words (e.g., *driver*-type words).

### Task Setup

* Built candidate real/pseudo affix words via suffix-based heuristics and lemmatization.
* Selected 5 matched pairs controlling approximately for word length and frequency.
* Tested RT difference (pseudo − real) using:

  * Paired t-test
  * Wilcoxon signed-rank test

---

### Selected Pairs

* finger vs lesser
* summer vs larger
* silver vs bigger
* father vs higher
* sister vs better

---

### Summary Statistics

* Mean RT (pseudo): **367.27 ms**
* Mean RT (real): **333.21 ms**
* Mean difference: **34.07 ms**
* Paired t-test p-value: **0.1084**
* Wilcoxon p-value: **0.1875**

**Interpretation:**
Pseudo-affixed words are slower on average, but with only 5 pairs the effect is not statistically significant.

![Pseudo Affix Comparison](figure6_h2_pseudo_affix.png)

---

## 3.4. Part III Note

Findings:

* Hypothesis 1 is **not supported**: lemma frequency does not outperform surface frequency.
* Hypothesis 2 is **directionally supported**, but evidence is weak due to small sample size.

Overall, surface-form frequency provides stronger predictive power than lemma aggregation in this dataset.
