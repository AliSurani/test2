
# Methodology

This task focuses on improving semantic textual similarity (STS) performance using a customized multitask BERT architecture. Our primary goal is to explore, implement, and evaluate various architectural and optimization strategies within a unified and extensible pipeline that supports both baseline and advanced methods via configurable command-line arguments. The overarching methodology is to systematically extend the standard BERT model into a flexible research platform for sentence-level similarity tasks, leveraging modern advancements in neural architectures and loss functions.

### **Overall Pipeline Design**

We extend the existing BERT-based multitask classifier into a modular architecture that supports a wide range of configurable options. These modifications allow us to control different aspects of the model, such as feature representations, similarity head architectures, regularization techniques, contrastive learning objectives, and training dynamics. All improvements are implemented in a single training script, with runtime configurability via CLI flags.

### **Core Enhancements and Architecture Components**

#### 1. **Similarity Head (`--sim_head`)**

A new argument `--sim_head` controls the architecture of the similarity regressor. Four types are supported:

* `"linear"`: A simple linear projection layer.
* `"mlp"`: A shallow MLP with one hidden layer and ReLU activation.
* `"deep_with_gelu"`: A deeper feed-forward network using GELU activation for richer modeling.
* `"deep_with_relu"`: Same as above, but with ReLU activation.

These enable controlled experiments on the effect of shallow vs. deep architectures on STS performance.

#### 2. **Feature Representation (`--sim_feats`)**

The `--sim_feats` flag controls how the sentence pair embeddings are combined before similarity regression:

* `"base"`: Concatenation of `[emb1, emb2, |emb1 - emb2|, emb1 * emb2]`
* `"avg"`: Same as base, but also adds `(emb1 + emb2) / 2` for additional relational information.

This allows richer modeling of similarity through explicit vector arithmetic features.

#### 3. **Residual Connection (`--use_residual`)**

We introduce an optional residual layer to improve gradient flow and representation stability. This applies a linear transformation followed by ReLU and adds it back to the original embeddings before computing similarity.

#### 4. **Feature Normalization (`--use_norm`)**

To stabilize training and standardize feature distributions, we apply `LayerNorm` after concatenating the similarity feature vector. This is controlled via `--use_norm`.


### **Optimization Enhancements**

#### 5. **Learning Rate Scheduling**

To improve convergence and generalization, we add support for:

* `--use_linear_scheduler`: Applies linear warm-up and decay.
* `--use_cosine_scheduler`: Applies cosine decay with warm-up.

Only one can be used at a time. Warmup steps are set to 10% of the total training steps.

#### 6. **Loss Function Customization**

We extend the loss function to optimize for Pearson correlation (our evaluation metric):

* Final loss = `MSE + (1 - Pearson)` with optional contrastive components.
* The contrastive component is scaled using `--contrastive_loss_weight` (default = 0.5).

#### 7. **Contrastive Learning (`--use_contrastive`)**

To improve the embedding space quality, we implement SimCSE-style contrastive loss:

* For each input sentence, two views are created using dropout noise.
* Cosine embedding loss is computed between them, encouraging similar representations.
* This is applied in addition to the primary loss function.

#### 8. **In-Batch Contrastive Learning (`--use_inbatch_contrastive`)**

To further enhance representation learning using hard negatives, we implement in-batch contrastive loss:

* Sentence embeddings from both inputs are compared with all others in the batch.
* Cross-entropy loss is computed using similarity matrix with a temperature-scaled dot product.
* This introduces stronger signal for learning discriminative embeddings.

#### **9. Hyperparameter Search for Best Model**

To further optimize the best-performing model, we implement hyperparameter search across key parameters:

* Learning Rate (`lr`)
* Hidden Dropout Probability (`hidden_dropout_prob`)
* Batch Size (`batch_size`)


### **Unified Experimentation Platform**

All enhancements are modularized and can be turned on or off via CLI arguments. This enables rapid experimentation with different architectural and training strategies without changing the core code structure. The final goal is to discover an optimal combination that significantly boosts STS correlation compared to the baseline model.

---


# Experiments
Keep track of your experiments here. What are the experiments? Which tasks and models are you considering?

Write down all the main experiments and results you did, even if they didn't yield an improved performance. Bad results are also results. The main findings/trends should be discussed properly. Why a specific model was better/worse than the other?

You are **required** to implement one baseline and improvement per task. Of course, you can include more experiments/improvements and discuss them. 

You are free to include other metrics in your evaluation to have a more complete discussion.

Be creative and ambitious.

For each experiment answer briefly the questions:

- What experiments are you executing? Don't forget to tell how you are evaluating things.
- What were your expectations for this experiment?
- What have you changed compared to the base model (or to previous experiments, if you run experiments on top of each other)?
- What were the results?
- Add relevant metrics and plots that describe the outcome of the experiment well. 
- Discuss the results. Why did improvement _A_ perform better/worse compared to other improvements? Did the outcome match your expectations? Can you recognize any trends or patterns?

## Results 
Summarize all the results of your experiments in tables:

| **Stanford Sentiment Treebank (SST)** | **Metric 1** |**Metric n** |
|----------------|-----------|------- |
|Baseline |45.23%           |...            | 
|Improvement 1          |58.56%            |...          
|Improvement 2        |52.11%|...|
|...        |...|...|

| **Quora Question Pairs (QQP)** | **Metric 1** |**Metric n** |
|----------------|-----------|------- |
|Baseline |45.23%           |...            | 
|Improvement 1          |58.56%            |...          
|Improvement 2        |52.11%|...|
|...        |...|...|

| **Semantic Textual Similarity (STS)** | **Metric 1** |**Metric n** |
|----------------|-----------|------- |
|Baseline |45.23%           |...            | 
|Improvement 1          |58.56%            |...          
|Improvement 2        |52.11%|...|
|...        |...|...|

| **Paraphrase Type Detection (PTD)** | **Metric 1** |**Metric n** |
|----------------|-----------|------- |
|Baseline |45.23%           |...            | 
|Improvement 1          |58.56%            |...          
|Improvement 2        |52.11%|...|
|...        |...|...|

| **Paraphrase Type Generation (PTG)** | **Metric 1** |**Metric n** |
|----------------|-----------|------- |
|Baseline |45.23%           |...            | 
|Improvement 1          |58.56%            |...          
|Improvement 2        |52.11%|...|
|...        |...|...|

Discuss your results, observations, correlations, etc.

Results should have three-digit precision.
 

### Hyperparameter Optimization 
Describe briefly how you found your optimal hyperparameter. If you focussed strongly on Hyperparameter Optimization, you can also include it in the Experiment section. 

_Note: Random parameter optimization with no motivation/discussion is not interesting and will be graded accordingly_

## Visualizations 
Add relevant graphs of your experiments here. Those graphs should show relevant metrics (accuracy, validation loss, etc.) during the training. Compare the  different training processes of your improvements in those graphs. 

For example, you could analyze different questions with those plots like: 
- Does improvement A converge faster during training than improvement B? 
- Does Improvement B converge slower but perform better in the end? 
- etc...

## Members Contribution 
Explain what member did what in the project:

**Member 1:** _implemented the training objective using X, Y, and Z. Supported member 2 in refactoring the code. Data cleaning, etc._

**Member 2:** ...

...

We should be able to understand each member's contribution within 5 minutes. 

# AI-Usage Card
Artificial Intelligence (AI) aided the development of this project. Please add a link to your AI-Usage card [here](https://ai-cards.org/).

# References 
Write down all your references (other repositories, papers, etc.) that you used for your project.


