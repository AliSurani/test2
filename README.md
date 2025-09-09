#  2) Semantic Textual Similarity (STS)

## 1) File structure
File structure for this task:
```
 ⁠project_root/
├─ multitask_classifier_sts.py
├─ data/
│  ├─ sts-similarity-dev.csv
│  ├─ sts-similarity-test-student.csv
│  └─ sts-similarity-train.csv
├─ predictions/
│  └─ bert/ 
│     ├─ sts-similarity-dev-output.csv
      └─ sts-similarity-test-output.csv
```

## 2) Execution modes

**Baseline:**
```
python multitask_classifier.py \
--task=sts \
--option=finetune \
--use_gpu
```
**Experiment 1 & 2: *{Pearson Loss + Mean Pooling + Linear NN}***
```
python multitask_classifier_sts.py \
--task=sts --option=finetune --use_gpu \
--sim_head=linear
```
**Experiment 3: *{Pearson Loss + Mean Pooling + MLP}***
```
python multitask_classifier_sts.py \
--task=sts --option=finetune --use_gpu \
--sim_head=mlp
```
**Experiment 4: *{Pearson Loss + Mean Pooling + Deep NN with GELU}***
```
python multitask_classifier_sts.py \
--task=sts --option=finetune --use_gpu \
--sim_head=deep_with_gelu
```
**Experiment 5: *{Pearson Loss + Mean Pooling + Deep NN with ReLU}***
```
python multitask_classifier_sts.py \
--task=sts --option=finetune --use_gpu \
--sim_head=deep_with_relu
```
**Experiment 6: *{Pearson Loss + Mean Pooling + MLP + avg_sentence_embeddings}***
```
python multitask_classifier_sts.py \
--task=sts --option=finetune --use_gpu \
--sim_head=mlp --sim_feats=avg
```
**Experiment 7: *{Pearson Loss + Mean Pooling + Deep NN with GELU + avg_sentence_embeddings}***
```
python multitask_classifier_sts.py \
--task=sts --option=finetune --use_gpu \
--sim_head=deep_with_gelu --sim_feats=avg
```
**Experiment 8: *{Pearson Loss + Mean Pooling + MLP + avg_sentence_embeddings + Residual & Norm layer}***
```
python multitask_classifier_sts.py \
--task=sts --option=finetune --use_gpu \
--sim_head=mlp --sim_feats=avg \
--use_residual --use_norm
```
**Experiment 9: *{Pearson Loss + Mean Pooling + Deep NN with GELU + avg_sentence_embeddings + Residual & Norm layer}***
```
python multitask_classifier_sts.py \
--task=sts --option=finetune --use_gpu \
--sim_head=deep_with_gelu --sim_feats=avg \
--use_residual --use_norm
```
**Experiment 10: *{Pearson Loss + Mean Pooling + MLP + avg_sentence_embeddings + Residual & Norm layer + Contrastive Learning}***
```
python multitask_classifier_sts.py \
--task=sts --option=finetune --use_gpu \
--sim_head=mlp --sim_feats=avg \
--use_residual --use_norm \
--use_contrastive
```
**Experiment 11: *{Pearson Loss + Mean Pooling + Deep NN with GELU + avg_sentence_embeddings + Residual & Norm layer + Contrastive Learning}***
```
python multitask_classifier_sts.py \
--task=sts --option=finetune --use_gpu \
--sim_head=deep_with_gelu --sim_feats=avg \
--use_residual --use_norm \
--use_contrastive
```
**Experiment 12: *{Pearson Loss + Mean Pooling + MLP + avg_sentence_embeddings + Residual & Norm layer + Inbatch Contrastive Learning}***
```
python multitask_classifier_sts.py \
--task=sts --option=finetune --use_gpu \
--sim_head=mlp --sim_feats=avg \
--use_residual --use_norm \
--use_inbatch_contrastive
```
**Experiment 13: *{Pearson Loss + Mean Pooling + Deep NN with GELU + avg_sentence_embeddings + Residual & Norm layer + Inbatch Contrastive Learning}***
```
python multitask_classifier_sts.py \
--task=sts --option=finetune --use_gpu \
--sim_head=deep_with_gelu --sim_feats=avg \
--use_residual --use_norm \
--use_inbatch_contrastive
```
**Experiment 14: *{Pearson Loss + Mean Pooling + Deep NN with GELU + avg_sentence_embeddings + Residual & Norm layer + Contrastive Learning + Linear Scheduler}***
```
python multitask_classifier_sts.py \
--task=sts --option=finetune --use_gpu \
--sim_head=deep_with_gelu --sim_feats=avg \
--use_residual --use_norm \
--use_contrastive --use_linear_scheduler
```
**Experiment 15: *{Pearson Loss + Mean Pooling + Deep NN with GELU + avg_sentence_embeddings + Residual & Norm layer + Contrastive Learning + Cosine Scheduler}***
```
python multitask_classifier_sts.py \
--task=sts --option=finetune --use_gpu \
--sim_head=deep_with_gelu --sim_feats=avg \
--use_residual --use_norm \
--use_contrastive --use_cosine_scheduler
```
**Experiment 16: *{Pearson Loss + Mean Pooling + Deep NN with GELU + avg_sentence_embeddings + Residual & Norm layer + Contrastive Learning + Linear Scheduler + Best Hyperparameters}***
```
python multitask_classifier_sts.py \
--task=sts --option=finetune --use_gpu \
--sim_head=deep_with_gelu --sim_feats=avg \
--use_residual --use_norm \
--use_contrastive --use_linear_scheduler \
--batch_size=16 --lr=0.00003 --hidden_dropout_prob=0.1 
```

## 3) CLI quick reference — QQP
```
# General
--use_gpu                             Use CUDA if available
--seed                                Global random seed (default: 11711)
--epochs                              Training epochs (default: 10)
--lr                                  AdamW learning rate (default: 1e-5)

# Model & Tokenization
--option                              BERT training mode (default: pretrain)
--batch_size                          Samples per update (default: 64)
--max_seq_length                      Maximum token length (default: 128)

# Head (QQP) & Dropout
--hidden_dropout_prob                 Dropout rate for layers (default: 0.3)
--qqp_hidden                          Hidden units in head (default: 512)

# Optimization & Scheduling
--weight_decay                        L2 regularization strength (default: 0.01)
--warmup_ratio                        LR warmup proportion (default: 0.1)
--max_grad_norm                       Gradient clipping threshold (default: 1.0)
--fp16                                Enable mixed-precision training
--grad_ckpt                           Enable gradient checkpointing in BERT

# Loss Controls & Class Imbalance
--label_smoothing                     Binary label smoothing for QQP  (default: 0.0)
--pos_weight                          Positive class weight for BCE; 0 disables (default: 0.0)

# Contrastive (MNRL) Controls
--mnr_weight                          Weight for Multiple Negative Ranking loss (default: 0.2)
--mnr_temperature                     Softmax temperature for MNR (default: 0.05)
--mnr_symmetric                       Apply MNR both directions
--proj_dim                            Projection dim before QQP head/MNR (default: 256)
    
# Stemming Consistency (R-Drop)
--stem_aug                            Use stemming-based consistency loss for QQP
--stemmer                             Stemming algorithm choice (default: porter)
--stem_alpha                          Weight of KL in stemming consistency (default: 1.0)
    
# Approach Selector
--approach                            Choose between 5 approaches for improving QQP Baseline (default: mnrl)
                                      (choices: "mnrl", "optimized_pipeline", "mnrl_with_optimized_pipeline", "sbert", "stemming")

```

## 4) Methodology

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


## 5) Experiments

### **Set 1: Improving Loss Function and Sentence Representation**

#### **Experiment 1: Adding Pearson Loss and Mean Pooling**

**Overview:**

The first improvement over the baseline introduces two critical changes:

  1. Replacing the loss function with a combination of MSE and 1 - Pearson correlation.
  2. Changing the sentence embedding strategy from using `pooler_output` (i.e., the \[CLS] token) to mean pooling over all hidden states.
     
  These changes aim to improve alignment with the STS evaluation metric and provide better semantic representation of sentences.

**Expectations:**

  We expected a substantial increase in Pearson correlation by:

  * Directly optimizing the loss toward the evaluation objective.
  * Capturing richer sentence-level semantics using mean pooling instead of relying solely on the \[CLS] token, which often lacks contextual depth.

**Changes:**

  * Replaced baseline MSE-only loss with a combined loss: `MSE + (1 - Pearson correlation)`
  * Replaced pooler\_output with mean pooling over the last hidden states for both sentences
  * Retained `sim_head = linear` and `sim_feats = base`
  * All other architectural and training parameters (dropout, batch size, LR) remained at default

**Results:**

  * **Baseline dev\_corr:** 0.353
  * **Experiment 1 dev\_corr:** **0.709**

**Discussion:**

The experiment confirms that redefining the loss function to include Pearson correlation plays a major role in improving STS task performance. Unlike MSE, which penalizes absolute error, Pearson correlation rewards predictions that preserve the relative similarity rankings, which aligns much better with human judgment.

Additionally, switching from `pooler_output` to mean pooling significantly enhanced sentence embeddings. While \[CLS] is trained primarily for classification, it often lacks deep context aggregation, especially for semantic similarity. Mean pooling, by averaging all token embeddings, offers a more holistic view of the sentence; capturing nuances and structure that are often missed by the \[CLS] token.

The combination of metric-aligned loss and improved representation is what led to a dramatic increase in dev correlation, setting a strong foundation for further architectural and optimization experiments.


### **Set 2: Neural Network Layer Selection (Similarity Head - `sim_head`)**

**Overview:**

In this set of experiments, we evaluate different architectures for the similarity regression head (`sim_head`) on the Semantic Textual Similarity (STS) task. All other parameters are held constant, and only the `sim_head` configuration is changed.

**Expectation:**

We expected deeper and more expressive architectures to capture richer semantic interactions between sentence pairs, thus improving correlation scores over the simple linear head.

**Changes:**

The only modification across these experiments was the type of `sim_head` used:

* **`linear`**: A simple linear layer.
* **`mlp`**: A 2-layer Multi-Layer Perceptron with ReLU and dropout.
* **`deep_with_gelu`**: A deeper 3-layer feedforward network with GELU activation.
* **`deep_with_relu`**: Same as above, but with ReLU activation.

All other components — sentence embedding strategy, feature vector, and training setup — were kept constant for fair comparison.

**Results:**

| Experiment | `sim_head`       | Dev Correlation |
| ---------: | ---------------- | --------------- |
|      Exp 2 | `linear`         | 0.736           |
|      Exp 3 | `mlp`            | **0.824**       |
|      Exp 4 | `deep_with_gelu` | 0.818           |
|      Exp 5 | `deep_with_relu` | 0.806           |


**Discussion**

* All non-linear heads outperformed the baseline `linear` head, showing that deeper or non-linear architectures are more suitable for the STS task.
* Surprisingly, deeper architectures (`deep_with_gelu`, `deep_with_relu`) underperformed compared to the MLP. This may be due to:
  * Overfitting on limited STS training data.
  * Vanishing gradient issues or insufficient depth-regularization.
* While `mlp` achieved the highest dev correlation, `deep_with_gelu` was close behind and offers more architectural flexibility for future improvements.
* The difference between `deep_with_gelu` and `deep_with_relu` emphasizes the importance of activation choice; smoother GELU activations align better with transformer-based models.
* We decided to carry forward both `mlp` and `deep_with_gelu` heads for the next experiment sets.
* Carring forward both architectures allows us to evaluate trade-offs between complexity, generalization, and extensibility in future experiments (e.g., with residual connections, contrastive losses, etc.).
  

## **Set 3: Sentence Pair Embedding (`sim_feats`)**

**Overview:**

This set evaluates whether enriching the sentence pair feature vector with an additional average component (`avg`) leads to improved semantic similarity prediction. We test this using the top-performing heads (`mlp` and `deep_with_gelu`) from the previous experiment.

**Expectations:**

The hypothesis was that including the average of the two sentence embeddings, along with their element-wise difference and product, would offer a more expressive feature space for similarity modeling, potentially boosting performance.

**Changes:**

* Changed `sim_feats` from `"base"` to `"avg"`
* Continued with the best two heads: `mlp` and `deep_with_gelu`
* Resulting feature vector: `[emb1, emb2, |emb1 - emb2|, emb1 * emb2, (emb1 + emb2)/2]` 

**Results:**

| Experiment | `sim_head`       | `sim_feats` | Dev Correlation |
| ---------: | ---------------- | ----------- | --------------- |
|      Exp 6 | `mlp`            | `avg`       | **0.824**       |
|      Exp 7 | `deep_with_gelu` | `avg`       | 0.818           |


**Discussion:**

Although we expected the addition of the average component (`avg`) to significantly improve the correlation scores due to its ability to capture global semantic similarity between the sentence embeddings, the actual performance remained largely unchanged compared to the `base` feature set. Both `mlp` and `deep_with_gelu` architectures retained their prior performance levels, indicating that the benefit of adding the average term may be architecture-dependent or saturated given the other expressive features already in use (like `|diff|` and `product`).

That said, we still carry forward `sim_feats=avg` for future experiments due to its theoretical merit:

* It provides a global similarity cue that complements the local differences (`|diff|`) and interactions (`product`).
* It introduces no overhead or degradation, maintaining stable performance while potentially offering benefits when combined with other enhancements like contrastive learning or normalization.

In short, while the empirical gains were modest in this isolated experiment, the conceptual richness and future synergy potential justify keeping the `avg` component in our continued architecture.


## **Set 4: Adding Normalization and Residual Layers**

**Overview:**

In this experiment set, we evaluate whether adding normalization (`use_norm`) and residual layers (`use_residual`) improves sentence similarity prediction when applied on top of the best-performing configurations from Set 3 (i.e., `sim_feats=avg` with `mlp` and `deep_with_gelu` heads).

We aim to determine whether:

* LayerNorm helps stabilize and scale the concatenated similarity feature vector.
* Residual connections applied to sentence embeddings (before computing similarity features) help preserve semantic information and improve gradient flow.

**Expectations:**

Given the high capacity of `mlp` and `deep_with_gelu` heads, we hypothesized that:

* Normalization would reduce internal covariate shift and accelerate convergence.
* Residual layers would enhance representation learning by retaining raw features alongside transformed ones, improving expressiveness and stability.

Together, these additions were expected to yield modest but consistent improvements over prior configurations.

**Changes:**

Built directly on top of Set 3, which used:

* `sim_feats = avg` (i.e., `[emb1, emb2, |emb1 - emb2|, emb1 * emb2, (emb1 + emb2)/2]`)
* `sim_head ∈ {mlp, deep_with_gelu}`

For Set 4, we added:

* `use_norm = True` → applies `LayerNorm` over the similarity feature vector.
* `use_residual = True` → adds a `Residual(emb + f(emb))` transformation before computing similarity.

**Results:**

| Experiment | `sim_head`       | `sim_feats` | `use_norm` | `use_residual` | Dev Correlation |
| ---------: | ---------------- | ----------- | ---------- | -------------- | --------------- |
|      Exp 8 | `mlp`            | `avg`       | `True`          | `True`              | **0.830**       |
|      Exp 9 | `deep_with_gelu` | `avg`       | `True`          | `True`              | 0.828           |


**Discussion:**

Both experiments achieved the highest dev Pearson correlation scores so far in the STS task, confirming the benefit of introducing architectural enhancements such as normalization and residual learning.

* Experiment 8 (`mlp`) emerged as the top performer (0.830). The MLP head likely benefits from normalized feature inputs, which prevent saturation of hidden layers and support smoother convergence. Residuals also preserve key semantics from the original embeddings.
* Experiment 9 (`deep_with_gelu`) followed closely (0.828). While deeper layers can benefit more from residuals due to vanishing gradients, the added complexity might slightly reduce it's ability to generalize without further regularization.

The increase from Set 3 (max = 0.824) to Set 4 (max = 0.830) validates this stepwise enhancement.

We carry forward both `mlp` and `deep_with_gelu` architectures with:

* `sim_feats = avg`
* `use_norm = True`
* `use_residual = True`

This decision is based on:

* Empirical improvements in correlation.
* Theoretical grounding in deep learning literature (e.g., LayerNorm and Residuals are known to stabilize and deepen effective capacity).
* Maintaining architectural diversity to support further stages such as contrastive learning, scheduler tuning, and ensemble modeling.


## **Set 5: Contrastive Learning**

**Overview:**

This set investigates whether integrating SimCSE-style contrastive learning can improve sentence similarity modeling for the STS task. The core idea is to regularize sentence representations by encouraging consistency under dropout-induced augmentations, making the embeddings more robust and semantically meaningful.

**Expectations:**

The expectation was that contrastive learning would:

* Encourage intra-sentence consistency by pulling together dropout-augmented views of the same sentence (unsupervised SimCSE)
* Improve sentence-level representation quality, which in turn would benefit pairwise similarity scoring
* Lead to higher Pearson correlation on the STS dev set by making sentence embeddings more informative and aligned with semantic meaning

**Changes:**

This builds directly on Set 4, and adds:

* `use_contrastive = True` → enables SimCSE-style contrastive loss on individual sentence embeddings during STS training.

### **Results:**

| Experiment | `sim_head`       | `sim_feats` | `use_norm` | `use_residual` | `use_contrastive` | Dev Correlation |
| ---------: | ---------------- | ----------- | ---------- | -------------- | ----------------- | --------------- |
|     Exp 10 | `mlp`            | `avg`       | `True`          | `True`              | `True`                 | 0.825           |
|     Exp 11 | `deep_with_gelu` | `avg`       | `True`          | `True`              | `True`                 | **0.834**       |


**Discussion:**

The addition of SimCSE-style contrastive learning had mixed but meaningful results:

* Experiment 11 (`deep_with_gelu`) reached a new highest dev Pearson correlation of 0.834, suggesting that deeper networks like GELU-activated models benefit more from self-regularization techniques like contrastive learning. The deeper architecture is better suited to absorb the noise-based alignment pressure and encode more robust semantics.

* Experiment 10 (`mlp`) saw a slight dip from its previous high (0.830 → 0.825), which might suggest that simpler networks get easily over-regularized by contrastive loss without enough expressive depth to balance representation learning and alignment.

Despite this tradeoff, both models maintained high performance, indicating that contrastive learning is a valuable addition to the pipeline.

We carry forward only `deep_with_gelu` with contrastive learning to the next stages due to:

* Its strongest performance (0.834)
* Compatibility between contrastive learning and deep + nonlinear architectures
* Strong theoretical grounding from SimCSE and related literature, where contrastive pretraining benefits deeper models more prominently

`mlp` will be retained as a baseline but not actively explored in contrastive learning branches.


## **Set 6: In-Batch Contrastive Learning**

**Overview:**

This set explores the effectiveness of in-batch contrastive learning using in-batch negatives (SimCSE-style) on the STS task. Instead of only pulling together positive pairs (as in Set 5), we now push apart unrelated examples within a batch by treating other samples as negatives—expected to better structure the sentence embedding space.

**Expectations:**

In-batch contrastive learning was expected to:

* Further improve the discriminative power of embeddings by encouraging inter-sentence diversity
* Boost semantic similarity prediction by pushing unrelated sentence representations apart
* Outperform the earlier contrastive setting by using real in-batch negatives instead of synthetic dropout noise

**Changes:**

Building on Set 5:

* Enabled `use_inbatch_contrastive = True` to add in-batch contrastive loss
* Disabled `use_contrastive = False` to isolate the effect of in-batch negatives
* Loss used = `MSE + (1 - Pearson) + 0.5 × InBatchContrastiveLoss`

**Results:**

| Experiment | `sim_head`       | `sim_feats` | `use_norm` | `use_residual` | `use_inbatch_contrastive` | Dev Correlation |
| ---------: | ---------------- | ----------- | ---------- | -------------- | ------------------------- | --------------- |
|     Exp 12 | `mlp`            | `avg`       | `True`          | `True`              | `True`                         | 0.787           |
|     Exp 13 | `deep_with_gelu` | `avg`       | `True`          | `True`              | `True`                         | 0.767           |


**Discussion:**

Contrary to expectations, both models experienced noticeable performance drops after introducing in-batch contrastive loss:

* `mlp` dropped from 0.825 → 0.787
* `deep_with_gelu` with GELU dropped from 0.834 → 0.767

This suggests that in-batch contrastive learning, while powerful in pretraining, may not be as effective in our fine-tuning setup on STS:

* STS is a regression task, and contrastive pressure to separate samples might conflict with the smooth, continuous nature of the similarity scores
* In-batch negatives may sometimes include semantically related pairs, leading to conflicting learning signals (i.e., pushing apart similar sentences)
* The model might be over-regularized, especially with limited batch sizes and no curriculum in place to guide negative mining
  
Based on this underperformance, we do not carry forward in-batch contrastive learning in its current form. Despite theoretical appeal, it requires more careful tuning (e.g., better negative sampling, larger batches, pretraining phase) to be effective. Future work may revisit it with hybrid approaches or curriculum learning.

We continue with `deep_with_gelu` using dropout-based SimCSE contrastive loss (from Set 5), as it showed more consistent improvements aligned with the STS task's needs.


## **Set 7: Learning Rate Schedulers**

**Overview:**

This set explores the impact of learning rate scheduling on model performance for the STS task. After locking in the strongest model configuration from earlier sets (Deep NN + GELU + enriched features + dropout-based contrastive learning), we now test whether adjusting the learning rate dynamically through scheduling improves correlation performance.

We use two widely adopted scheduler strategies:

* Linear Warmup Scheduler
* Cosine Warmup Scheduler

**Expectations:**

Scheduling the learning rate is expected to:

* Help avoid sharp convergence and overfitting
* Support gradual stabilization after initial warmup
* Potentially improve generalization by decaying the learning rate smoothly

We expected both schedulers to give better or at least more stable performance than the constant learning rate baseline.

**Changes:**

Building on the best configuration from Set 5:

* `sim_head = deep_with_gelu`
* `sim_feats = avg`
* `use_norm = True`
* `use_residual = True`
* `use_contrastive = True`

Changes:

* Enabled `use_linear_scheduler = True` in Exp 14
* Enabled `use_cosine_scheduler = True` in Exp 15

**Results:**

| Experiment | Scheduler Type   | Dev Correlation |
| ---------: | ---------------- | --------------- |
|     Exp 14 | Linear Scheduler | **0.824**       |
|     Exp 15 | Cosine Scheduler | 0.823           |


**Discussion:**

Both scheduler types performed competitively and nearly matched the earlier best model results (0.834 with no scheduler), but they didn’t offer significant further gains.

* Linear Scheduler slightly outperformed Cosine Scheduler (0.824 vs. 0.823)
* The linear schedule may provide a more stable and gradual decay aligned with our fine-tuning setup, especially since the STS task benefits from steady convergence rather than oscillating updates
* Cosine decay, while smoother and theoretically more robust for longer training, might have caused too rapid decay in our relatively short training schedule

We choose to carry forward the Linear Scheduler for future experiments. It offers:

* Slightly better performance
* Better alignment with controlled learning rate decay
* Simpler tuning and predictable learning behavior

As we move into final hyperparameter tuning, this linear schedule will serve as a reliable training backbone.

---

## 6) Results 
### Hyperparameter Optimization 

To identify the best hyperparameter configuration for the STS task, We performed a greedy search over three important training parameters:

* `batch_size`: Controls how many samples are processed in one iteration. A smaller batch size generally improves generalization but increases training time.
* `learning_rate`: Crucial for fine-tuning large pre-trained models. Small adjustments can lead to significant changes in convergence and final performance.
* `hidden_dropout_prob`: Helps regularize the model and prevent overfitting, especially important when training on smaller datasets like STS.

**Greedy Search Procedure**

We optimized the hyperparameters sequentially instead of trying all combinations (i.e., no full grid search):

**Step 1: Find the best `batch_size`**

* Fixed `learning_rate = 1e-5`, `dropout = 0.3`
* Tried: 64 → 32 → 16

| batch\_size | learning\_rate | dropout | Pearson Corr    |
| ----------- | -------------- | ------- | --------------- |
| 64          | 1e-5           | 0.3     | 0.824           |
| 32          | 1e-5           | 0.3     | 0.828           |
| **16**      | 1e-5           | 0.3     | **0.835**       |

* Best `batch_size` = 16

**Step 2: Find the best `learning_rate`**

* Fixed `batch_size = 16`, `dropout = 0.3`
* Tried: 1e-5 → 2e-5 → 3e-5

| batch\_size | learning\_rate | dropout | Pearson Corr    |
| ----------- | -------------- | ------- | --------------- |
| 16          | 1e-5           | 0.3     | 0.835           |
| 16          | 2e-5           | 0.3     | 0.844           |
| **16**      | **3e-5**       | 0.3     | **0.853**       |

* Best `learning_rate` = 3e-5

**Step 3: Find the best `hidden_dropout_prob`**

* Fixed `batch_size = 16`, `learning_rate = 3e-5`
* Tried: 0.3 → 0.2 → 0.1

| batch\_size | learning\_rate | dropout | Pearson Corr    |
| ----------- | -------------- | ------- | --------------- |
| 16          | 3e-5           | 0.3     | 0.853           |
| 16          | 3e-5           | 0.2     | 0.852           |
| **16**      | **3e-5**       | **0.1** | **0.855**       |

* Best `hidden_dropout_prob` = 0.1

**Final Best Hyperparameters**

| Parameter             | Value     |
| --------------------- | --------- |
| `batch_size`          | 16        |
| `learning_rate`       | 3e-5      |
| `hidden_dropout_prob` | 0.1       |
| **Best Correlation**         | **0.855** |

This methodical process allowed us to isolate the individual contribution of each hyperparameter and reach a configuration that maximized performance while maintaining stability and generalization.

### All Results 

| **Semantic Textual Similarity (STS)** | **Correlation** |
|----------------|-----------|
|Baseline |0.353           | 
|Experiment 1         |0.709|          
|Experiment 2        |0.736|
|Experiment 3        |0.824|
|Experiment 4        |0.818|
|Experiment 5        |0.806|
|Experiment 6        |0.824|
|Experiment 7        |0.818|
|Experiment 8        |0.830|
|Experiment 9        |0.828|
|Experiment 10        |0.825|
|Experiment 11        |0.834|
|Experiment 12        |0.787|
|Experiment 13        |0.767|
|Experiment 14        |0.824|
|Experiment 15        |0.823|
|**Experiment 16**        |**0.855**|


Experiment 1: *{Pearson Loss + Mean Pooling}*

Experiment 2: *{Pearson Loss + Mean Pooling + Linear NN}*

Experiment 3: *{Pearson Loss + Mean Pooling + MLP}*

Experiment 4: *{Pearson Loss + Mean Pooling + Deep NN with GELU}*

Experiment 5: *{Pearson Loss + Mean Pooling + Deep NN with ReLU}*

Experiment 6: *{Pearson Loss + Mean Pooling + MLP + avg_sentence_embeddings}*

Experiment 7: *{Pearson Loss + Mean Pooling + Deep NN with GELU + avg_sentence_embeddings}*

Experiment 8: *{Pearson Loss + Mean Pooling + MLP + avg_sentence_embeddings + Residual & Norm layer}*

Experiment 9: *{Pearson Loss + Mean Pooling + Deep NN with GELU + avg_sentence_embeddings + Residual & Norm layer}*

Experiment 10: *{Pearson Loss + Mean Pooling + MLP + avg_sentence_embeddings + Residual & Norm layer + Contrastive Learning}*

Experiment 11: *{Pearson Loss + Mean Pooling + Deep NN with GELU + avg_sentence_embeddings + Residual & Norm layer + Contrastive Learning}*

Experiment 12: *{Pearson Loss + Mean Pooling + MLP + avg_sentence_embeddings + Residual & Norm layer + Inbatch Contrastive Learning}*

Experiment 13: *{Pearson Loss + Mean Pooling + Deep NN with GELU + avg_sentence_embeddings + Residual & Norm layer + Inbatch Contrastive Learning}*

Experiment 14: *{Pearson Loss + Mean Pooling + Deep NN with GELU + avg_sentence_embeddings + Residual & Norm layer + Contrastive Learning + Linear Scheduler}*

Experiment 15: *{Pearson Loss + Mean Pooling + Deep NN with GELU + avg_sentence_embeddings + Residual & Norm layer + Contrastive Learning + Cosine Scheduler}*

Experiment 16: *{Pearson Loss + Mean Pooling + Deep NN with GELU + avg_sentence_embeddings + Residual & Norm layer + Contrastive Learning + Linear Scheduler + Best Hyperparameters}*


## 7) Visualizations 
Add relevant graphs of your experiments here. Those graphs should show relevant metrics (accuracy, validation loss, etc.) during the training. Compare the  different training processes of your improvements in those graphs. 

For example, you could analyze different questions with those plots like: 
- Does improvement A converge faster during training than improvement B? 
- Does Improvement B converge slower but perform better in the end? 
- etc...

