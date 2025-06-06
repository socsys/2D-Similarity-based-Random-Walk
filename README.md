# 2D Similarity-based Random Walk for Improved Understanding of Online Conversations

**Zaid Almahmoud**, Vibhor Agarwal, Rana Mahmoud, and Nishanth Sastry, "Modeling discourse structure with 2D Similarity-based Random Walks for improved understanding of online conversations", 2025.

## Abstract

The proliferation of social media has made automated classification of online discourse, such as hate speech detection and polarity prediction, an essential task for maintaining digital safety and constructive discussions. However, online conversations are complex, context-dependent, and often structured in branching discourse trees, making such tasks especially challenging. Existing classification models typically rely on limited context sampling strategies that overlook the broader structure of discussions and the semantic relevance to the target utterance. In this paper, we introduce the 2D Similarity-based Random Walk, a novel context sampling method that explores multiple paths within discourse graphs to capture more comprehensive and relevant context. Through empirical analysis, we show that the proposed 2D-walk covers a wider range of branches in the discourse while traversing longer or comparable paths within those branches compared to traditional single-path (1D) baseline. This leads to samples with greater semantic relevance to the target utterance and a higher number of utterances per sample. We evaluate our method on two benchmark datasets: Guest, for detecting misogynistic hate speech, and Kialo, for predicting the polarity of argumentative replies. Extensive experiments using GPT-4, a Multi-Head Attention model, and BERT confirm that 2D-walk consistently improves classification performance across both datasets compared to the 1D-walk and the no-context (0D) baselines. Our findings underscore the importance of discourse-aware sampling and suggest that leveraging both structural and semantic relationships in conversation graphs is key to advancing robust language understanding in social media contexts.

The paper PDF is available [here]().

## Citation

```

```
****
