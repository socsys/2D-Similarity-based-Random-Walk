import torch, sys
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict, Callable
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging


logger = logging.getLogger(__name__)

class SoftmaxLossBert(nn.Module):
    """
    This loss was used in our SBERT publication (https://arxiv.org/abs/1908.10084) to train the SentenceTransformer
    model on NLI data. It adds a softmax classifier on top of the output of two transformer networks.
    :param model: SentenceTransformer model
    :param sentence_embedding_dimension: Dimension of your sentence embeddings
    :param num_labels: Number of different labels
    :param concatenation_sent_rep: Concatenate vectors u,v for the softmax classifier?
    :param concatenation_sent_difference: Add abs(u-v) for the softmax classifier?
    :param concatenation_sent_multiplication: Add u*v for the softmax classifier?
    :param loss_fct: Optional: Custom pytorch loss function. If not set, uses nn.CrossEntropyLoss()
    """
    def __init__(self,
                 model: SentenceTransformer,
                 model_uv: SentenceTransformer | None,
                 multihead_attn,
                 linear_proj_q,
                 linear_proj_k,
                 linear_proj_v,
                 linear_proj_node,
                 sentence_embedding_dimension: int,
                 num_labels: int,
                 concatenation_sent_rep: bool = True,
                 concatenation_sent_difference: bool = True,
                 concatenation_sent_multiplication: bool = False,
                 concatenation_thesis_rep: bool = False,
                 concatenation_uv_rep: bool = True,
                 loss_fct: Callable = nn.CrossEntropyLoss()):
        super(SoftmaxLossBert, self).__init__()
        self.model = model
        self.model_uv = model_uv
        self.multihead_attn = multihead_attn
        self.linear_proj_q = linear_proj_q
        self.linear_proj_k = linear_proj_k
        self.linear_proj_v = linear_proj_v
        self.linear_proj_node = linear_proj_node
        self.num_labels = num_labels
        self.concatenation_sent_rep = concatenation_sent_rep
        self.concatenation_sent_difference = concatenation_sent_difference
        self.concatenation_sent_multiplication = concatenation_sent_multiplication
        self.concatenation_thesis_rep = concatenation_thesis_rep
        self.concatenation_uv_rep = concatenation_uv_rep

        self.model.eval()

        num_vectors_concatenated = 0
        if concatenation_sent_rep:
            num_vectors_concatenated += 2
        if concatenation_sent_difference:
            num_vectors_concatenated += 1
        if concatenation_sent_multiplication:
            num_vectors_concatenated += 1
        if concatenation_thesis_rep:
            num_vectors_concatenated += 1
        if concatenation_uv_rep:
            num_vectors_concatenated += 1
        logger.info("Softmax loss: #Vectors concatenated: {}".format(num_vectors_concatenated))
        self.loss_fct = loss_fct

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        self.model.eval()

        u = None
        if self.concatenation_uv_rep:
            u = self.model_uv(sentence_features.pop())['sentence_embedding']

        v = [self.model.module[0](sentence_feature)['token_embeddings'] for sentence_feature in sentence_features]
        v = torch.mean(v[0], dim=1)

        features = torch.cat([u,v, torch.abs(u-v)], 1)

        output = self.model.module[2](features)

        if labels is not None:
            loss = self.loss_fct(output, labels.view(-1))
            return loss
        else:
            return v, output
        

