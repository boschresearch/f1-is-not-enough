#  Answer-Fact Coupling Regularizer

We base our implementation on the [code published by Qi et al.](https://github.com/qipeng/golden-retriever)
which in turn is based on [the HotpotQA baseline model](https://github.com/hotpotqa/hotpot).


In particular, we follow their [model interface definition](https://github.com/qipeng/golden-retriever/blob/master/BiDAFpp/run.py).
Therefore, all our models return four variables: 
* `logit1`: Answer span start logits (batch size $\times$ length of context)
* `logit2`: Answer span end logits (batch size $\times$ length of context)
* `predict_type`: Answer type (yes/no/span)(batch size $\times$ 3)
* `predict_support`: Supporting fact (i.e. explanation) logits (batch size $\times$ number of facts $\times$ 2)

### Standard Loss
Given these variables, we calculate the answer loss (`loss_1`) and the explanation loss (`loss_2`) following the [code published by Qi et al.](https://github.com/qipeng/golden-retriever/blob/master/BiDAFpp/run.py) and their definitions of the ground truth variables `y1`, `y2` and `is_support`.
```python
loss_1 = (nll_sum(predict_type, q_type) + nll_sum(logit1, y1) + nll_sum(logit2, y2)) / context_idxs.size(0)
loss_2 = nll_average(predict_support.view(-1, 2), is_support.view(-1))
```

###  Answer-Fact Coupling Loss
In addition, we calculate our answer-fact coupling regularization term as defined in Equation 1 in [our paper](https://www.aclweb.org/anthology/2020.emnlp-main.575/).
If `use_sp_prob_mult` is true, the loss is calculated following our definition of $p_e^*$, if not, $p_e^+$ is used.
We determine the values of $c_1$, $c_2$ and $c_3$ within a hyperparameter search which we discuss in Appendix C of [our paper](https://www.aclweb.org/anthology/2020.emnlp-main.575/).
```python
# p_a
p_start = torch.nn.functional.softmax(logit1, dim=-1)
p_end = torch.nn.functional.softmax(logit2, dim=-1)
p_start_token = p_start[range(y1.size(0)), y1]
p_end_token = p_end[range(y2.size(0)), y2]
p_answer = p_start_token * p_end_token

# p_e
predict_support_probabilities = torch.nn.functional.softmax(predict_support, dim=-1)[:, :, 1]
bs = predict_support_probabilities.size(0)
p_explanation = torch.zeros(bs).cuda()
for b_idx in range(bs):
    gt_support_idxs = torch.nonzero(is_support[b_idx] == 1, as_tuple=True)
    if use_sp_prob_mult:
        # Mult
        p_explanation[b_idx] = torch.prod(predict_support_probabilities[b_idx][gt_support_idxs])
    else:
        # Add
        p_explanation[b_idx] = torch.sum(predict_support_probabilities[b_idx][gt_support_idxs])

coupling_induced_cost = p_answer * ((1.0 - p_explanation) * c_1)\
                        + (1.0 - p_answer) * (p_explanation * c_2 + (1.0 - p_explanation) * c_3)
# Average over batch
answer_fact_coupling_loss = torch.sum(coupling_induced_cost) / bs
```

Finally, we combine the standard loss terms with our answer-fact coupling loss.
`sp_lambda` is used to trade-off the importance between predicting the correct answer and the ground truth explanation.
We do not introduce an additional weigthing factor for `answer_fact_coupling_loss` as it can directly be absorbed into the choice of $c_i$.

```python
loss = loss_1 + sp_lambda * loss_2 + answer_fact_coupling_loss
```