# -*- coding: utf-8 -*-


import torch
# from allennlp.nn import util
from src.classic_models.utils.model_utils import replace_masked_values


class MaxPoolerAggregator(torch.nn.Module):
    """
    A ``MaxPoolerAggregator`` is a max pooling layers.  As a
    :class:`Seq2VecEncoder`, the input to this module is of shape ``(batch_size, num_tokens,
    input_dim)``, and the output is of shape ``(batch_size, output_dim)``, where input_dim == output_dim.

    Parameters
    ----------
    """

    def __init__(self, ) -> None:
        super(MaxPoolerAggregator, self).__init__()

    def forward(self, input_tensors: torch.Tensor, mask: torch.Tensor):  # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        input_tensors : (batch_size, num_tokens, input_dim).
        mask : sentence mask, (batch_size, num_tokens).
        Returns
        -------
        input_max_pooled : torch.FloatTensor
            output_dim   = input_dim
            A tensor of shape ``(batch_size, output_dim)`` .
        """
        if mask is not None:
            # RuntimeError: Subtraction, the `-` operator, with a bool tensor is not supported.
            # If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.
            # mask = mask.bool()
            
            # Simple Pooling layers，将padding的部分设置称为很小的值
            input_tensors = replace_masked_values(
                input_tensors, mask.unsqueeze(2), -1e7
            )
    
        # 取每行最大值（input, dim）
        # 会返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引。
        input_max_pooled = torch.max(input_tensors, 1)[0]

        return input_max_pooled
