import torch
import pytorch_edit_distance_cuda as core
from pkg_resources import get_distribution

__version__ = get_distribution('pytorch_edit_distance').version


def remove_repetitions(
        sequences,      # type: torch.Tensor
        lengths         # type: torch.IntTensor
):
    """Remove repetitions.
    Sequences and lengths tensors will be modified inplace.

    Args:
      sequences (torch.Tensor): Tensor (N, T) where T is the maximum
        length of tokens from N sequences.
      lengths (torch.IntTensor): Tensor (N,) representing the
        number of tokens for each sequence.
    """
    core.remove_repetitions(sequences, lengths)


def remove_blank(
        sequences,      # type: torch.Tensor
        lengths,        # type: torch.IntTensor
        blank           # type: torch.Tensor
):
    """Remove tokens.
    Sequences and lengths tensors will be modified inplace.

    Args:
      sequences (torch.Tensor): Tensor (N, T) where T is the maximum
        length of tokens from N sequences.
      lengths (torch.IntTensor): Tensor (N,) representing the
        number of tokens for each sequence.
      blank (torch.Tensor): A set of tokens to remove.
    """
    core.remove_blank(sequences, lengths, blank)


def strip_separator(
        sequences,      # type: torch.Tensor
        lengths,        # type: torch.IntTensor
        separator       # type: torch.Tensor
):
    """Remove tokens.
    Sequences and lengths tensors will be modified inplace.

    Args:
      sequences (torch.Tensor): Tensor (N, T) where T is the maximum
        length of tokens from N sequences.
      lengths (torch.IntTensor): Tensor (N,) representing the
        number of tokens for each sequence.
      separator (torch.Tensor): A set of tokens to remove as
        leading/trailing tokens as well as repeated middle tokens.
    """
    core.strip_separator(sequences, lengths, separator)


def levenshtein_distance(
        hypotheses,             # type: torch.Tensor
        references,             # type: torch.Tensor
        hypothesis_lengths,     # type: torch.IntTensor
        references_lengths,     # type: torch.IntTensor
        blank,                  # type: torch.Tensor
        separator               # type: torch.Tensor
):
    """Levenshtein edit-distance for separated words or independent tokens.
    Return torch.ShortTensor (N, 4) with detail ins/del/sub/len statistics.

    Args:
      hypotheses (torch.Tensor): Tensor (N, H) where H is the maximum
        length of tokens from N hypotheses.
      references (torch.Tensor): Tensor (N, R) where R is the maximum
        length of tokens from N references.
      hypothesis_lengths (torch.IntTensor): Tensor (N,) representing the
        number of tokens for each hypothesis.
      references_lengths (torch.IntTensor): Tensor (N,) representing the
        number of tokens for each reference.
      blank (torch.Tensor): tokens used to represent the blank symbol.
      separator (torch.Tensor): tokens used to represent the separator symbol.
    """
    assert hypotheses.dim() == 2
    assert references.dim() == 2
    assert hypothesis_lengths.dim() == 1
    assert references_lengths.dim() == 1
    assert hypotheses.size(0) == hypothesis_lengths.numel()
    assert references.size(0) == references_lengths.numel()
    assert hypothesis_lengths.numel() == references_lengths.numel()
    return core.levenshtein_distance(hypotheses, references,
                                     hypothesis_lengths, references_lengths,
                                     blank, separator)


def wer(hs, rs, hn, rn, blank, space):
    operations = levenshtein_distance(hs, rs, hn, rn, blank, space).float()
    error = operations[:, :3].sum(dim=1) / operations[:, 3]
    return error


class AverageWER(object):

    def __init__(self, blank, space, detail=2, title='wer'):
        self.blank = blank
        self.space = space
        self.detail = detail
        self.title = title
        self.operations = 0

    def update(self, hs, rs, hn, rn):
        operations = levenshtein_distance(hs, rs, hn, rn, self.blank, self.space)
        self.operations += operations.sum(dim=0).float()

    def __str__(self):
        _ins = self.operations[0]
        _del = self.operations[1]
        _sub = self.operations[2]
        _len = self.operations[3]
        _err = _ins + _del + _sub
        info = '%s %.1f' % (self.title, _err / _len * 100)
        if self.detail == 1:
            info += ' [ %d ins, %d del, %d sub ]' % (_ins, _del, _sub)
        elif self.detail == 2:
            _ins = _ins / _len * 100
            _del = _del / _len * 100
            _sub = _sub / _len * 100
            info += ' [ %.1f ins, %.1f del, %.1f sub ]' % (_ins, _del, _sub)
        return info


class AverageCER(AverageWER):

    def __init__(self, blank, space, detail=2, title='cer'):
        blank = torch.cat([blank, space])
        space = torch.empty([], dtype=space.dtype, device=space.device)
        super(AverageCER, self).__init__(blank, space, detail, title)
