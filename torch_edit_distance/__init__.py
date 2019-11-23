import torch
import torch_edit_distance_cuda as core
from pkg_resources import get_distribution

__version__ = get_distribution('torch_edit_distance').version


def collapse_repeated(
        sequences,      # type: torch.Tensor
        lengths         # type: torch.IntTensor
):
    """Merge repeated tokens.
    Sequences and lengths tensors will be modified inplace.

    Args:
      sequences (torch.Tensor): Tensor (N, T) where T is the maximum
        length of tokens from N sequences.
      lengths (torch.IntTensor): Tensor (N,) representing the
        number of tokens for each sequence.
    """
    core.collapse_repeated(sequences, lengths)


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


def compute_wer(hs, rs, hn, rn, blank, space):
    data = levenshtein_distance(hs, rs, hn, rn, blank, space).float()
    wer = data[:, :3].sum(dim=1) / data[:, 3]
    return wer


class AverageWER(object):

    def __init__(self, blank, space, title='WER', detail=2):
        self.blank = blank
        self.space = space
        self.title = title
        self.detail = detail
        self.data = 0

    def update(self, hs, rs, hn, rn):
        data = levenshtein_distance(hs, rs, hn, rn, self.blank, self.space)
        self.data += data.sum(dim=0).float()

    def values(self):

        _ins = self.data[0]
        _del = self.data[1]
        _sub = self.data[2]
        _len = self.data[3]

        _err = (_ins + _del + _sub) / _len * 100

        if self.detail == 2:
            _ins = _ins / _len * 100
            _del = _del / _len * 100
            _sub = _sub / _len * 100

        return _err, _ins, _del, _sub

    def summary(self, writer, epoch):
        _err, _ins, _del, _sub = self.values()
        if self.detail > 0:
            writer.add_scalar(self.title + '/insertions', _ins, epoch)
            writer.add_scalar(self.title + '/deletions', _del, epoch)
            writer.add_scalar(self.title + '/substitutions', _sub, epoch)
        writer.add_scalar(self.title, _err, epoch)

    def __str__(self):
        _err, _ins, _del, _sub = self.values()
        info = '%s %.1f' % (self.title, _err)
        if self.detail == 1:
            info += ' [ %d ins, %d del, %d sub ]' % (_ins, _del, _sub)
        elif self.detail == 2:
            info += ' [ %.1f ins, %.1f del, %.1f sub ]' % (_ins, _del, _sub)
        return info


class AverageCER(AverageWER):

    def __init__(self, blank, space, title='CER', detail=2):
        blank = torch.cat([blank, space])
        space = torch.empty([], dtype=space.dtype, device=space.device)
        super(AverageCER, self).__init__(blank, space, title, detail)
