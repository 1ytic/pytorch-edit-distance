# PyTorch edit-distance functions

Useful functions for E2E Speech Recognition training with PyTorch and CUDA.

Here is a simple use case with Reinforcement Learning and RNN-T loss:

```python
blank = torch.tensor([0], dtype=torch.int).cuda()
space = torch.tensor([1], dtype=torch.int).cuda()

xs = model.greedy_decode(xs, sampled=True)

torch_edit_distance.remove_blank(xs, xn, blank)

rewards = 1 - torch_edit_distance.compute_wer(xs, ys, xn, yn, blank, space)

nll = rnnt_loss(zs, ys, xn, yn)

loss = nll * rewards
```

### levenshtein_distance

Levenshtein edit-distance with detailed statistics for ins/del/sub operations.

### collapse_repeated

Merge repeated tokens, useful for CTC-based model.

### remove_blank

Remove unnecessary blank tokens, useful for CTC, RNN-T, RNA models.

### strip_separator

Remove leading, trailing and repeated middle separators.

## Requirements

- C++11 compiler (tested with GCC 9.4.0).
- Python: 3.5, 3.6, 3.7, 3.8, 3.9 (tested with version 3.8).
- [PyTorch](http://pytorch.org/) >= 1.5.0 (tested with version 1.13.1+cu116).
- [CUDA Toolkit](https://developer.nvidia.com/cuda-zone) (tested with version 11.2).

## Install

There is no compiled version of the package. The following setup instructions compile the package from the source code locally.

### From Pypi

```bash
pip install torch_edit_distance
```

### From GitHub

```bash
git clone https://github.com/1ytic/pytorch-edit-distance
cd pytorch-edit-distance
python setup.py install
```

## Test

```bash
python -m torch_edit_distance.test
```
