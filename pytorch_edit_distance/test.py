import torch
import unittest
from numpy.testing import assert_array_almost_equal
from pytorch_edit_distance_cuda import *


class EditDistanceTest(unittest.TestCase):

    blank = torch.tensor([0], dtype=torch.int).cuda()
    separator = torch.tensor([1], dtype=torch.int).cuda()

    def test_repetitions(self):

        x = torch.tensor([[0, 1, 1, 2, 2, 2, 3], [0, 0, 0, 1, 2, 3, 3]]).cuda()
        y = torch.tensor([[0, 1, 1, 2, 2, 2, 3], [0, 1, 2, 3, 2, 3, 3]])

        n = torch.tensor([3, 7], dtype=torch.int).cuda()
        m = torch.tensor([2, 4], dtype=torch.int)

        remove_repetitions(x, n)

        assert_array_almost_equal(x.cpu(), y)
        assert_array_almost_equal(n.cpu(), m)

    def test_blank(self):

        x = torch.tensor([[0, 1, 1, 2, 2, 2, 3], [0, 0, 0, 1, 2, 3, 3]], dtype=torch.int).cuda()
        y = torch.tensor([[1, 1, 2, 2, 2, 2, 3], [1, 2, 3, 3, 2, 3, 3]], dtype=torch.int)

        n = torch.tensor([4, 7], dtype=torch.int).cuda()
        m = torch.tensor([3, 4], dtype=torch.int)

        remove_blank(x, n, self.blank)

        assert_array_almost_equal(x.cpu(), y)
        assert_array_almost_equal(n.cpu(), m)

    def test_strip(self):

        x = torch.tensor([[1, 0, 1, 1, 2, 1, 3], [1, 0, 1, 2, 3, 1, 1]], dtype=torch.int8).cuda()
        y = torch.tensor([[0, 1, 2, 1, 2, 1, 3], [0, 1, 2, 3, 1, 1, 1]], dtype=torch.int8)

        n = torch.tensor([6, 7], dtype=torch.int).cuda()
        m = torch.tensor([3, 4], dtype=torch.int)

        strip_separator(x, n, self.separator.type(torch.int8))

        assert_array_almost_equal(x.cpu(), y)
        assert_array_almost_equal(n.cpu(), m)

    def test_wer(self):

        # hyp: [[A B], [AB]]
        # ref: [[A A B], [A]]

        x = torch.tensor([[1, 0, 1, 1, 2, 1, 3], [1, 0, 1, 2, 3, 1, 1]], dtype=torch.int).cuda()
        y = torch.tensor([[0, 1, 2, 1, 2, 1, 3], [0, 1, 2, 3, 1, 1, 1]], dtype=torch.int).cuda()

        z = torch.tensor([[0, 1, 0, 3], [0, 0, 1, 1]], dtype=torch.int)

        n = torch.tensor([7, 6], dtype=torch.int).cuda()
        m = torch.tensor([7, 3], dtype=torch.int).cuda()

        r = levenshtein_distance(x, y, n, m, self.blank, self.separator)

        assert_array_almost_equal(r.cpu(), z)

    def test_cer(self):

        # hyp: [[A B], [A]]
        # ref: [[A A B], [A B]]

        x = torch.tensor([[1, 0, 1, 1, 2, 1, 3], [0, 1, 2, 3, 1, 1, 1]], dtype=torch.int).cuda()
        y = torch.tensor([[0, 1, 2, 1, 2, 1, 3], [1, 0, 1, 2, 3, 1, 1]], dtype=torch.int).cuda()

        z = torch.tensor([[0, 1, 0, 3], [0, 1, 0, 2]], dtype=torch.int)

        n = torch.tensor([7, 3], dtype=torch.int).cuda()
        m = torch.tensor([7, 5], dtype=torch.int).cuda()

        r = levenshtein_distance(x, y, n, m,
                                 torch.cat([self.blank, self.separator]),
                                 torch.empty([], dtype=torch.int).cuda())

        assert_array_almost_equal(r.cpu(), z)


if __name__ == "__main__":
    unittest.main()
