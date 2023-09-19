import unittest

import torch

from dynamic_embedding_pruning import EmbeddingPruner


class TestEmbeddingPruner(unittest.TestCase):
    def test_embedding_weight_after_prepare(self) -> None:
        embedding_weight = torch.Tensor(
            [
                [1, 1, 1, 1],
                [2, 2, 2, 2],
                [3, 3, 3, 3],
                [4, 4, 4, 4],
            ],
        )
        embedding = torch.nn.Embedding.from_pretrained(embedding_weight)
        model = torch.nn.Sequential(embedding)

        subject = EmbeddingPruner()
        selected_token_ids = torch.tensor([3, 1])
        subject.prepare(model, embedding, selected_token_ids)

        expected = torch.tensor(
            [
                [4, 4, 4, 4],
                [2, 2, 2, 2],
            ],
        )
        self.assertTrue(torch.equal(expected, embedding.weight))

    @torch.no_grad()
    def test_embedding_weight_after_restore(self) -> None:
        embedding_weight = torch.Tensor(
            [
                [1, 1, 1, 1],
                [2, 2, 2, 2],
                [3, 3, 3, 3],
                [4, 4, 4, 4],
            ],
        )
        embedding = torch.nn.Embedding.from_pretrained(embedding_weight)
        model = torch.nn.Sequential(embedding)

        subject = EmbeddingPruner()
        selected_token_ids = torch.tensor([3, 1])
        model_context = subject.prepare(model, embedding, selected_token_ids)

        embedding.weight[0:2] = torch.tensor(
            [
                [5, 5, 5, 5],
                [6, 6, 6, 6],
            ],
        )

        subject.restore(model_context)

        expected = torch.tensor(
            [
                [1, 1, 1, 1],
                [6, 6, 6, 6],
                [3, 3, 3, 3],
                [5, 5, 5, 5],
            ],
        )
        self.assertTrue(torch.equal(expected, embedding.weight))

    @torch.no_grad()
    def test_state_dict_after_prepare(self) -> None:
        embedding_weight = torch.Tensor(
            [
                [1, 1, 1, 1],
                [2, 2, 2, 2],
                [3, 3, 3, 3],
                [4, 4, 4, 4],
            ],
        )
        embedding = torch.nn.Embedding.from_pretrained(embedding_weight)
        model = torch.nn.Sequential(embedding)

        subject = EmbeddingPruner()
        selected_token_ids = torch.tensor([3, 1])
        subject.prepare(model, embedding, selected_token_ids)

        embedding.weight[0:2] = torch.tensor(
            [
                [5, 5, 5, 5],
                [6, 6, 6, 6],
            ],
        )

        expected = {
            "0.weight": torch.nn.Parameter(
                torch.tensor(
                    [
                        [1, 1, 1, 1],
                        [6, 6, 6, 6],
                        [3, 3, 3, 3],
                        [5, 5, 5, 5],
                    ],
                ),
                requires_grad=False,
            )
        }
        actual = model.state_dict()

        self.assertCountEqual(expected, actual)
        for key in expected:
            self.assertTrue(torch.equal(expected[key], actual[key]))

    def test_load_state_dict_after_prepare(self) -> None:
        embedding_weight = torch.Tensor(
            [
                [1, 1, 1, 1],
                [2, 2, 2, 2],
                [3, 3, 3, 3],
                [4, 4, 4, 4],
            ],
        )
        embedding = torch.nn.Embedding.from_pretrained(embedding_weight)
        model = torch.nn.Sequential(embedding)

        subject = EmbeddingPruner()
        selected_token_ids = torch.tensor([3, 1])
        subject.prepare(model, embedding, selected_token_ids)

        state_dict = {
            "0.weight": torch.nn.Parameter(
                torch.tensor(
                    [
                        [5, 5, 5, 5],
                        [6, 6, 6, 6],
                        [7, 7, 7, 7],
                        [8, 8, 8, 8],
                    ],
                ),
                requires_grad=False,
            )
        }
        model.load_state_dict(state_dict)

        expected = torch.tensor(
            [
                [8, 8, 8, 8],
                [6, 6, 6, 6],
            ],
        )
        self.assertTrue(torch.equal(expected, embedding.weight))
