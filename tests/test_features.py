"""
Unit tests for embeddings module
"""

import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.sgns_trainer import _build_token_freq, _generate_skipgram_pairs


class TestEmbeddings(unittest.TestCase):

    def test_token_freq(self):
        """Test token frequency counting"""
        seqs = [[1, 2, 3], [2, 3, 4]]
        freqs = _build_token_freq(seqs, vocab_size=5)
        self.assertEqual(freqs[2], 2)
        self.assertEqual(freqs[3], 2)

    def test_skipgram_pairs(self):
        """Test skip-gram pair generation"""
        seqs = [[1, 2, 3]]
        pairs = _generate_skipgram_pairs(seqs, window_size=1)
        self.assertTrue(len(pairs) > 0)
        self.assertIsInstance(pairs[0], tuple)


if __name__ == '__main__':
    unittest.main()

# ============================================================================
# FILE: tests/test_features.py
# ============================================================================

"""
Unit tests for features module
"""

import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.embedding_extractor import EmbeddingFeatureExtractor


class TestFeatures(unittest.TestCase):

    def test_embedding_pooling(self):
        """Test embedding pooling"""
        # Create dummy embeddings
        embedding_matrix = np.random.rand(10, 50)
        vocab = {'word1': 1, 'word2': 2}

        extractor = EmbeddingFeatureExtractor(embedding_matrix, vocab, pooling='mean')
        token_seqs = [[1, 2], [2, 1]]
        features = extractor.transform(token_seqs)

        self.assertEqual(features.shape[0], 2)
        self.assertEqual(features.shape[1], 50)


if __name__ == '__main__':
    unittest.main()
