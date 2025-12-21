"""
Unit tests for preprocessing module
"""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.arabic_cleaner import clean_arabic_text
from src.preprocessing.tokenizer import preprocess_arabic_string


class TestArabicCleaner(unittest.TestCase):

    def test_clean_arabic_text(self):
        """Test basic cleaning"""
        text = "والله شايفينك الناس كلها صابر يا عبد"
        cleaned = clean_arabic_text(text)
        self.assertIsInstance(cleaned, str)
        self.assertTrue(len(cleaned) > 0)

    def test_url_removal(self):
        """Test URL removal"""
        text = "تحقق من http://example.com للمزيد"
        cleaned = clean_arabic_text(text)
        self.assertIn("<URL>", cleaned)

    def test_number_replacement(self):
        """Test number replacement"""
        text = "السعر 100 ريال"
        cleaned = clean_arabic_text(text)
        self.assertIn("<NUM>", cleaned)


class TestTokenizer(unittest.TestCase):

    def test_preprocess_arabic_string(self):
        """Test Arabic string preprocessing"""
        text = "والله!"
        processed = preprocess_arabic_string(text)
        self.assertEqual(processed, "والله")

    def test_tokenize_arabic(self):
        """Test tokenization"""
        # Add test for tokenize_arabic
        pass


if __name__ == '__main__':
    unittest.main()