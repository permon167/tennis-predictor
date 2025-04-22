import unittest
from src.data_preprocessing import load_data, preprocess_data

class TestDataPreprocessing(unittest.TestCase):

    def test_load_data(self):
        # Test loading data from a file
        data = load_data('data/raw/sample_data.csv')
        self.assertIsNotNone(data)
        self.assertGreater(len(data), 0)

    def test_preprocess_data(self):
        # Test preprocessing of data
        raw_data = load_data('data/raw/sample_data.csv')
        processed_data = preprocess_data(raw_data)
        self.assertIn('processed_column', processed_data.columns)
        self.assertEqual(processed_data.isnull().sum().sum(), 0)

if __name__ == '__main__':
    unittest.main()