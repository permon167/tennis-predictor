import unittest
from src.feature_engineering import create_features

class TestFeatureEngineering(unittest.TestCase):

    def test_create_features(self):
        # Example input data
        input_data = {
            'player_1_stats': [1, 2, 3],
            'player_2_stats': [4, 5, 6]
        }
        
        # Expected output data
        expected_output = {
            'feature_1': 7,
            'feature_2': -3
        }
        
        # Call the function to test
        output = create_features(input_data)
        
        # Assert that the output matches the expected output
        self.assertEqual(output, expected_output)

if __name__ == '__main__':
    unittest.main()