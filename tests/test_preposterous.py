from preposterous.preposterous import PrePostDF
import pandas as pd
import unittest
from unittest.mock import patch


class TestPrePostDF(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'date_column': ['2018.01.01 00:00:00', '2018.01.02 00:00:00', '2018.01.03 00:00:00'],
            "How does your stomach feel?": ['Noticeable', 'Totally fine', 'Noticeable'],
            "Choose intervention ": ['A', 'A', ''],
        })
        self.pdf = PrePostDF()

    def test_import_reporter(self):
        output = PrePostDF.import_reporter(self.df)
        self.assertEqual(output.index.size, 3)

    @patch('pandas.read_csv')
    def test_read_csv(self, mock_csv):
        mock_csv.return_value = self.df
        output = self.pdf.read_csv('filename')
        self.assertEqual(output.index.size, 3)

    @patch('preposterous.preposterous.PrePostDF.read_csv')
    def test_add_outcome(self, mock_csv):
        mock_csv.return_value = PrePostDF.import_reporter(self.df)
        self.pdf.add_outcome(filename='data/reporter20180729')
        self.assertEqual(self.pdf.df.index.size, 3)

    @patch('preposterous.preposterous.PrePostDF.read_csv')
    def test_add_intervention(self, mock_csv):
        mock_csv.return_value = PrePostDF.import_reporter(self.df)
        self.pdf.add_intervention(filename='data/reporter20180729')
        self.assertEqual(self.pdf.df.index.size, 3)
        self.assertEqual(list(self.pdf.pps.keys()), ['A', ''])


if __name__ == '__main__':
    unittest.main()
