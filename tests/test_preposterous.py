from preposterous.preposterous import PrePostDF
import pandas as pd
import numpy as np
from io import StringIO
import sys
import unittest
from unittest.mock import patch


class TestPrePostDF(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range(start='2018.01.01', end='2018.01.03', freq='1h')

        np.random.seed(42)
        self.df = pd.DataFrame({
            'date_column': list(map(str, dates)),
            "outcome": np.random.choice(['Noticeable', 'Totally fine', 'Distracting'],
                                                            size=len(dates)),
            "intervention": np.random.choice(['A', 'B', ''], p=[.1, .1, .8], size=len(dates)),
        })
        self.pdf = PrePostDF()

    def test_import_reporter(self):
        output = PrePostDF.import_reporter(self.df)
        self.assertEqual(output.index.size, 49)

    @patch('preposterous.preposterous.PrePostDF.read_csv')
    def test_basic_info(self, mock_csv):
        mock_csv.return_value = PrePostDF.import_reporter(self.df)
        self.pdf.add_intervention(filename='data/reporter20180729')
        self.pdf.add_outcome(filename='data/reporter20180729')
        capturedOutput = StringIO()          # Create StringIO object
        sys.stdout = capturedOutput                   #  and redirect stdout.
        self.pdf.basic_info()
        sys.stdout = sys.__stdout__                   # Reset redirect.
        self.assertTrue('Latest recording: 2018.01.03' in capturedOutput.getvalue())
        self.assertTrue('Latest recording: 2018.01.03' in capturedOutput.getvalue())

    @patch('pandas.read_csv')
    def test_read_csv(self, mock_csv):
        mock_csv.return_value = self.df.rename(columns={'intervention': 'Choose intervention'})
        output = self.pdf.read_csv('filename')
        self.assertEqual(output.index.size, 49)

    @patch('preposterous.preposterous.PrePostDF.read_csv')
    def test_add_outcome(self, mock_csv):
        mock_csv.return_value = PrePostDF.import_reporter(self.df)
        self.pdf.add_outcome(filename='data/reporter20180729')
        self.assertEqual(self.pdf.df.index.size, 49)

    @patch('preposterous.preposterous.PrePostDF.read_csv')
    def test_add_intervention(self, mock_csv):
        mock_csv.return_value = PrePostDF.import_reporter(self.df)
        self.pdf.add_intervention(filename='data/reporter20180729')
        self.assertEqual(self.pdf.df.index.size, 49)
        self.assertEqual(list(self.pdf.pps.keys()), ['', 'A', 'B'])

    @patch('preposterous.preposterous.PrePostDF.read_csv')
    def test_prepost_generator(self, mock_csv):
        mock_csv.return_value = PrePostDF.import_reporter(self.df)
        self.pdf.add_intervention(filename='data/reporter20180729')
        self.pdf.add_outcome(filename='data/reporter20180729')
        pps = self.pdf._prepost_generator('A')

        self.assertListEqual(pps.columns.tolist(), [('pre', 'Distracting'), ('pre', 'Noticeable'), ('pre', 'Totally fine'), ('post', 'Distracting'), ('post', 'Noticeable'), ('post', 'Totally fine')])
        self.assertSequenceEqual(pps.shape, (8, 6))
        self.assertTrue((pps >= 0).all().all())
        self.assertTrue(pps.notnull().all().all())

    @patch('preposterous.preposterous.PrePostDF.read_csv')
    def test_fisher_test(self, mock_csv):
        mock_csv.return_value = PrePostDF.import_reporter(self.df)
        self.pdf.add_intervention(filename='data/reporter20180729')
        self.pdf.add_outcome(filename='data/reporter20180729')
        self.pdf.pps['A'] = self.pdf._prepost_generator('A')
        ft = self.pdf.fisher_test('A')

        self.assertListEqual(ft.columns.tolist(), ['Noticeable', 'Totally fine', 'ratio'])
        self.assertSequenceEqual(ft.shape, (2, 3))
        self.assertTrue((ft >= 0).all().all())
        self.assertTrue(ft.notnull().all().all())

    @patch('preposterous.preposterous.PrePostDF.read_csv')
    def test_outcomes(self, mock_csv):
        mock_csv.return_value = PrePostDF.import_reporter(self.df)
        self.pdf.add_intervention(filename='data/reporter20180729')
        self.pdf.add_outcome(filename='data/reporter20180729')
        self.pdf.outcomes(positive_outcomes=['Totally fine'], negative_outcomes=['Noticeable', 'Distracting'])

        # Broad test to confirm function outut has not materially changed since last manual review
        self.assertListEqual(self.pdf.confusion_matrix.loc[''].tolist(), [4, 9, 8, 0, 14])
        self.assertListEqual(self.pdf.confusion_matrix.loc['A'].tolist(), [1, 3, 0, 1, 3])
        self.assertListEqual(self.pdf.confusion_matrix.loc['B'].tolist(), [1, 0, 2, 0, 3])

    @patch('preposterous.preposterous.PrePostDF.read_csv')
    def test_p_distribution(self, mock_csv):
        mock_csv.return_value = PrePostDF.import_reporter(self.df)
        self.pdf.add_intervention(filename='data/reporter20180729')
        self.pdf.add_outcome(filename='data/reporter20180729')
        self.pdf.outcomes(positive_outcomes=['Totally fine'], negative_outcomes=['Noticeable', 'Distracting'])
        posterior = self.pdf._p_distribution(intervention='A')

        # Broad test to confirm function outut has not materially changed since last manual review
        self.assertListEqual(posterior.round(1).tolist(), [0.0, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 0.0, 0.0, 0.0])

    @patch('preposterous.preposterous.PrePostDF.read_csv')
    def test_calculate_relative_effectiveness(self, mock_csv):
        mock_csv.return_value = PrePostDF.import_reporter(self.df)
        self.pdf.add_intervention(filename='data/reporter20180729')
        self.pdf.add_outcome(filename='data/reporter20180729')
        self.pdf.outcomes(positive_outcomes=['Totally fine'], negative_outcomes=['Noticeable', 'Distracting'])
        self.pdf.calculate_relative_effectiveness(interventions=['A', 'B'])

        # Broad test to confirm function outut has not materially changed since last manual review
        self.assertListEqual(self.pdf.relative_effectiveness['A'].round(1).tolist(), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.2, 0.2, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    @patch('matplotlib.pyplot.Figure.savefig')
    def test_plot_relative_effectiveness(self, savefig_patch):
        self.pdf.relative_effectiveness = pd.DataFrame({'A':[0, .1, .9]}, index=[.25, .5, .75])
        fig = self.pdf.plot_relative_effectiveness()

        # General test to confirm plot was created, and would have been saved to disk
        savefig_patch.assert_called()


if __name__ == '__main__':
    unittest.main()
