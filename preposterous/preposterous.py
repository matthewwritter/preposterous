import pandas as pd
import scipy.stats
import re


def _datetimelike(s):
    """Test for datetime formatting"""
    return isinstance(s, str) \
        and bool(re.search('\d\d\d\d', s)) \
        and bool(re.search(r'\d\d[.\-:]\d\d', s))


class PrePostDF:
    """Standardize QS data into long timestamped format, and expose convenience functions"""
    def __init__(self):
        self.pps = {}
        self.df = pd.DataFrame(columns=['outcome', 'intervention'], dtype=str)

    def add_outcome(self, df=None, filename=None):
        """Add timestamped outcomes"""
        if (df is None) and (filename is None):
            raise ValueError("Either df or filename must be specified")
        elif df is not None:
            if not isinstance(df, pd.DataFrame):
                raise ValueError("df must be a dataframe")
            raise NotImplementedError("Only CSV loading currently supported")
        elif filename is not None:
            if not isinstance(filename, str):
                raise ValueError("filename must be a string")
            self.df = pd.concat([self.df, self.read_csv(filename)[['outcome']]], sort=True)

    def add_intervention(self, df=None, filename=None):
        """Add timestamped interventions"""
        if (df is None) and (filename is None):
            raise ValueError("Either df or filename must be specified")
        elif df is not None:
            if not isinstance(df, pd.DataFrame):
                raise ValueError("df must be a dataframe")
            raise NotImplementedError("Only CSV loading currently supported")
        elif filename is not None:
            if not isinstance(filename, str):
                raise ValueError("filename must be a string")
        self.df = pd.concat([self.df, self.read_csv(filename)[['intervention']]], sort=True)

        for intervention in self.df['intervention'].value_counts().index:
            self.pps[intervention] = self._prepost_generator(intervention)

    def read_csv(self, filename):
        """Detect data exporter and call appropriate importer"""
        if filename == 'latest':
            raise NotImplementedError("Automatically finding filenames not currently supported")
        elif filename[-4:] != '.csv':
            filename += '.csv'

        df = pd.read_csv(filename)
        df.columns = df.columns.str.strip()
        if 'Choose intervention' in df.columns:
            return self.import_reporter(df)
        else:
            raise ValueError("Data exporter not recognized for {}. Columns: {}".format(filename, df.columns))

    @staticmethod
    def import_reporter(df):
        """Load and format output from Reporter app"""
        dt_col = df.head().applymap(_datetimelike).sum()[lambda x: x > 0].index[0]
        df['dt'] = pd.to_datetime(df[dt_col])
        df = df.set_index('dt')
        df = df.rename(
            columns={'Choose intervention': 'intervention', 'How does your stomach feel?': 'outcome'})
        df['outcome'] = df['outcome'].apply(lambda x: x.strip() if pd.notnull(x) else '')
        return df[['intervention', 'outcome']]

    def basic_info(self):
        print("Earliest recording: {:%Y.%m.%d}".format(self.df.index.min()))
        print("Latest recording: {:%Y.%m.%d}".format(self.df.index.max()))
        print("Recordings per day: {:0.1f}".format(self.df.groupby(pd.Grouper(freq='1d')).size().mean()))
        print(self.df.tail())

    def _prepost_generator(self, intervention, window=2):
        """Return df formatted for easy pre/post analysis"""
        outcome_list = [x for x in self.df.outcome.fillna('').unique().tolist() if len(x)]
        prepost_intervention = pd.DataFrame(columns=pd.MultiIndex.from_product([['pre', 'post'], outcome_list]))
        for ix in self.df[self.df.intervention == intervention].index:
            for arm, outcome in prepost_intervention.columns:
                if arm == 'pre':
                    prepost_intervention.loc[ix, (arm, outcome)] = self.df[(1 == 1)
                                                                           & (self.df.index > ix - pd.Timedelta(window,
                                                                                                                'h'))
                                                                           & (self.df.index <= ix)
                                                                           & (self.df.outcome == outcome)
                                                                           ].index.size
                elif arm == 'post':
                    prepost_intervention.loc[ix, (arm, outcome)] = self.df[(1 == 1)
                                                                           & (self.df.index > ix)
                                                                           & (self.df.index <= ix + pd.Timedelta(window,
                                                                                                                 'h'))
                                                                           & (self.df.outcome == outcome)
                                                                           ].index.size
                else:
                    raise ValueError('arm should only be pre or post')

        return prepost_intervention

    def generate_confusion_matrix(self, intervention, target='Totally fine'):
        """p-value of non-zero effect"""
        mask = self.pps[intervention].sum(axis=1, level=0).applymap(lambda x: x > 0).all(axis=1)
        valid_prepost = self.pps[intervention].loc[mask, :]
        confusion_matrix = valid_prepost.sum(axis=0).unstack().loc[:, ['Noticeable', target]]
        oddsratio, pvalue = scipy.stats.fisher_exact(confusion_matrix.values)
        print('sample size: {:0.0f}'.format(mask.sum()))
        print('p-value of no effect: {:0.2f}'.format(pvalue))

        confusion_matrix['ratio'] = confusion_matrix.iloc[:, 1].div(confusion_matrix.sum(axis=1))
        return confusion_matrix
