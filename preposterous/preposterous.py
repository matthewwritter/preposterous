import pandas as pd
import numpy as np
import scipy.stats
import re
import matplotlib.pyplot as plt


def _datetimelike(s):
    """Test for datetime formatting"""
    return (
            isinstance(s, str)
            and bool(re.search('\d\d\d\d', s))
            and bool(re.search(r'\d\d[.\-:]\d\d', s))
    )


class PrePostDF:
    """Standardize QS data into long timestamped format, and expose convenience functions"""

    def __init__(self):
        """Set up data containers"""

        # Pre/Post df. Key is intervention
        self.pps = {}

        # Index will be datetimes
        self.df = pd.DataFrame(columns=['outcome', 'intervention'], dtype=str)

        # Index will be interventions
        self.confusion_matrix = pd.DataFrame(
            columns=['positive', 'negative', 'other_intervention', 'no_outcomes', 'multiple_outcomes'],
            dtype=int,
        )
        self.confusion_matrix.loc[:, :] = 0

    def add_outcome(self, df=None, filename=None):
        """Add timestamped outcomes

        Code assumes that just one type of outcome is used
        """
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
        elif filename[-4:] == '.csv':
            pass
        else:
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

        # Split rows with multiple interventions
        multi_intervention_ix = df.intervention.fillna('')[lambda x: x.str.contains(',')].index
        mono_intervention_ix = df.index.difference(multi_intervention_ix)
        multi_intervention_rows = []
        for ix, row in df.reindex(multi_intervention_ix).iterrows():
            for intervention in row.intervention.split(','):
                multi_intervention_rows.append(pd.Series(
                    {
                     'intervention': intervention,
                     'outcome': row.outcome,
                    },
                    name=ix
                ))

        if len(multi_intervention_rows):
            df = df.reindex(mono_intervention_ix).append(multi_intervention_rows)

        return df[['intervention', 'outcome']]

    def basic_info(self):
        print("Earliest recording: {:%Y.%m.%d}".format(self.df.index.min()))
        print("Latest recording: {:%Y.%m.%d}".format(self.df.index.max()))
        print("Recordings per day: {:0.1f}".format(self.df.groupby(pd.Grouper(freq='1d')).size().mean()))

        outcomes = self.df.outcome.value_counts()
        outcomes = pd.concat([outcomes, outcomes.div(outcomes.sum())], axis=1, keys=['n', 'pct']).round(2)
        interventions = self.df.intervention.value_counts()
        interventions = pd.concat([interventions, interventions.div(interventions.sum())], axis=1,
                                  keys=['n', 'pct']).round(2)

        print("\nOutcomes with n>5:")
        print(outcomes[outcomes.n > 5])
        print("\nInterventions with n>5:")
        print(interventions[interventions.n > 5])
        print("\nLong dataframe tail:")
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

    def fisher_test(self, intervention, target='Totally fine'):
        """p-value of non-zero effect"""
        mask = self.pps[intervention].sum(axis=1, level=0).applymap(lambda x: x > 0).all(axis=1)
        valid_prepost = self.pps[intervention].loc[mask, :]
        confusion_matrix = valid_prepost.sum(axis=0).unstack().loc[:, ['Noticeable', target]]
        oddsratio, pvalue = scipy.stats.fisher_exact(confusion_matrix.values)
        print('sample size: {:0.0f}'.format(mask.sum()))
        print('p-value of no effect: {:0.2f}'.format(pvalue))

        confusion_matrix['ratio'] = confusion_matrix.iloc[:, 1].div(confusion_matrix.sum(axis=1))
        return confusion_matrix

    def outcomes(self, positive_outcomes, negative_outcomes, interventions=None, window=2):
        """Return dataframe of interventions x outcomes, including invalid experiments"""

        if interventions is None:
            interventions = self.df.intervention.value_counts()[lambda x: x > 5].index
        self.confusion_matrix = self.confusion_matrix.reindex(interventions)

        self.confusion_matrix.loc[:, :] = 0

        for intervention in interventions:
            for dt in self.df[self.df.intervention == intervention].index:
                start = dt + pd.Timedelta(1, 's')
                end = start + pd.Timedelta(window, 'h')
                raw_outcomes = self.df.sort_index().loc[start:end, :].replace('', np.nan)

                # Throw out if any other interventions besides AT the final timestamp
                if raw_outcomes[raw_outcomes.intervention.notnull()].index.max() < raw_outcomes.index.max():
                    self.confusion_matrix.loc[
                        intervention, 'other_intervention'] += 1
                # Throw out if there are no outcomes
                elif raw_outcomes.outcome.notnull().sum() < 1:
                    self.confusion_matrix.loc[intervention, 'no_outcomes'] += 1
                # Throw out if there is more than one type of outcome
                elif raw_outcomes.outcome.dropna().nunique() > 1:
                    self.confusion_matrix.loc[
                        intervention, 'multiple_outcomes'] += 1
                # Negative outcomes
                elif raw_outcomes.outcome.dropna().unique()[0] in negative_outcomes:
                    self.confusion_matrix.loc[intervention, 'negative'] += 1
                # Positive outcomes
                elif raw_outcomes.outcome.dropna().unique()[0] in positive_outcomes:
                    self.confusion_matrix.loc[intervention, 'positive'] += 1
                else:
                    print(raw_outcomes)
                    raise ValueError("Unexpected outcome")
                self.confusion_matrix = self.confusion_matrix.astype('int')

    def _p_distribution(self, intervention, inverse=False, hypotheses=10):
        """Return series of probability hypotheses"""
        posterior = pd.Series(index=np.arange(0, 1, 1. / hypotheses), name=intervention)
        posterior.loc[:] = 1. / hypotheses  # Flat prior
        mask = (self.confusion_matrix.index == intervention)

        # For comparison against all other interventions
        if inverse:
            mask = ~mask

        for p in posterior.index:
            posterior.loc[p] = (
                    posterior.loc[p]  # p(h)
                    * scipy.stats.binom_test(  # p(d|h)
                        x=self.confusion_matrix.loc[mask, ['positive']].sum().sum(),
                        n=self.confusion_matrix.loc[mask, ['positive', 'negative']].sum().sum(),
                        p=p
                    )
            )

        # Divide by p(d) to return p(h|d)
        return posterior.div(posterior.sum())

    def calculate_relative_effectiveness(self, interventions=None):
        """Return dataframe of relative probability hypotheses x interventions"""
        if interventions is None:
            interventions = self.confusion_matrix.index

        relative_effectiveness_rows = []
        for intervention in interventions:
            distribution = self._p_distribution(intervention)
            idistribution = self._p_distribution(intervention, inverse=True)
            for i1 in distribution.index:
                for i2 in idistribution.index:
                    relative_effectiveness_rows.append(
                        [intervention,  # Will be indexed
                         round(i1 - i2, 1),  # Difference between hypotheses, will be indexed
                         (distribution.loc[i1]
                          * idistribution.loc[i2])  # Probability of those two hypotheses
                         ]
                    )

        # Sum the probabilities of all the ways that each difference in effectiveness could happen
        self.relative_effectiveness = pd.DataFrame(
            relative_effectiveness_rows, columns=['intervention', 'delta', 'prob']
        ).groupby(['intervention', 'delta']).sum().unstack(0).round(2)['prob']

    def plot_relative_effectiveness(self):
        """Return and save graphical summary"""
        fig, ax = plt.subplots(1, 1, figsize=(self.relative_effectiveness.shape[1]*3, 7))

        rows = []
        for intervention in self.relative_effectiveness.columns:
            rows.append({
                'x': intervention,
                'ymin': self.relative_effectiveness.loc[:, intervention].cumsum()[lambda x: x > .05].index.min(),
                'ymax': self.relative_effectiveness.loc[:, intervention].cumsum()[lambda x: x < .95].index.max()
            })
        rows = sorted(rows, key=lambda x: x['ymax'])

        for line in rows:
            ax.vlines(**line)

        fig.savefig('relative_effectiveness_{:%Y%m%d}.png'.format(pd.datetime.today()))

        return fig
