# from sys import stdout
# import shutil
import time
import os

# import warnings
import random
import itertools
from operator import itemgetter

import jellyfish as jf
import swifter
from scipy.optimize import linear_sum_assignment
# from scipy.stats import wasserstein_distance

from model_prediction.analyzers import alpha_oracle as ao
from model_prediction.analyzers.alpha_oracle import Rel


import pandas as pd
import numpy as np


class Evaluator():

    def __init__(self, one_timestamp, variant, next_mode, mode):
        """constructor"""
        self.one_timestamp = one_timestamp
        self.variant = variant
        self.next_mode = next_mode
        self.mode = mode
        # print("onetimestamp :", self.one_timestamp)
        # print("Variant :", self.variant)
        # print("Mode : ", self.mode)

    def measure(self, metric, data, feature=None):
        evaluator = self._get_metric_evaluator(metric)
        return evaluator(data, feature)

    def _get_metric_evaluator(self, metric):
        if metric == 'accuracy':
            if self.mode in ['next']:
                return self._accuracy_evaluation
            elif self.mode in ['batch']:
                return self._accuracy_evaluation_batch
        if metric == 'mae_next':
            if self.mode in ['next']:
                return self._mae_next_evaluation
            elif self.mode in ['batch']:
                return self._mae_next_evaluation_batch
        elif metric == 'similarity':
            return self._similarity_evaluation
        elif metric == 'mae_suffix':
            return self._mae_remaining_evaluation
        elif metric == 'els':
            return self._els_metric_evaluation
        elif metric == 'els_min':
            return self._els_min_evaluation
        elif metric == 'mae_log':
            return self._mae_metric_evaluation
        elif metric == 'dl':
            return self._dl_distance_evaluation
        else:
            raise ValueError(metric)

    def _accuracy_evaluation(self, data, feature):
        data = data.copy()
        #print("Data Initially in Accuracy Meaurement :", data)
        data = data[[(feature + '_expect'), (feature + '_pred'),
                     'run_num', 'implementation']]
        #accuracy evaluation exp = pred then 1 else 0 in case of multi_pred it checks if the predicted value is in the list then it sets it to 1
        if self.variant in ['multi_pred']:
            eval_acc = (lambda x:
                        1 if x[feature + '_expect'] in x[feature + '_pred'] else 0)
        else:
            eval_acc = (lambda x:
                        1 if x[feature + '_expect'] == x[feature + '_pred'] else 0)
        #print("Data After Accuracy Measurement:", data)
        data[feature + '_acc'] = data.apply(eval_acc, axis=1)

        # agregate true positives
        data = (data.groupby(['implementation', 'run_num'])[feature + '_acc']
                .agg(['sum', 'count'])
                .reset_index())

        # calculate accuracy
        data['accuracy'] = np.divide(data['sum'], data['count'])

        #print("data accuracy 5:", data)
        return data

    def _accuracy_evaluation_batch(self, data, feature):
        data = data.copy()
        #print("Data Initially in Accuracy Meaurement :", data)
        data = data[[(feature + '_expect'), (feature + '_pred'),
                     'caseid']]
        eval_acc = (lambda x:
                    1 if x[feature + '_expect'] == x[feature + '_pred'] else 0)
        #print("Data After Accuracy Measurement:", data)
        data[feature + '_acc'] = data.apply(eval_acc, axis=1)

        # agregate true positives
        data = (data.groupby(['caseid'])[feature + '_acc']
                .agg(['sum', 'count'])
                .reset_index())

        # calculate accuracy
        data['accuracy'] = np.divide(data['sum'], data['count'])

        #print("data accuracy 5:", data)
        return data

    def _mae_next_evaluation(self, data, feature):
        data = data.copy()
        # print("Inside MAE : ", data[(feature + '_pred')])
        if self.next_mode in ['next_action']:
            data[(feature + '_pred')] = np.mean(data[(feature + '_pred')].tolist(), axis=1)
        data = data[[(feature + '_expect'), (feature + '_pred'),
                     'run_num', 'implementation']]
        ae = (lambda x: np.abs(x[feature + '_expect'] - x[feature + '_pred']))
        data['ae'] = data.apply(ae, axis=1)
        data = (data.groupby(['implementation', 'run_num'])['ae']
                .agg(['mean'])
                .reset_index()
                .rename(columns={'mean': 'mae'}))
        return data

    def _mae_next_evaluation_batch(self, data, feature):
        data = data.copy()
        # if self.next_mode in ['next_action']:
        #     data[(feature + '_pred')] = np.mean(data[(feature + '_pred')].tolist(), axis=1)
        data = data[[(feature + '_expect'), (feature + '_pred'),
                     'caseid']]
        ae = (lambda x: np.abs(x[feature + '_expect'] - x[feature + '_pred']))
        data['ae'] = data.apply(ae, axis=1)
        data = (data.groupby(['caseid'])['ae']
                .agg(['mean'])
                .reset_index()
                .rename(columns={'mean': 'mae'}))
        return data

    def _similarity_evaluation(self, data, feature):
        data = data.copy()
        data = data[[(feature + '_expect'), (feature + '_pred'),
                     'run_num', 'implementation', 'pref_size']]
        # append all values and create alias
        values = (data[feature + '_pred'].tolist() +
                  data[feature + '_expect'].tolist())
        # values = list(set(itertools.chain.from_iterable(values)))
        values = np.unique(np.array(values)).tolist()
        index = self.create_task_alias(values)
        for col in ['_expect', '_pred']:
            list_to_string = lambda x: ''.join([index[y] for y in [x]])
            data['suff' + col] = (data[feature + col]
                                  .swifter.progress_bar(False)
                                  .apply(list_to_string))

        # measure similarity between pairs
        def distance(x, y):
            return (1 - (jf.damerau_levenshtein_distance(x, y) /
                         np.max([len(x), len(y)])))
        data['similarity'] = (data[['suff_expect', 'suff_pred']]
                              .swifter.progress_bar(False)
                              .apply(lambda x: distance(x.suff_expect,
                                                        x.suff_pred), axis=1))
        # agregate similarities
        data = (data.groupby(['implementation', 'run_num', 'pref_size'])['similarity']
                .agg(['mean'])
                .reset_index()
                .rename(columns={'mean': 'similarity'}))
        data = (pd.pivot_table(data,
                               values='similarity',
                               index=['run_num', 'implementation'],
                               columns=['pref_size'],
                               aggfunc=np.mean,
                               fill_value=0,
                               margins=True,
                               margins_name='mean')
                .reset_index())
        data = data[data.run_num != 'mean']
        # print("--Data--")
        # print(data)
        return data

    def _mae_remaining_evaluation(self, data, feature):
        # print("data @@@")
        # print(data)
        data = data.copy()
        data = data[[(feature + '_expect'), (feature + '_pred'),
                                'run_num', 'implementation', 'pref_size']]
        ae = (lambda x: np.abs(np.sum(x[feature + '_expect']) -
                               np.sum(x[feature + '_pred'])))
        data['ae'] = data.apply(ae, axis=1)
        data = (data.groupby(['implementation', 'run_num', 'pref_size'])['ae']
                .agg(['mean'])
                .reset_index()
                .rename(columns={'mean': 'mae'}))
        data = (pd.pivot_table(data,
                               values='mae',
                               index=['run_num', 'implementation'],
                               columns=['pref_size'],
                               aggfunc=np.mean,
                               fill_value=0,
                               margins=True,
                               margins_name='mean')
                .reset_index())
        data = data[data.run_num != 'mean']
        return data

# =============================================================================
# Timed string distance (Event Log Similarity - ELS)
# =============================================================================
    def _els_metric_evaluation(self, data, feature):
        # print(data)
        # print(data.columns)
        if 'dur' not in data.columns:
            #then it must have end timestamp and start timestamp
            data = self.add_calculated_times(data)
        data = self.scaling_data(data)
        log_data = data[data.implementation == 'log']
        alias = self.create_task_alias(data.task.unique())
        alpha_concurrency = ao.AlphaOracle(log_data, alias, True, True)
        # log reformating
        log_data = self.reformat_events(log_data.to_dict('records'),
                                        'task',
                                        alias)
        variants = data[['run_num', 'implementation']].drop_duplicates()
        variants = variants[variants.implementation!='log'].to_dict('records')
        similarity = list()
        for var in variants:
            pred_data = data[(data.implementation == var['implementation']) &
                             (data.run_num == var['run_num'])]
            pred_data = self.reformat_events(pred_data.to_dict('records'),
                                             'task',
                                             alias)
            mx_len = len(log_data)
            cost_matrix = [[0 for c in range(mx_len)] for r in range(mx_len)]
            # Create cost matrix
            # start = timer()
            for i in range(0, mx_len):
                for j in range(0, mx_len):
                    comp_sec = self.create_comparison_elements(pred_data,
                                                               log_data, i, j)
                    length = np.max([len(comp_sec['seqs']['s_1']),
                                     len(comp_sec['seqs']['s_2'])])
                    distance = self.tsd_alpha(comp_sec,
                                              alpha_concurrency.oracle)/length
                    cost_matrix[i][j] = distance
            # end = timer()
            # print(end - start)
            # Matching using the hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(np.array(cost_matrix))
            # Create response
            for idx, idy in zip(row_ind, col_ind):
                similarity.append(dict(caseid=pred_data[idx]['caseid'],
                                       sim_order=pred_data[idx]['profile'],
                                       log_order=log_data[idy]['profile'],
                                       sim_score=(1-(cost_matrix[idx][idy])),
                                       implementation=var['implementation'],
                                       run_num=var['run_num']))
        data = pd.DataFrame(similarity)
        data = (data.groupby(['implementation', 'run_num'])['sim_score']
                .agg(['mean'])
                .reset_index()
                .rename(columns={'mean': 'els'}))
        return data

    def _els_min_evaluation(self, data, feature):
        if 'dur' not in data.columns:
            #then it must have end timestamp and start timestamp
            data = self.add_calculated_times(data)
        data = self.scaling_data(data)
        log_data = data[data.implementation == 'log']
        alias = self.create_task_alias(data.task.unique())
        alpha_concurrency = ao.AlphaOracle(log_data, alias, True, True)
        # log reformating
        log_data = self.reformat_events(log_data.to_dict('records'),
                                        'task',
                                        alias)
        variants = data[['run_num', 'implementation']].drop_duplicates()
        variants = variants[variants.implementation!='log'].to_dict('records')
        similarity = list()
        for var in variants:
            pred_data = data[(data.implementation == var['implementation']) &
                             (data.run_num == var['run_num'])]
            pred_data = self.reformat_events(pred_data.to_dict('records'),
                                             'task',
                                             alias)
            temp_log_data = log_data.copy()
            for i in range(0, len(pred_data)):
                comp_sec = self.create_comparison_elements(pred_data,
                                                           temp_log_data, i, 0)
                min_dist = self.tsd_alpha(comp_sec, alpha_concurrency.oracle)
                min_idx = 0
                for j in range(1, len(temp_log_data)):
                    comp_sec = self.create_comparison_elements(pred_data,
                                                               temp_log_data, i, j)
                    sim = self.tsd_alpha(comp_sec, alpha_concurrency.oracle)
                    if min_dist > sim:
                        min_dist = sim
                        min_idx = j
                length = np.max([len(pred_data[i]['profile']),
                                 len(temp_log_data[min_idx]['profile'])])
                similarity.append(dict(caseid=pred_data[i]['caseid'],
                                       sim_order=pred_data[i]['profile'],
                                       log_order=temp_log_data[min_idx]['profile'],
                                       sim_score=(1-(min_dist/length)),
                                       implementation=var['implementation'],
                                       run_num = var['run_num']))
                del temp_log_data[min_idx]
        data = pd.DataFrame(similarity)
        data = (data.groupby(['implementation', 'run_num'])['sim_score']
                .agg(['mean'])
                .reset_index()
                .rename(columns={'mean': 'els'}))
        return data

    def create_comparison_elements(self, serie1, serie2, id1, id2):
        """
        Creates a dictionary of the elements to compare

        Parameters
        ----------
        serie1 : List
        serie2 : List
        id1 : integer
        id2 : integer

        Returns
        -------
        comp_sec : dictionary of comparison elements

        """
        comp_sec = dict()
        comp_sec['seqs'] = dict()
        comp_sec['seqs']['s_1'] = serie1[id1]['profile']
        comp_sec['seqs']['s_2'] = serie2[id2]['profile']
        comp_sec['times'] = dict()
        if self.one_timestamp:
            comp_sec['times']['p_1'] = serie1[id1]['dur_act_norm']
            comp_sec['times']['p_2'] = serie2[id2]['dur_act_norm']
        else:
            comp_sec['times']['p_1'] = serie1[id1]['dur_act_norm']
            comp_sec['times']['p_2'] = serie2[id2]['dur_act_norm']
            comp_sec['times']['w_1'] = serie1[id1]['wait_act_norm']
            comp_sec['times']['w_2'] = serie2[id2]['wait_act_norm']
        return comp_sec

    def tsd_alpha(self, comp_sec, alpha_concurrency):
        """
        Compute the Damerau-Levenshtein distance between two given
        strings (s_1 and s_2)
        Parameters
        ----------
        comp_sec : dict
        alpha_concurrency : dict
        Returns
        -------
        Float
        """
        s_1 = comp_sec['seqs']['s_1']
        s_2 = comp_sec['seqs']['s_2']
        dist = {}
        lenstr1 = len(s_1)
        lenstr2 = len(s_2)
        for i in range(-1, lenstr1+1):
            dist[(i, -1)] = i+1
        for j in range(-1, lenstr2+1):
            dist[(-1, j)] = j+1
        for i in range(0, lenstr1):
            for j in range(0, lenstr2):
                if s_1[i] == s_2[j]:
                    cost = self.calculate_cost(comp_sec['times'], i, j)
                else:
                    cost = 1
                dist[(i, j)] = min(
                    dist[(i-1, j)] + 1, # deletion
                    dist[(i, j-1)] + 1, # insertion
                    dist[(i-1, j-1)] + cost # substitution
                    )
                if i and j and s_1[i] == s_2[j-1] and s_1[i-1] == s_2[j]:
                    if alpha_concurrency[(s_1[i], s_2[j])] == Rel.PARALLEL:
                        cost = self.calculate_cost(comp_sec['times'], i, j-1)
                    dist[(i, j)] = min(dist[(i, j)], dist[i-2, j-2] + cost)  # transposition
        return dist[lenstr1-1, lenstr2-1]

    def calculate_cost(self, times, s1_idx, s2_idx):
        """
        Takes two events and calculates the penalization based on mae distance

        Parameters
        ----------
        times : dict with lists of times
        s1_idx : integer
        s2_idx : integer

        Returns
        -------
        cost : float
        """
        if self.one_timestamp:
            p_1 = times['p_1']
            p_2 = times['p_2']
            cost = np.abs(p_2[s2_idx]-p_1[s1_idx]) if p_1[s1_idx] > 0 else 0
        else:
            p_1 = times['p_1']
            p_2 = times['p_2']
            w_1 = times['w_1']
            w_2 = times['w_2']
            t_1 = p_1[s1_idx] + w_1[s1_idx]
            if t_1 > 0:
                b_1 = (p_1[s1_idx]/t_1)
                cost = ((b_1*np.abs(p_2[s2_idx]-p_1[s1_idx])) +
                        ((1 - b_1)*np.abs(w_2[s2_idx]-w_1[s1_idx])))
            else:
                cost = 0
        return cost

# =============================================================================
# dl distance
# =============================================================================
    def _dl_distance_evaluation(self, data, feature):
        """
        similarity score
        Demerau-Levinstain distance measurement
        Parameters
        ----------
        log_data : list of events
        simulation_data : list simulation event log

        Returns
        -------
        similarity : float

        """
        if 'dur' not in data.columns:
            #then it must have end timestamp and start timestamp
            data = self.add_calculated_times(data)
        data = self.scaling_data(data)
        log_data = data[data.implementation == 'log']
        alias = self.create_task_alias(data.task.unique())
        # alpha_concurrency = ao.AlphaOracle(log_data, alias, True, True)
        # log reformating
        log_data = self.reformat_events(log_data.to_dict('records'),
                                        'task',
                                        alias)
        variants = data[['run_num', 'implementation']].drop_duplicates()
        variants = variants[variants.implementation != 'log'].to_dict('records')
        similarity = list()
        for var in variants:
            pred_data = data[(data.implementation == var['implementation']) &
                             (data.run_num == var['run_num'])]
            pred_data = self.reformat_events(pred_data.to_dict('records'),
                                             'task',
                                             alias)
            mx_len = len(log_data)
            dl_matrix = [[0 for c in range(mx_len)] for r in range(mx_len)]
            # Create cost matrix
            # start = timer()
            for i in range(0, mx_len):
                for j in range(0, mx_len):
                    d_l = self.calculate_distances(pred_data, log_data, i, j)
                    dl_matrix[i][j] = d_l
            # end = timer()
            # print(end - start)
            dl_matrix = np.array(dl_matrix)
            # Matching using the hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(np.array(dl_matrix))
            # Create response
            for idx, idy in zip(row_ind, col_ind):
                similarity.append(dict(caseid=pred_data[idx]['caseid'],
                                       sim_order=pred_data[idx]['profile'],
                                       log_order=log_data[idy]['profile'],
                                       sim_score=(1-(dl_matrix[idx][idy])),
                                       implementation=var['implementation'],
                                       run_num=var['run_num']))
        data = pd.DataFrame(similarity)
        data = (data.groupby(['implementation', 'run_num'])['sim_score']
                .agg(['mean'])
                .reset_index()
                .rename(columns={'mean': 'dl'}))
        return data

    @staticmethod
    def calculate_distances(serie1, serie2, id1, id2):
        """
        Parameters
        ----------
        serie1 : list
        serie2 : list
        id1 : index of the list 1
        id2 : index of the list 2

        Returns
        -------
        dl : float value
        ae : absolute error value
        """
        length = np.max([len(serie1[id1]['profile']),
                         len(serie2[id2]['profile'])])
        d_l = jf.damerau_levenshtein_distance(
            ''.join(serie1[id1]['profile']),
            ''.join(serie2[id2]['profile']))/length
        return d_l

# =============================================================================
# mae distance : Earth Mover’s Distance (EMD)
# =============================================================================

    def _mae_metric_evaluation(self, data, feature):
        """
        mae distance between logs

        Parameters
        ----------
        log_data : list of events
        simulation_data : list simulation event log

        Returns
        -------
        similarity : float

        """
        # print(data)
        # print(data.columns)
        if 'dur' not in data.columns:
            #then it must have end timestamp and start timestamp
            data = self.add_calculated_times(data)
        data = self.scaling_data(data)
        log_data = data[data.implementation == 'log']
        alias = self.create_task_alias(data.task.unique())
        # alpha_concurrency = ao.AlphaOracle(log_data, alias, True, True)
        # log reformating
        log_data = self.reformat_events(log_data.to_dict('records'),
                                        'task',
                                        alias)
        variants = data[['run_num', 'implementation']].drop_duplicates()
        variants = variants[variants.implementation != 'log'].to_dict('records')
        similarity = list()
        for var in variants:
            pred_data = data[(data.implementation == var['implementation']) &
                             (data.run_num == var['run_num'])]
            pred_data = self.reformat_events(pred_data.to_dict('records'),
                                             'task',
                                             alias)
            mx_len = len(log_data)
            ae_matrix = [[0 for c in range(mx_len)] for r in range(mx_len)]
            # Create cost matrix
            # start = timer()
            for i in range(0, mx_len):
                for j in range(0, mx_len):
                    # cicle_time_s1 = (pred_data[i]['end_time'] -
                    #                  pred_data[i]['start_time']).total_seconds()
                    # cicle_time_s2 = (log_data[j]['end_time'] -
                    #                  log_data[j]['start_time']).total_seconds()
                    print("CaseId and Enent No of prediction : ", pred_data[i]['caseid'], pred_data[i]['event_nr'])
                    print("CaseId and Enent No of prediction : ", log_data[j]['caseid'], log_data[j]['event_nr'])
                    ae_aggr = list()
                    _length = min(len(pred_data[i]['dur']), len(log_data[j]['dur']))
                    for k in range(0, _length):
                        print("length : ", len(pred_data[i]['dur']), len(log_data[j]['dur']), k)
                        cicle_time_s1 = (pred_data[i]['dur'][k])
                        cicle_time_s2 = (log_data[j]['dur'][k])
                        print("cicle_time_s1 : ", cicle_time_s1)
                        print("log duration : ", cicle_time_s2)
                        print("Difference ", cicle_time_s1 - cicle_time_s2)
                        ae_aggr.append(cicle_time_s1 - cicle_time_s2)
                    print("List of difference : ", ae_aggr)
                    ae = np.abs(ae_aggr)
                    ae_matrix[i][j] = ae
                    print("Matrix Input")
                    print(ae_matrix[i][j])
                    print(type(print(ae_matrix[i][j])))
            # end = timer()
            # print(end - start)
            print("Matrix Before")
            print(ae_matrix)
            print(type(ae_matrix))
            ae_matrix = np.array(ae_matrix)
            print("Matrix After")
            print(ae_matrix)
            print(type(ae_matrix))
            # Matching using the hungarian algorithm
            # ae_matrix = ae_matrix.tolist()
            row_ind, col_ind = linear_sum_assignment(np.array(ae_matrix))
            # Create response
            for idx, idy in zip(row_ind, col_ind):
                similarity.append(dict(caseid=pred_data[idx]['caseid'],
                                       sim_order=pred_data[idx]['profile'],
                                       log_order=log_data[idy]['profile'],
                                       sim_score=(ae_matrix[idx][idy]),
                                       implementation=var['implementation'],
                                       run_num=var['run_num']))
        data = pd.DataFrame(similarity)
        data = (data.groupby(['implementation', 'run_num'])['sim_score']
                .agg(['mean'])
                .reset_index()
                .rename(columns={'mean': 'mae_log'}))
        return data
# =============================================================================
# Support methods
# =============================================================================
    @staticmethod
    def calculate_splits(df, max_cases=1000):
        print(len(df.caseid.unique()))
        # calculate the number of bytes a row occupies
        n_splits = int(np.ceil(len(df.caseid.unique()) / max_cases))
        return n_splits

    @staticmethod
    def folding_creation(df, splits, output):
        idxs = [x for x in range(0, len(df), round(len(df)/splits))]
        idxs.append(len(df))
        folds = [pd.DataFrame(df.iloc[idxs[i-1]:idxs[i]])
                 for i in range(1, len(idxs))]
        # Export folds
        file_names = list()
        for i, fold in enumerate(folds):
            file_name = os.path.join(output,'split_'+str(i+1)+'.csv')
            fold.to_csv(file_name, index=False)
            file_names.append(file_name)
        return file_names

    @staticmethod
    def define_ranges(data, num_folds):
        num_events = int(np.round(len(data)/num_folds))
        folds = list()
        for i in range(0, num_folds):
            sidx = i * num_events
            eidx = (i + 1) * num_events
            if i == 0:
                folds.append({'min': 0, 'max': eidx})
            elif i == (num_folds - 1):
                folds.append({'min': sidx, 'max': len(data)})
            else:
                folds.append({'min': sidx, 'max': eidx})
        return folds

    @staticmethod
    def create_task_alias(categories):
        """
        Create string alias for tasks names or tuples of tasks-roles names

        Parameters
        ----------
        features : list

        Returns
        -------
        alias : alias dictionary

        """
        variables = sorted(categories)
        characters = [chr(i) for i in range(0, len(variables))]
        aliases = random.sample(characters, len(variables))
        alias = dict()
        for i, _ in enumerate(variables):
            alias[variables[i]] = aliases[i]
        return alias

    def add_calculated_times(self, log):
        """Appends the indexes and relative time to the dataframe.
        parms:
            log: dataframe.
        Returns:
            Dataframe: The dataframe with the calculated features added.
        """
        log['dur'] = 0
        log = log.to_dict('records')
        log = sorted(log, key=lambda x: x['caseid'])
        for _, group in itertools.groupby(log, key=lambda x: x['caseid']):
            events = list(group)
            ordk = 'end_timestamp' if self.one_timestamp else 'start_timestamp'
            events = sorted(events, key=itemgetter(ordk))
            for i in range(0, len(events)):
                # In one-timestamp approach the first activity of the trace
                # is taken as instant since there is no previous timestamp
                if self.one_timestamp:
                    if i == 0:
                        dur = 0
                    else:
                        dur = (events[i]['end_timestamp'] -
                               events[i-1]['end_timestamp']).total_seconds()
                else:
                    dur = (events[i]['end_timestamp'] -
                           events[i]['start_timestamp']).total_seconds()
                    if i == 0:
                        wit = 0
                    else:
                        wit = (events[i]['start_timestamp'] -
                               events[i-1]['end_timestamp']).total_seconds()
                    events[i]['wait'] = wit
                events[i]['dur'] = dur
        return pd.DataFrame.from_dict(log)

    def scaling_data(self, data):
        """
        Scales times values activity based

        Parameters
        ----------
        data : dataframe

        Returns
        -------
        data : dataframe with normalized times

        """
        df_modif = data.copy()
        np.seterr(divide='ignore')
        summ = data.groupby(['task'])['dur'].max().to_dict()
        dur_act_norm = (lambda x: x['dur']/summ[x['task']]
                        if summ[x['task']] > 0 else 0)
        df_modif['dur_act_norm'] = df_modif.apply(dur_act_norm, axis=1)
        if not self.one_timestamp:
            summ = data.groupby(['task'])['wait'].max().to_dict()
            wait_act_norm = (lambda x: x['wait']/summ[x['task']]
                            if summ[x['task']] > 0 else 0)
            df_modif['wait_act_norm'] = df_modif.apply(wait_act_norm, axis=1)
        return df_modif

    def reformat_events(self, data, features, alias):
        """Creates series of activities, roles and relative times per trace.
        parms:
            log_df: dataframe.
            ac_table (dict): index of activities.
            rl_table (dict): index of roles.
        Returns:
            list: lists of activities, roles and relative times.
        """
        # Update alias
        if isinstance(features, list):
            [x.update(dict(alias=alias[(x[features[0]],
                                             x[features[1]])])) for x in data]
        else:
            [x.update(dict(alias=alias[x[features]])) for x in data]
        temp_data = list()
        # define ordering keys and columns
        if self.one_timestamp:
            columns = ['alias', 'dur', 'dur_act_norm']
            sort_key = 'event_nr'
        else:
            sort_key = 'event_nr'
            columns = ['alias', 'dur',
                       'dur_act_norm', 'wait', 'wait_act_norm']
        data = sorted(data, key=lambda x: (x['caseid'], x[sort_key]))
        for key, group in itertools.groupby(data, key=lambda x: x['caseid']):
            trace = list(group)
            temp_dict = dict()
            for col in columns:
                serie = [y[col] for y in trace]
                if col == 'alias':
                    temp_dict = {**{'profile': serie}, **temp_dict}
                else:
                    serie = [y[col] for y in trace]
                temp_dict = {**{col: serie}, **temp_dict}
            temp_dict = {**{'caseid': key, 'event_nr': trace[0][sort_key],
                            'event_nr': trace[-1][sort_key]},
                         **temp_dict}
            temp_data.append(temp_dict)
        return sorted(temp_data, key=itemgetter('event_nr'))


    @staticmethod
    def create_file_list(path, prefix):
        file_list = list()
        for root, dirs, files in os.walk(path):
            for f in files:
                if prefix in f:
                    file_list.append(f)
        return file_list
