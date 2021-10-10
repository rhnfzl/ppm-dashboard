# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 19:18:18 2020

@author: Manuel Camargo
"""
import pandas as pd
import numpy as np

import itertools
from operator import itemgetter

from support_modules import role_discovery as rl
class FeaturesMannager():


    def __init__(self, params):
        """constructor"""
        self.rp_sim = params['rp_sim']
        self.model_type = params['model_type']
        self.one_timestamp = params['one_timestamp']
        self.resources = pd.DataFrame
        self.norm_method = params['norm_method']
        self.filename = params['file_name']
        self.activity = params['activity']
        self._scalers = dict()
        self.scale_dispatcher = {'basic': self._scale_base,
                                 'inter': self._scale_inter}
        # self.scale_dispatcher = {'basic': self._scale_base} #since only inter case feature models are being trained the configuration file is renamed in such a way

    def calculate(self, log, add_cols, type_call):
        if type_call == 'train':
            log = self.add_resources(log)
        elif type_call == 'predict' and 'role' not in log.columns: #if the role hasn't been calculated before hand
            log = self.add_resources(log)
        log = self.add_calculated_times(log)
        # print("Log Properties : ", log.dtypes, "Additional Cols :", add_cols)
        #log = self.filter_features(log, add_cols) #----Filters out the features, Preprocessing of the vlaues helps to select the required features beforehand
        return self.scale_features(log, add_cols)

    def add_resources(self, log):
        # Resource pool discovery
        res_analyzer = rl.ResourcePoolAnalyser(log, sim_threshold=self.rp_sim)
        # Role discovery
        self.resources = pd.DataFrame.from_records(res_analyzer.resource_table)
        self.resources = self.resources.rename(index=str,
                                               columns={"resource": "user"})
        # Add roles information
        log = log.merge(self.resources, on='user', how='left')
        log = log[~log.task.isin(['Start', 'End'])]
        log = log.reset_index(drop=True)
        return log

    def filter_features(self, log, add_cols):
        # print("Log Properties : ", log.dtypes, log.columns)
        # Add intercase features
        #columns = ['caseid', 'task', 'user', 'end_timestamp', 'role', 'dur', 'label'] #filtering features which will passed to train and test
        columns = ['caseid', 'task', 'user', 'end_timestamp', 'role', 'dur']  # filtering features which will passed to train and test
        if not self.one_timestamp:
            columns.extend(['start_timestamp', 'wait'])
        columns.extend(add_cols)
        log = log[columns]
        return log

    def add_calculated_times(self, log):
        """Appends the indexes and relative time to the dataframe.
        parms:
            log: dataframe.
        Returns:
            Dataframe: The dataframe with the calculated features added.
        """
        log['dur'] = 0
        log['acc_cycle'] = 0
        log['daytime'] = 0
        log = log.to_dict('records')
        log = sorted(log, key=lambda x: x['caseid'])
        for _, group in itertools.groupby(log, key=lambda x: x['caseid']):
            events = list(group)
            ordk = 'end_timestamp' if self.one_timestamp else 'start_timestamp'
            events = sorted(events, key=itemgetter(ordk))
            for i in range(0, len(events)):
                # In one-timestamp approach the first activity of the trace
                # is taken as instantsince there is no previous timestamp
                # to find a range
                if self.one_timestamp:
                    if i == 0:
                        dur = 0
                        acc = 0
                    else:
                        dur = (events[i]['end_timestamp'] -
                               events[i-1]['end_timestamp']).total_seconds()
                        acc = (events[i]['end_timestamp'] -
                               events[0]['end_timestamp']).total_seconds()
                else:
                    dur = (events[i]['end_timestamp'] -
                           events[i]['start_timestamp']).total_seconds()
                    acc = (events[i]['end_timestamp'] -
                           events[0]['start_timestamp']).total_seconds()
                    if i == 0:
                        wit = 0
                    else:
                        wit = (events[i]['start_timestamp'] -
                               events[i-1]['end_timestamp']).total_seconds()
                    events[i]['wait'] = wit if wit >= 0 else 0
                events[i]['dur'] = dur
                events[i]['acc_cycle'] = acc
                time = events[i][ordk].time()
                time = time.second + time.minute*60 + time.hour*3600
                events[i]['daytime'] = time
                events[i]['weekday'] = events[i]['end_timestamp'].weekday()
        return pd.DataFrame.from_dict(log)

    def scale_features(self, log, add_cols):
        scaler = self._get_scaler(self.model_type)
        return scaler(log, add_cols)

    def register_scaler(self, model_type, scaler):
        try:
            self._scalers[model_type] = self.scale_dispatcher[scaler]
        except KeyError:
            raise ValueError(scaler)

    def _get_scaler(self, model_type):
        scaler = self._scalers.get(model_type)
        if not scaler:
            raise ValueError(model_type)
        return scaler

    def _scale_base(self, log, add_cols):
        if self.one_timestamp:
            log, scale_args = self.scale_feature(log, 'dur', self.norm_method)
        else:
            log, dur_scale = self.scale_feature(log, 'dur', self.norm_method)
            log, wait_scale = self.scale_feature(log, 'wait', self.norm_method)
            scale_args = {'dur': dur_scale, 'wait': wait_scale}
        return log, scale_args

    def _scale_inter(self, log, add_cols):
        # add_cat = {}
        if self.activity != 'training':
            norm_cols = [col for col in log.columns if '_norm' in col]
            scale_args = dict()
        else:
            norm_cols = list()

        if self.one_timestamp:
            if not [wcol for wcol in norm_cols if 'dur' in wcol]:
                log, scale_args = self.scale_feature(log, 'dur', self.norm_method)
        else:
            if not [wcol for wcol in norm_cols if 'dur' in wcol]:
                log, dur_scale = self.scale_feature(log, 'dur', self.norm_method)
            if not [wcol for wcol in norm_cols if 'wait' in wcol]:
                log, wait_scale = self.scale_feature(log, 'wait', self.norm_method)
            scale_args = {'dur': dur_scale, 'wait': wait_scale}
        for col in add_cols:
            if not [wcol for wcol in norm_cols if col in wcol]:
                if col == 'daytime':
                        log, _ = self.scale_feature(log, 'daytime', 'day_secs', True)
                elif col == 'open_cases':
                        log, _ = self.scale_feature(log, 'open_cases', 'max')
                elif col == 'weekday':
                        log, _ = self.scale_feature(log, 'weekday', None)
                elif 'sepsis' in self.filename: #Log specific Logic
                    #-- Variables which change during the case use lognorm
                    #-- Variables which don't change during the case use max
                    if col == 'Diagnose_ord': #because of large number of catagorical variable
                            log = self.ordinal_encoder(log, 'Diagnose')
                            log, _ = self.scale_feature(log, 'Diagnose_ord', 'max')
                    elif col == 'CRP':
                            log, _ = self.scale_feature(log, 'CRP', self.norm_method)
                    elif col == 'LacticAcid':
                            log, _ = self.scale_feature(log, 'LacticAcid', self.norm_method)
                    elif col == 'Leucocytes':
                            log, _ = self.scale_feature(log, 'Leucocytes', self.norm_method)
                    elif col == 'Diagnose':
                            log, _ = self.scale_feature(log, 'Diagnose', self.norm_method)
                    elif col == 'Age':
                            log, _ = self.scale_feature(log, 'Age', 'max')
                else:
                        log, _ = self.scale_feature(log, col, self.norm_method, True)
        return log, scale_args

    # =========================================================================
    # Scale features to create dur_norm
    # =========================================================================

    @staticmethod
    def scale_feature(log, feature, method, replace=False):
        """Scales a number given a technique.
        Args:
            log: Event-log to be scaled.
            feature: Feature to be scaled.
            method: Scaling method max, lognorm, normal, per activity.
            replace (optional): replace the original value or keep both.
        Returns:
            Scaleded value between 0 and 1.
        """
        scale_args = dict()
        _temp_scale_args = dict()
        if method == 'lognorm':
            log[feature + '_log'] = np.log1p(log[feature])
            max_value = np.max(log[feature+'_log'])
            min_value = np.min(log[feature+'_log'])
            log[feature+'_norm'] = np.divide(
                    np.subtract(log[feature+'_log'], min_value), (max_value - min_value))
            log = log.drop((feature + '_log'), axis=1)
            scale_args = {'max_value': max_value, 'min_value': min_value}
        elif method == 'normal':
            max_value = np.max(log[feature])
            min_value = np.min(log[feature])
            log[feature+'_norm'] = np.divide(
                    np.subtract(log[feature], min_value), (max_value - min_value))
            scale_args = {'max_value': max_value, 'min_value': min_value}
        elif method == 'standard':
            mean = np.mean(log[feature])
            std = np.std(log[feature])
            log[feature + '_norm'] = np.divide(np.subtract(log[feature], mean),
                                               std)
            scale_args = {'mean': mean, 'std': std}
        elif method == 'max':
            max_value = np.max(log[feature])
            log[feature + '_norm'] = (np.divide(log[feature], max_value)
                                      if max_value > 0 else 0)
            scale_args = {'max_value': max_value}
        elif method == 'day_secs':
            max_value = 86400
            log[feature + '_norm'] = (np.divide(log[feature], max_value)
                                      if max_value > 0 else 0)
            scale_args = {'max_value': max_value}
        elif method is None:
            log[feature+'_norm'] = log[feature]
        else:
            raise ValueError(method)
        if replace:
            log = log.drop(feature, axis=1)
        return log, scale_args

    @staticmethod
    def ordinal_encoder(log, feature, replace=False):
        for i in range(len([feature])):

            cat_label = feature

            temp_list = log[[cat_label]].values.tolist()

            subsec_set = {(x[0]) for x in temp_list}

            subsec_set = sorted(list(subsec_set))

            _index = dict()

            for ix, _ in enumerate(subsec_set):
                _index[subsec_set[ix]] = ix + 1

            _idx = lambda x: _index[x[cat_label]]

            log[cat_label + '_ord'] = log.apply(_idx, axis=1)

            if replace:
                log = log.drop(feature, axis=1)
        return log

    # @staticmethod
    # def embedding_encoder(log, feature, replace=False):
    #
    #     return log