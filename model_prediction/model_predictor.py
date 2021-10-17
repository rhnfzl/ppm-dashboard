# -*- coding: utf-8 -*-
import os
import json

import streamlit as st
import pandas as pd
import numpy as np
import configparser as cp
import matplotlib.pyplot as plt

from num2words import num2words as nw
from st_aggrid import AgGrid
from tensorflow.keras.models import load_model


from support_modules.readers import log_reader as lr
from support_modules import support as sup
from support_modules import features_manager as feat
from model_prediction import interfaces as it
from model_prediction.analyzers import sim_evaluator as ev

pd.set_option('mode.chained_assignment', None) #supressing the warning of chained indexing

class ModelPredictor():
    """
    This is the main class encharged of the model evaluation
    """

    def __init__(self, parms):
        self.output_route = os.path.join('output_files', parms['folder'])
        self.parms = parms
        # load parameters
        self.load_parameters()
        self.model_name, _ = os.path.splitext(parms['model_file'])
        self.model = load_model(os.path.join(self.output_route,
                                             parms['model_file']))

        self.log = self.load_log_test(self.output_route, self.parms)

        self.samples = dict()
        self.predictions = None
        self.confirmation_results = None
        self.sim_values = list()
        self.run_num = 0

        self.model_def = dict()
        self.read_model_definition(self.parms['model_type'])
        # print("Model Type :", self.model_def)
        self.parms['additional_columns'] = self.model_def['additional_columns']
        # self.acc = self.execute_predictive_task()
        self.execute_predictive_task()
        # self._export_results(self.output_route)

    def execute_predictive_task(self):

        # -- Adding the code block for going to feature manager to facilitate anyother test data which is not formated beforehand
        feat_mannager = feat.FeaturesMannager(self.parms)
        feat_mannager.register_scaler(self.parms['model_type'],
                                      self.model_def['vectorizer'])
        self.log, _ = feat_mannager.calculate(
            self.log, self.parms['additional_columns'], 'predict')
        # ---
        sampler = it.SamplesCreator()
        sampler.create(self, self.parms['activity'])

        # create examples for next event and suffix
        if self.parms['mode'] == 'batch':
            self.parms['num_cases'] = len(self.log.caseid.unique())
            #--logic for timestamp
            _min_tm_df = self.log.groupby('caseid')['end_timestamp'].min().to_frame()
            _min_tm_df.reset_index(level=0, inplace=True)
            _min_tm_dict = _min_tm_df.to_dict('records')
            # self.parms['min_time'] = self.log.end_timestamp.min()
            self.parms['min_time'] = _min_tm_dict
            if self.parms['batch_mode'] == 'pre_prefix':
                self.parms['caseid'] = np.array(self.log.drop(self.log.sort_values(['caseid']).groupby('caseid').head(self.parms['batchprefixnum']).index).caseid)
            else:
                self.parms['caseid'] = np.array(self.log.caseid)  # adding caseid to the parms for batch mode
        # predict
        self.imp = self.parms['variant']  # passes value arg_max and random_choice
        self.run_num = 0
        #prediction call
        for i in range(0, self.parms['rep']):
            self.predict_values()
            self.run_num += 1
        # export predictions
        # if self.parms['mode'] == 'batch':
            # self.export_predictions()
        # assesment
        evaluator = EvaluateTask()
        print("Debug Point I")
        print(self.predictions.iloc[:5])
        #--predicted negative time to positive
                #--predicted negative time to positive
        if self.predictions['tm_pred'].dtypes == 'O':
            for i in range(len(self.predictions['tm_pred'])):
                _xc = list()
                for j in range(len(self.predictions['tm_pred'][i])):
                    _xc.append(abs(self.predictions['tm_pred'][i][j]))
                self.predictions['tm_pred'][i] = _xc
        else:
            self.predictions['tm_pred'] = self.predictions['tm_pred'].abs()

        results_copy = self.predictions.copy()

        self.dashboard_prediction(results_copy, self.parms, self.confirmation_results)

        # if self.parms['mode'] == 'next':
        #     evaluator.evaluate(self.parms, self.predictions)
        # elif self.parms['mode'] == 'batch':
        evaluator.evaluate(self.parms, self.predictions)

    def predict_values(self):
        # Predict values
        executioner = it.PredictionTasksExecutioner()
        executioner.predict(self, self.parms['activity'], self.parms['mode'], self.parms['next_mode'], self.parms['batch_mode'])

    @staticmethod
    def load_log_test(output_route, parms):
        df_test = lr.LogReader(
            os.path.join(output_route, 'parameters', 'test_log.csv'),
            parms['read_options'])
        if parms['mode'] == 'next':
            df_test = pd.DataFrame(df_test.data)
            df_test = df_test[~df_test.task.isin(['Start', 'End']) & df_test.caseid.isin([parms['nextcaseid']])]
        elif parms['mode'] == 'batch':
            df_test = pd.DataFrame(df_test.data)
            df_test = df_test[~df_test.task.isin(['Start', 'End'])]
            df_test = df_test.groupby("caseid").filter(
                lambda x: len(x) >= min(parms['batchlogrange']) and len(x) <= max(parms['batchlogrange']))

            if df_test.empty is True:
                st.error("The Range of Event Number doesn't have any Case Id's")
                raise ValueError(output_route)
            _count = df_test['caseid'].value_counts().to_frame()
            _count.reset_index(level=0, inplace=True)
            _count.sort_values(by=['caseid'], inplace=True)
            # print("Count of respective Case Id")
            # print(_count)
        return df_test

    def load_parameters(self):
        # Loading of parameters from training
        path = os.path.join(self.output_route,
                            'parameters',
                            'model_parameters.json')
        with open(path) as file:
            data = json.load(file)
            if 'activity' in data:
                del data['activity']
            self.parms = {**self.parms, **{k: v for k, v in data.items()}}
            self.parms['dim'] = {k: int(v) for k, v in data['dim'].items()}
            if self.parms['one_timestamp']:
                self.parms['scale_args'] = {
                    k: float(v) for k, v in data['scale_args'].items()}
            else:
                for key in data['scale_args'].keys():
                    self.parms['scale_args'][key] = {
                        k: float(v) for k, v in data['scale_args'][key].items()}
            self.parms['index_ac'] = {int(k): v
                                      for k, v in data['index_ac'].items()}
            self.parms['index_rl'] = {int(k): v
                                      for k, v in data['index_rl'].items()}

            file.close()
            self.ac_index = {v: k for k, v in self.parms['index_ac'].items()}
            self.rl_index = {v: k for k, v in self.parms['index_rl'].items()}

    def sampling(self, sampler):
        # print("Model Type : ", self.parms['model_type'])
        # print("Model Def : ", self.model_def['vectorizer'])
        sampler.register_sampler(self.parms['model_type'],
                                 self.model_def['vectorizer'])
        self.samples = sampler.create_samples(
            self.parms, self.log, self.ac_index, self.rl_index, self.model_def['additional_columns'])
    #
    def predict(self, executioner, mode):
        # if mode == 'next':
        #     results = executioner.predict(self.parms,
        #                                   self.model,
        #                                   self.samples,
        #                                   self.imp,
        #                                   self.model_def['vectorizer'])
        # elif mode == 'batch':
        if self.parms['next_mode'] == 'next_action':
                results, conf_results = executioner.predict(self.parms,
                                              self.model,
                                              self.samples,
                                              self.imp,
                                              self.model_def['vectorizer'])
                conf_results = pd.DataFrame(conf_results)

                if self.confirmation_results is None:
                    self.confirmation_results = conf_results
                else:
                    self.confirmation_results = self.confirmation_results.append(results,
                                                               ignore_index=True)

        else:
                results = executioner.predict(self.parms,
                                                self.model,
                                                self.samples,
                                                self.imp,
                                                self.model_def['vectorizer'])

        results = pd.DataFrame(results)
        results['run_num'] = self.run_num
        results['implementation'] = self.imp
        if self.predictions is None:
            self.predictions = results
        else:
            self.predictions = self.predictions.append(results,
                                                       ignore_index=True)

    # def export_predictions(self):
    #     output_folder = os.path.join(self.output_route, 'results')
    #     if not os.path.exists(output_folder):
    #         os.makedirs(output_folder)
    #     filename = self.model_name + '_' + self.parms['activity'] + '.csv'
    #     self.predictions.to_csv(os.path.join(output_folder, filename),
    #                             index=False)

    # @staticmethod
    # # ------This Can be removed as it is not getting used anywhere------
    # def scale_feature(log, feature, parms, replace=False):
    #     """Scales a number given a technique.
    #     Args:
    #         log: Event-log to be scaled.
    #         feature: Feature to be scaled.
    #         method: Scaling method max, lognorm, normal, per activity.
    #         replace (optional): replace the original value or keep both.
    #     Returns:
    #         Scaleded value between 0 and 1.
    #     """
    #     method = parms['norm_method']
    #     scale_args = parms['scale_args']
    #     if method == 'lognorm':
    #         log[feature + '_log'] = np.log1p(log[feature])
    #         max_value = scale_args['max_value']
    #         min_value = scale_args['min_value']
    #         log[feature + '_norm'] = np.divide(
    #             np.subtract(log[feature + '_log'], min_value), (max_value - min_value))
    #         log = log.drop((feature + '_log'), axis=1)
    #     elif method == 'normal':
    #         max_value = scale_args['max_value']
    #         min_value = scale_args['min_value']
    #         log[feature + '_norm'] = np.divide(
    #             np.subtract(log[feature], min_value), (max_value - min_value))
    #     elif method == 'standard':
    #         mean = scale_args['mean']
    #         std = scale_args['std']
    #         log[feature + '_norm'] = np.divide(np.subtract(log[feature], mean),
    #                                            std)
    #     elif method == 'max':
    #         max_value = scale_args['max_value']
    #         log[feature + '_norm'] = (np.divide(log[feature], max_value)
    #                                   if max_value > 0 else 0)
    #     elif method is None:
    #         log[feature + '_norm'] = log[feature]
    #     else:
    #         raise ValueError(method)
    #     if replace:
    #         log = log.drop(feature, axis=1)
    #     return log

    def read_model_definition(self, model_type):
        Config = cp.ConfigParser(interpolation=None)
        Config.read('models_spec.ini')
        # File name with extension
        self.model_def['additional_columns'] = sup.reduce_list(
            Config.get(model_type, 'additional_columns'), dtype='str')
        self.model_def['vectorizer'] = Config.get(model_type, 'vectorizer')



    #-------------------------------------------------------------------------------------------------------------------
    #                   Dashboard Code
    # -------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def dashboard_prediction(pred_results_df, parms, confirmation_results):

        print("Debug Point II")
        print("Dashboard Initial Dataframe")
        print(pred_results_df.iloc[:5])
        print("Session State")
        print(st.session_state)
        # state_of_theprocess = st.empty()
        # if parms['next_mode'] == 'next_action':
        # # -----------------------------------------------------------------------------------------------------------------------------------
        # # Removing 'ac_prefix', 'rl_prefix', 'tm_prefix', 'run_num', 'implementation' from the result
        # #creating the prefix for execution mode only, first the last row has been selected to accomodate all the input for
        # # activity, role, time. for time it has been converted from numpy array to array of array and then to flat array
        # # all the three attributes has been put in dictionary and then convrted to dataframe
        # print("Activity : ", pred_results_df['ac_prefix'].iloc[-1:].values.tolist()[0])
        # print("Role : ", pred_results_df['rl_prefix'].iloc[-1:].values.tolist()[0])
        # print("Time : ", sum([x.tolist() for x in pred_results_df['tm_prefix'].iloc[-1:].values.tolist()[0]], []))
        #
        # result_dash_hist_execution = pd.DataFrame({"Activity" : pred_results_df['ac_prefix'].iloc[-1:].values.tolist()[0],
        #                               "Role" : pred_results_df['rl_prefix'].iloc[-1:].values.tolist()[0],
        #                               "Time" : sum([x.tolist() for x in pred_results_df['tm_prefix'].iloc[-1:].values.tolist()[0]], [])})
        #
        # # Replacing from Dictionary Values to it's original name
        # result_dash_hist_execution['Activity'] = result_dash_hist_execution.Activity.replace(parms['index_ac'])
        # result_dash_hist_execution['Role'] = result_dash_hist_execution.Role.replace(parms['index_rl'])
        # # removing the fist row
        # result_dash_hist_execution = result_dash_hist_execution.iloc[1:]
        # #-----------------------------------------------------------------------------------------------------------------------------------
        # results_dash = pred_results_df[['ac_prefix', 'rl_prefix', 'tm_prefix', 'run_num', 'implementation']].copy()
        # results_dash = pred_results_df.drop(['ac_prefix', 'rl_prefix', 'tm_prefix', 'run_num', 'implementation'], axis=1)
        #
        # # Replacing from Dictionary Values to it's original name
        # results_dash['ac_expect'] = results_dash.ac_expect.replace(parms['index_ac'])
        # results_dash['rl_expect'] = results_dash.rl_expect.replace(parms['index_rl'])
        # print("Final Dataframe Before Display")
        # print(pred_results_df)
        if parms['mode'] in ['batch']:
            results_dash = ModelPredictor.dashboard_prediction_intial_manuplation(pred_results_df, parms)
            #as the static function is calling static function class has to be mentioned
            ModelPredictor.dashboard_prediction_batch(results_dash, parms)
        elif parms['mode'] in ['next']:
            # ModelPredictor.dashboard_prediction_next(results_dash, parms, result_dash_hist_execution)
            # ModelPredictor.dashboard_prediction_next(pred_results_df, parms)
            ModelPredictor.dashboard_nextprediction_write(pred_results_df, parms, confirmation_results)

    @staticmethod
    def dashboard_prediction_intial_manuplation(pred_results_df, parms):
        # if parms['mode'] == 'next':
        #     results_dash = pred_results_df[['ac_prefix', 'rl_prefix', 'tm_prefix', 'pref_size', 'run_num', 'implementation']].copy()
        #     results_dash = pred_results_df.drop(['ac_prefix', 'rl_prefix', 'tm_prefix', 'pref_size', 'run_num', 'implementation'], axis=1)
        # elif parms['mode'] == 'batch':
        results_dash = pred_results_df[['ac_prefix', 'rl_prefix', 'tm_prefix', 'run_num', 'implementation']].copy()
        results_dash = pred_results_df.drop(['ac_prefix', 'rl_prefix', 'tm_prefix', 'run_num', 'implementation'], axis=1)
        # Replacing from Dictionary Values to it's original name
        results_dash['ac_expect'] = results_dash.ac_expect.replace(parms['index_ac'])
        results_dash['rl_expect'] = results_dash.rl_expect.replace(parms['index_rl'])
        return results_dash


    @staticmethod
    def dashboard_prediction_next(results_dash, parms):

        if parms['variant'] in ['multi_pred']:

            #converting the values to it's actual name from parms
            #--For Activity and Role
            results_dash = ModelPredictor.dashboard_multiprediction_acrl(results_dash, parms)


        elif parms['next_mode'] == 'next_action': #--Because it has two coulns to show one for SME and one for preditiction 1

            results_dash = ModelPredictor.dashboard_multiprediction_acrl(results_dash, parms)

        else:
            #converting the values to it's actual name from parms
            #--For Activity, Role
            results_dash = ModelPredictor.dashboard_maxprediction(results_dash, parms)

        # ModelPredictor.dashboard_nextprediction_write(results_dash, parms, result_dash_hist_execution)
        return results_dash


    @staticmethod
    def dashboard_prediction_batch(results_dash, parms):
        #All the results has to be displayed in Tabular form i.e DataFrame
        if parms['variant'] in ['multi_pred', 'multi_pred_rand']:
            #converting the values to it's actual name from parms
            #--For Activity and Role
            ModelPredictor.dashboard_multiprediction_acrl(results_dash, parms)

            results_dash.drop(['ac_pred', 'ac_prob', 'rl_pred', 'rl_prob'], axis=1, inplace=True)

            multipreddict = ModelPredictor.dashboard_multiprediction_columns(parms)

            if parms['batchpredchoice'] in ['Prediction', 'Generative'] and parms['batch_mode'] == 'pre_prefix':
                _lst = [['caseid', 'pref_size', 'ac_expect'] + multipreddict["ac_pred"] + multipreddict["ac_prob"] +
                        ['rl_expect'] + multipreddict["rl_pred"] + multipreddict["rl_prob"] + ['tm_expect'] + multipreddict['tm_pred']]
            else:
                _lst = [['caseid', 'pref_size', 'ac_expect'] + multipreddict["ac_pred"] + multipreddict["ac_prob"] +
                        ['rl_expect'] + multipreddict["rl_pred"] + multipreddict["rl_prob"] +
                        ['tm_expect', "end_timestamp_expected", 'tm_pred', "end_timestamp_pred"]]

            results_dash = results_dash[_lst[0]]

            #--rearrange
            # results_dash = results_dash[['caseid', 'pref_size', 'ac_expect', 'ac_pred1', 'ac_prob1',
            #                              'ac_pred2', 'ac_prob2', 'ac_pred3', 'ac_prob3', 'rl_expect',
            #                              'rl_pred1', 'rl_prob1', 'rl_pred2', 'rl_prob2', 'rl_pred3',
            #                              'rl_prob3', 'tm_expect', 'tm_pred']]
            #--rename
            for _iz in range(parms['multiprednum']):
                for _, _jz in enumerate(results_dash.columns):
                    if _jz[:2] == 'ac':
                        if _jz[3:8] == 'pred'+str(_iz+1):
                            results_dash.rename(
                                columns={_jz : nw(_iz + 1, lang="en", to="ordinal_num") + " AC Prediction"}, inplace=True)
                        elif _jz[3:8] == 'prob'+str(_iz+1):
                            results_dash.rename(
                                columns={_jz: nw(_iz + 1, lang="en", to="ordinal_num") + " AC Confidence"}, inplace=True)
                    elif _jz[:2] == 'rl':
                        if _jz[3:8] == 'pred'+str(_iz+1):
                            results_dash.rename(
                                columns={_jz : nw(_iz + 1, lang="en", to="ordinal_num") + " RL Prediction"}, inplace=True)
                            # continue
                        elif _jz[3:8] == 'prob'+str(_iz+1):
                            results_dash.rename(
                                columns={_jz : nw(_iz + 1, lang="en", to="ordinal_num") + " RL Confidence"}, inplace=True)
                    else:
                        if parms['batchpredchoice'] in ['Prediction', 'Generative'] and parms['batch_mode'] == 'pre_prefix':
                            if _jz[:2] == 'tm':
                                if _jz[3:8] == 'pred' + str(_iz + 1):
                                    results_dash.rename(
                                        columns={_jz: nw(_iz + 1, lang="en", to="ordinal_num") + " TM Prediction"},
                                        inplace=True)

            if parms['batchpredchoice'] in ['Prediction', 'Generative'] and parms['batch_mode'] == 'pre_prefix':
                results_dash.rename(
                    columns={'caseid': 'Case ID', 'pref_size': 'Event_Number', 'ac_expect': 'AC Expected',
                             'rl_expect': 'RL Expected', 'tm_expect': 'TM Expected (Seconds)'}, inplace=True)
            else:
                results_dash.rename(
                    columns={'caseid': 'Case ID', 'pref_size': 'Event_Number', 'ac_expect': 'AC Expected',
                             'rl_expect': 'RL Expected', 'tm_expect': 'TM Expected (Seconds)',
                             "end_timestamp_expected": "Timestamp Expected", 'tm_pred': 'TM Predicted (Seconds)',
                             "end_timestamp_pred": "Timestamp Predicted"}, inplace=True)

        else:
            #converting the values to it's actual name from parms
            #--For Activity, Role
            results_dash = ModelPredictor.dashboard_maxprediction(results_dash, parms)
            results_dash.rename(
                columns={'caseid': 'Case_ID', 'pref_size': 'Event_Number', 'ac_expect': 'AC Expected',
                         'ac_pred': 'AC Predicted', 'ac_prob': 'AC Confidence', 'rl_expect': 'RL Expected',
                         'rl_pred': 'RL Predicted', 'rl_prob': 'RL Confidence', 'tm_expect': 'TM Expected (Seconds)',
                         "end_timestamp_expected": "Timestamp Expected", 'tm_pred': 'TM Predicted (Seconds)',
                         "end_timestamp_pred": "Timestamp Predicted"}, inplace=True)

            # results_dash.columns = pd.MultiIndex.from_tuples(
            #     zip(['', 'Activity', '', '', 'Role', '', '', 'Time', ''],
            #         results_dash.columns))

        # with st.expander('ℹ️'):
        #     st.info("🔮 Batch processed Prediction")
        #--Rounding all the decimal value
        results_dash = results_dash.round(2)
        st.subheader("🔮 Batch Processed Predictions")
        # AgGrid(results_dash, fit_columns_on_grid_load=True)
        AgGrid(results_dash)

    @staticmethod
    def dashboard_multiprediction_acrl(results_dash, parms):

        multipreddict = ModelPredictor.dashboard_multiprediction_columns(parms)

        # --------------------results_dash['ac_pred'] = results_dash.ac_pred.replace(parms['index_ac'])
        for ix in range(len(results_dash['ac_pred'])):
            for jx in range(len(results_dash['ac_pred'][ix])):
                # replacing the value from the parms dictionary
                results_dash['ac_pred'][ix].append(parms['index_ac'][results_dash.ac_pred[ix][jx]])
                # Converting probability into percentage
                results_dash['ac_prob'][ix][jx] = (results_dash['ac_prob'][ix][jx] * 100)
            # poping out the values from the list
            ln = int(len(results_dash['ac_pred'][ix]) / 2)
            #st.session_state['_activity_pred'] = results_dash['ac_pred'][ix][:ln]
            del results_dash['ac_pred'][ix][:ln]
            results_dash[multipreddict["ac_pred"]] = pd.DataFrame(results_dash.ac_pred.tolist(),
                                                                  index=results_dash.index)
            results_dash[multipreddict["ac_prob"]] = pd.DataFrame(results_dash.ac_prob.tolist(),
                                                                  index=results_dash.index)
        #print("Session State At dashboard_multiprediction_acrl After for loop: ", st.session_state)
        # --------------------results_dash['rl_pred'] = results_dash.rl_pred.replace(parms['index_rl'])
        for ix in range(len(results_dash['rl_pred'])):
            for jx in range(len(results_dash['rl_pred'][ix])):
                # replacing the value from the parms dictionary
                results_dash['rl_pred'][ix].append(parms['index_rl'][results_dash.rl_pred[ix][jx]])
                # Converting probability into percentage
                results_dash['rl_prob'][ix][jx] = (results_dash['rl_prob'][ix][jx] * 100)
            # popping out the values from the list
            ln = int(len(results_dash['rl_pred'][ix]) / 2)
            del results_dash['rl_pred'][ix][:ln]
            results_dash[multipreddict["rl_pred"]] = pd.DataFrame(results_dash.rl_pred.tolist(),
                                                                  index=results_dash.index)

            results_dash[multipreddict["rl_prob"]] = pd.DataFrame(results_dash.rl_prob.tolist(),
                                                         index=results_dash.index)

        if parms['next_mode'] == 'next_action' or (parms['mode'] == 'batch' and parms['batchpredchoice'] in ['Prediction', 'Generative'] and parms['batch_mode'] == 'pre_prefix' and parms[
                    'variant'] in ['multi_pred', 'multi_pred_rand']):
            # --------------------results_dash['tm_pred']
            for ix in range(len(results_dash['tm_pred'])):
                results_dash[multipreddict["tm_pred"]] = pd.DataFrame(results_dash.tm_pred.tolist(),
                                                                      index=results_dash.index)

        return results_dash

    @staticmethod
    @st.cache(persist=True, allow_output_mutation=True)
    def dashboard_maxprediction(results_dash, parms):
        results_dash['ac_pred'] = results_dash.ac_pred.replace(parms['index_ac'])
        results_dash['rl_pred'] = results_dash.rl_pred.replace(parms['index_rl'])
        results_dash['ac_prob'] = (results_dash['ac_prob'] * 100)
        results_dash['rl_prob'] = (results_dash['rl_prob'] * 100)
        return results_dash

    @staticmethod
    # def dashboard_nextprediction_write(results_dash, parms, result_dash_hist_execution):
    def dashboard_nextprediction_write(pred_results_df, parms, confirmation_results):

        if parms['next_mode'] in ['history_with_next', 'what_if']:

            # print("State of the Sate Space WhatIf: ", st.session_state)
            # print("-------------------------------------------------------")

            results_dash = ModelPredictor.dashboard_prediction_intial_manuplation(pred_results_df, parms)

            results_dash = ModelPredictor.dashboard_prediction_next(results_dash, parms)

            if parms['next_mode'] == 'history_with_next':

                ModelPredictor.dashboard_nextprediction_execute_write(results_dash, parms)

            elif parms['next_mode'] == 'what_if':

                ModelPredictor.dashboard_nextprediction_whatif_write(results_dash, parms)

        elif parms['next_mode'] == 'next_action':

            # -----------------------------------------------------------------------------------------------------------------------------------
            # Removing 'ac_prefix', 'rl_prefix', 'tm_prefix', 'run_num', 'implementation' from the result
            # creating the prefix for execution mode only, first the last row has been selected to accomodate all the input for
            # activity, role, time. for time it has been converted from numpy array to array of array and then to flat array
            # all the three attributes has been put in dictionary and then convrted to dataframe
            # print("Activity : ", pred_results_df['ac_prefix'].iloc[-1:].values.tolist()[0])
            # print("Role : ", pred_results_df['rl_prefix'].iloc[-1:].values.tolist()[0])
            # print("Time : ", sum([x.tolist() for x in pred_results_df['tm_prefix'].iloc[-1:].values.tolist()[0]], []))

            result_dash_hist_execution = pd.DataFrame(
                {"Activity": pred_results_df['ac_prefix'].iloc[-1:].values.tolist()[0],
                 "Role": pred_results_df['rl_prefix'].iloc[-1:].values.tolist()[0],
                 "Time": sum([x.tolist() for x in pred_results_df['tm_prefix'].iloc[-1:].values.tolist()[0]], [])})

            # Replacing from Dictionary Values to it's original name
            result_dash_hist_execution['Activity'] = result_dash_hist_execution.Activity.replace(parms['index_ac'])
            result_dash_hist_execution['Role'] = result_dash_hist_execution.Role.replace(parms['index_rl'])
            # removing the fist row
            result_dash_hist_execution = result_dash_hist_execution.iloc[1:]
            # -----------------------------------------------------------------------------------------------------------------------------------

            results_dash = ModelPredictor.dashboard_prediction_intial_manuplation(pred_results_df, parms)

            results_dash = ModelPredictor.dashboard_prediction_next(results_dash, parms)

            ModelPredictor.dashboard_nextprediction_evaluate_write(results_dash, parms, result_dash_hist_execution, confirmation_results)

        # elif parms['next_mode'] == 'what_if':
        #
        #     results_dash = ModelPredictor.dashboard_prediction_intial_manuplation(pred_results_df, parms)
        #
        #     results_dash = ModelPredictor.dashboard_prediction_next(results_dash, parms)
        #
        #     ModelPredictor.dashboard_nextprediction_whatif_write(results_dash, parms)

        # print("------------------------------How Session State Looks Like : ", st.session_state)

    @staticmethod
    def dashboard_nextprediction_execute_write(results_dash, parms):
        st.header('📜 Process Historical Behaviour')
        results_dash_expected = results_dash[['ac_expect', 'rl_expect', "tm_expect"]]
        results_dash_expected.rename(
            columns={'ac_expect': 'Activity', 'pos_rl_ss': 'Role', "pos_tm_ss": 'Time'},
            inplace=True)
        st.dataframe(results_dash_expected.iloc[:-1])
        st.markdown("""---""")
        if parms['variant'] in ['multi_pred']:

            multipreddict = ModelPredictor.dashboard_multiprediction_columns(parms)

            with st.container():
                colstm = st.columns(1)
                with colstm[0]:
                    st.subheader('⌛ Predicted Time Duration of Predictions')
                    st.write(results_dash[["tm_pred"]].rename(columns={"tm_pred": 'Expected'}, inplace=False).iloc[-1:].T, use_column_width=True)
            cols = st.columns(parms['multiprednum'])

            for kz in range(parms['multiprednum']):
                    with cols[kz]:
                        ModelPredictor.dashboard_nextprediction_write_acrl(results_dash, parms, multipreddict, kz)
            #st.markdown("""---""")

        else:
            st.header("🤔 Max Probability Prediction")
            cols1, cols2, cols3, cols4 = st.columns([2, 2, 1, 0.5])
            with cols1:
                st.subheader('🏋️ Activity')
                # writes Activity and it's respective confidence on the dashboard with the renamed coulumns name but not modified in the dataframe
                st.write(results_dash[["ac_pred", "ac_prob"]].rename(
                    columns={"ac_pred": 'Predicted', "ac_prob": 'Confidence'}, inplace=False).iloc[-1:])
            with cols2:
                st.subheader('👨‍💻 Role')
                # writes Role and it's respective confidence on the dashboard with the renamed coulumns name but not modified in the dataframe
                st.write(results_dash[["rl_pred", "rl_prob"]].rename(
                    columns={"rl_pred": 'Predicted', "rl_prob": 'Confidence'}, inplace=False).iloc[-1:])
            with cols3:
                st.subheader('⌛ Time')
                st.write(results_dash[["tm_pred"]].rename(columns={"tm_pred": 'Predicted'}, inplace=False).iloc[-1:])
            with cols4:
                st.subheader('🏷️ Label')
                ModelPredictor.dashboard_label_decider(sum(results_dash[["ac_pred"]].values.tolist(), []), parms)
            st.markdown("""---""")
        #--Predictions of Predictions
        if st.session_state['multi_pred_ss']['ss_multipredict1']['ac_pred'] != []:
            ModelPredictor.dashboard_nextprediction_execute_multiverse(parms)

    @staticmethod
    def dashboard_nextprediction_execute_multiverse(parms):
        for lk in range(parms['multiprednum']):
            st.header("📜 " + nw(lk + 1, lang="en", to="ordinal_num") + " Prediction " + " Historical Behaviour ")

            _hist_columns = ['pos_ac_ss', 'pos_rl_ss']  # selecting the columns
            _hist_predicted_dict = dict(
                [(k, st.session_state['initial_prediction']['ss_initpredict' + str(lk + 1)][k]) for k in
                 _hist_columns])  # constructing new dict fromm sessionstate
            _hist_predicted_dict.update(dict([(k, st.session_state[k]) for k in ['pos_tm_ss']]))  # Apending time
            # --Manuplation to see the Value properly
            _hist_predicted_dict['pos_tm_ss'] = sum(_hist_predicted_dict['pos_tm_ss'], [])  # flattening of time
            _hist_predicted_dict['pos_tm_ss'] = [ModelPredictor.rescale(x, parms, parms['scale_args']) for x in
                                                 _hist_predicted_dict[
                                                     'pos_tm_ss']]  # Normalizing back to original value
            _hist_predicted_dict = {k: _hist_predicted_dict[k][1:] for k in
                                    _hist_predicted_dict}  # removing first item in each key in a dictionary

            _hist_predicted_df = pd.DataFrame.from_dict(_hist_predicted_dict)
            # Replacing from Dictionary Values to it's original name
            _hist_predicted_df['pos_ac_ss'] = _hist_predicted_df.pos_ac_ss.replace(parms['index_ac'])
            _hist_predicted_df['pos_rl_ss'] = _hist_predicted_df.pos_rl_ss.replace(parms['index_rl'])
            _hist_predicted_df.rename(
                columns={'pos_ac_ss': 'Activity', 'pos_rl_ss': 'Role', "pos_tm_ss": 'Time'},
                inplace=True)

            # _hist_predicted_df = _hist_predicted_df.iloc[1:]
            st.dataframe(_hist_predicted_df.iloc[:-1])
            st.markdown("""---""")
            _multi_columns = ['ac_pred', 'ac_prob', 'rl_pred', 'rl_prob', 'tm_pred']

            _multi_predicted_dict = dict(
                [(k, st.session_state['multi_pred_ss']['ss_multipredict' + str(lk + 1)][k]) for k in _multi_columns])
            _multi_predicted_dict['tm_pred'] = sum(_multi_predicted_dict['tm_pred'], [])  # flattening of time
            _multi_predicted_dict['tm_pred'] = [ModelPredictor.rescale(x, parms, parms['scale_args']) for x in
                                                _multi_predicted_dict['tm_pred']]  # Normalizing back to original value
            _multi_predicted_df = pd.DataFrame.from_dict(_multi_predicted_dict)
            _multi_predicted_df = ModelPredictor.dashboard_maxprediction(_multi_predicted_df, parms)
            _multi_predicted_df.index = _multi_predicted_df.index + 1 #to match with the index value of the main prediction
            st.subheader("🔮 Max Probability Prediction of " + nw(lk + 1, lang="en", to="ordinal_num") + " Prediction")
            cols1, cols2, cols3, cols4 = st.columns([2, 2, 1, 0.5])
            with cols1:
                st.subheader('🏋️ Activity')
                # writes Activity and it's respective confidence on the dashboard with the renamed coulumns name but not modified in the dataframe
                st.write(_multi_predicted_df[["ac_pred", "ac_prob"]].rename(
                    columns={"ac_pred": 'Predicted', "ac_prob": 'Confidence'}, inplace=False).iloc[-1:])
            with cols2:
                st.subheader('👨‍💻 Role')
                # writes Role and it's respective confidence on the dashboard with the renamed coulumns name but not modified in the dataframe
                st.write(_multi_predicted_df[["rl_pred", "rl_prob"]].rename(
                    columns={"rl_pred": 'Predicted', "rl_prob": 'Confidence'}, inplace=False).iloc[-1:])
            with cols3:
                st.subheader('⌛ Time')
                st.write(
                    _multi_predicted_df[["tm_pred"]].rename(columns={"tm_pred": 'Predicted'}, inplace=False).iloc[-1:])
            with cols4:
                st.subheader('🏷️ Label')
                _lbkey = [sum(_hist_predicted_df[['Activity']].values.tolist(), [])[0]] + sum(_multi_predicted_df[["ac_pred"]].values.tolist(), []) #concatenate
                # _lbkey = sorted(set(_lbkey), key=_lbkey.index) #remove duplicates and maintain the order
                # print("Multiverse " + str(lk), _lbkey, [sum(_hist_predicted_df[['Activity']].values.tolist(), [])[0]], sum(_multi_predicted_df[["ac_pred"]].values.tolist(), []))
                ModelPredictor.dashboard_label_decider(_lbkey, parms)

    @staticmethod
    def dashboard_nextprediction_evaluate_write(results_dash, parms, result_dash_hist_execution, confirmation_results):

        # process_history_behaviour, label_status = ModelPredictor.dashboard_history_label()
        # with st.container():
        #     colshist, colslabel = st.columns(2)

        # Fixing Index Number for better readability
        _temp_result_dash_hist_execution = result_dash_hist_execution.iloc[:parms['nextcaseid_attr']["filter_index"] + 1]
        _temp_result_dash_hist_execution.index = _temp_result_dash_hist_execution.index - 1

        st.header('📜 Process Historical Behaviour')
        st.dataframe(_temp_result_dash_hist_execution)
        st.markdown("""---""")

        # print("Prediction Dataframe  : ", results_dash.columns)

        # print("Parameter : ", parms)
        results_dash.index = results_dash.index + (parms['nextcaseid_attr']["filter_index"] + 1)

        # if parms['variant'] in ['multi_pred']:
        cols = st.columns(parms['multiprednum']+2) #One for SME Case another for expecte

        multipreddict = ModelPredictor.dashboard_multiprediction_columns(parms)
        _iterations = parms['multiprednum']+2 #One for SME Case another for expected
        for kz in range(_iterations):
            if kz <= (_iterations-2):
                with cols[kz]:
                    ModelPredictor.dashboard_nextprediction_write_acrl(results_dash, parms, multipreddict, kz)

            elif kz == (_iterations-1) and parms['next_mode']:
                with cols[kz]:
                    st.header("🧐 Expected ")

                    st.subheader('🏋️ Activity')
                    st.write(results_dash[["ac_expect"]][0:1].rename(columns={"ac_expect": 'Expected'}, inplace=False))
                    with st.expander('ℹ️'):
                        st.info("Suffix of Expected Activity")
                    # writes Activity and it's respective confidence on the dashboard with the renamed coulumns name but not modified in the dataframe
                        st.write(results_dash[["ac_expect"]][1:].rename(columns={"ac_expect": 'Expected'}, inplace=False))

                    st.markdown("""---""")
                    st.subheader('👨‍💻 Role')
                    # writes Role and it's respective confidence on the dashboard with the renamed coulumns name but not modified in the dataframe
                    st.write(results_dash[["rl_expect"]][0:1].rename(columns={"rl_expect": 'Expected'}, inplace=False))
                    with st.expander('ℹ️'):
                        st.info("Suffix of Expected Role")
                        st.write(results_dash[["rl_expect"]][1:].rename(columns={"rl_expect": 'Expected'}, inplace=False))

                    st.markdown("""---""")

                    st.subheader('⌛ Time')
                    st.write(results_dash[["tm_expect"]][0:1].rename(columns={"tm_expect": 'Expected'}, inplace=False))
                    with st.expander('ℹ️'):
                        st.info("Suffix of Expected Time")
                        st.write(results_dash[["tm_expect"]][1:].rename(columns={"tm_expect": 'Expected'}, inplace=False))

                    st.markdown("""---""")

                    st.subheader('🏷️ Label')
                    ModelPredictor.dashboard_label_decider(sum(results_dash[["ac_expect"]].values.tolist(), []), parms)

                    st.markdown("""---""")

        confirmation_results = confirmation_results.drop(columns=['conf_ac_prefix', 'conf_rl_prefix', 'conf_tm_prefix'])

        confirmation_results['conf_ac_expect'] = confirmation_results.conf_ac_expect.replace(parms['index_ac'])
        confirmation_results['conf_rl_expect'] = confirmation_results.conf_rl_expect.replace(parms['index_rl'])

        #------Multipreditict
        conf_ac_pred_lst = []
        conf_ac_prob_lst = []
        conf_rl_pred_lst = []
        conf_rl_prob_lst = []
        conf_tm_pred_lst = []

        multipreddict = {}

        for zx in range(parms['multiprednum']):
            zx += 1
            conf_ac_pred_lst.append("conf_ac_pred" + str(zx))
            conf_ac_prob_lst.append("conf_ac_prob" + str(zx))
            conf_rl_pred_lst.append("conf_rl_pred" + str(zx))
            conf_rl_prob_lst.append("conf_rl_prob" + str(zx))
        conf_tm_pred_lst.append("conf_tm_pred")

        multipreddict["conf_ac_pred"] = conf_ac_pred_lst
        multipreddict["conf_ac_prob"] = conf_ac_prob_lst
        multipreddict["conf_rl_pred"] = conf_rl_pred_lst
        multipreddict["conf_rl_prob"] = conf_rl_prob_lst
        multipreddict["conf_tm_pred"] = conf_tm_pred_lst

        for ix in range(len(confirmation_results['conf_ac_pred'])):
            for jx in range(len(confirmation_results['conf_ac_pred'][ix])):
                confirmation_results['conf_ac_pred'][ix].append(parms['index_ac'][confirmation_results.conf_ac_pred[ix][jx]])
                confirmation_results['conf_ac_prob'][ix][jx] = (confirmation_results['conf_ac_prob'][ix][jx] * 100)
            ln = int(len(confirmation_results['conf_ac_pred'][ix]) / 2)
            del confirmation_results['conf_ac_pred'][ix][:ln]
            confirmation_results[multipreddict["conf_ac_pred"]] = pd.DataFrame(confirmation_results.conf_ac_pred.tolist(),
                                                                                index=confirmation_results.index)
            confirmation_results[multipreddict["conf_ac_prob"]] = pd.DataFrame(confirmation_results.conf_ac_prob.tolist(),
                                                                                index=confirmation_results.index)
        for ix in range(len(confirmation_results['conf_rl_pred'])):
            for jx in range(len(confirmation_results['conf_rl_pred'][ix])):
                confirmation_results['conf_rl_pred'][ix].append(parms['index_rl'][confirmation_results.conf_rl_pred[ix][jx]])
                confirmation_results['conf_rl_prob'][ix][jx] = (confirmation_results['conf_rl_prob'][ix][jx] * 100)
            ln = int(len(confirmation_results['conf_rl_pred'][ix]) / 2)
            del confirmation_results['conf_rl_pred'][ix][:ln]
            confirmation_results[multipreddict["conf_rl_pred"]] = pd.DataFrame(confirmation_results.conf_rl_pred.tolist(),
                                                                                index=confirmation_results.index)
            confirmation_results[multipreddict["conf_rl_prob"]] = pd.DataFrame(confirmation_results.conf_rl_prob.tolist(),
                                                                                index=confirmation_results.index)
        for ix in range(len(confirmation_results['conf_tm_pred'])):
            confirmation_results[multipreddict["conf_tm_pred"]] = pd.DataFrame(confirmation_results.conf_tm_pred.tolist(),
                                                                                index=confirmation_results.index)

        _lst = [['conf_ac_expect'] + multipreddict["conf_ac_pred"] + multipreddict["conf_ac_prob"] +
                ['conf_rl_expect'] + multipreddict["conf_rl_pred"] + multipreddict["conf_rl_prob"] +
                ['conf_tm_expect'] + multipreddict['conf_tm_pred']]

        confirmation_results = confirmation_results[_lst[0]]

        # --rename dynamically
        for _iz in range(parms['multiprednum']):
            for _, _jz in enumerate(confirmation_results.columns):
                if _jz[:7] == 'conf_ac':
                    if _jz[8:13] == 'pred' + str(_iz + 1):
                        confirmation_results.rename(
                            columns={_jz: nw(_iz + 1, lang="en", to="ordinal_num") + " AC Prediction"}, inplace=True)
                    elif _jz[8:13] == 'prob' + str(_iz + 1):
                        confirmation_results.rename(
                            columns={_jz: nw(_iz + 1, lang="en", to="ordinal_num") + " AC Confidence"}, inplace=True)
                elif _jz[:7] == 'conf_rl':
                    if _jz[8:13] == 'pred' + str(_iz + 1):
                        confirmation_results.rename(
                            columns={_jz: nw(_iz + 1, lang="en", to="ordinal_num") + " RL Prediction"}, inplace=True)
                    elif _jz[8:13] == 'prob' + str(_iz + 1):
                        confirmation_results.rename(
                            columns={_jz: nw(_iz + 1, lang="en", to="ordinal_num") + " RL Confidence"}, inplace=True)

        confirmation_results.rename(
            columns={'conf_ac_expect': 'AC Expected', 'conf_rl_expect': 'RL Expected', 'conf_tm_expect': 'TM Expected (Seconds)', 'conf_tm_pred': 'TM Predicted (Seconds)'}, inplace=True)


        confirmation_results = confirmation_results.round(2)
        st.subheader("🤝 Conformance Check")
        # AgGrid(confirmation_results)
        st.write(confirmation_results)



            #st.markdown("""---""")
            # with st.container():
            #     colstm = st.columns(2)
            #     with colstm[0]:
            #         st.subheader('⌛ Predicted Time Duration')
            #         st.write(results_dash[["tm_pred"]].rename(columns={"tm_pred": 'Predicted'}, inplace=False).T,
            #                  use_column_width=True)
            #     with colstm[1]:
            #         st.subheader('⌚ Expected Time Duration')
            #         st.write(
            #             results_dash[["tm_expect"]].rename(columns={"tm_expect": 'Expected'}, inplace=False).T,
            #             use_column_width=True)

        # else:
        #     st.header("🤔 Max Probability Prediction")
        #     cols1, cols2, cols3, cols4 = st.columns([2, 2, 1, 0.5])
        #     with cols1:
        #         st.subheader('🏋️ Activity')
        #         # writes Activity and it's respective confidence on the dashboard with the renamed coulumns name but not modified in the dataframe
        #         st.write(results_dash[["ac_pred", "ac_prob"]].rename(columns={"ac_pred": 'Predicted', "ac_prob": 'Confidence'}, inplace=False))
        #     with cols2:
        #         st.subheader('👨‍💻 Role')
        #         # writes Role and it's respective confidence on the dashboard with the renamed coulumns name but not modified in the dataframe
        #         st.write(results_dash[["rl_pred", "rl_prob"]].rename(columns={"rl_pred": 'Predicted', "rl_prob": 'Confidence'}, inplace=False))
        #     with cols3:
        #         st.subheader('⌛ Time')
        #         st.write(results_dash[["tm_pred"]].rename(columns={"tm_pred": 'Predicted'}, inplace=False))
        #     with cols4:
        #         st.subheader('🏷️ Label')
        #         ModelPredictor.dashboard_label_decider(sum(results_dash[["ac_pred"]].values.tolist(), []), parms)
        #     st.markdown("""---""")
        #
        #     st.header("🧐 Expected ")
        #     ecols1, ecols2, ecols3, ecols4 = st.columns([2, 2, 1, 0.5])
        #     with ecols1:
        #         st.subheader('🏋️ Activity')
        #         # writes Activity and it's respective confidence on the dashboard with the renamed coulumns name but not modified in the dataframe
        #         st.write(results_dash[["ac_expect"]].rename(columns={"ac_expect": 'Expected'}, inplace=False))
        #         st.markdown("""---""")
        #     with ecols2:
        #         st.subheader('👨‍💻 Role')
        #         # writes Role and it's respective confidence on the dashboard with the renamed coulumns name but not modified in the dataframe
        #         st.write(results_dash[["rl_expect"]].rename(columns={"rl_expect": 'Expected'}, inplace=False))
        #     with ecols3:
        #         st.subheader('⌛ Time')
        #         st.write(results_dash[["tm_expect"]].rename(columns={"tm_expect": 'Expected'}, inplace=False))
        #     with ecols4:
        #         st.subheader('🏷️ Label')
        #         ModelPredictor.dashboard_label_decider(sum(results_dash[["ac_expect"]].values.tolist(), []), parms)

    @staticmethod
    def dashboard_nextprediction_whatif_write(results_dash, parms):
        st.header('📜 Process Historical Behaviour')
        _hist_columns = ['hist_ac_prefix', 'hist_rl_prefix', 'hist_tm_prefix', 'hist_pred_prefix']  # selecting the columns
        _hist_choice_dict = dict([(k, st.session_state['history_of_choice'][k]) for k in _hist_columns])  # constructing new dict fromm sessionstate
        _hist_choice_dict['hist_tm_prefix'] = sum(_hist_choice_dict['hist_tm_prefix'], []) # flattening of time
        _hist_choice_dict['hist_tm_prefix'] = [ModelPredictor.rescale(x, parms, parms['scale_args']) for x in _hist_choice_dict['hist_tm_prefix']]  # Normalizing back to original value
        _hist_choice_df = pd.DataFrame.from_dict(_hist_choice_dict)
        # Replacing from Dictionary Values to it's original name
        _hist_choice_df ['hist_ac_prefix'] = _hist_choice_df.hist_ac_prefix.replace(parms['index_ac'])
        _hist_choice_df['hist_rl_prefix'] = _hist_choice_df.hist_rl_prefix.replace(parms['index_rl'])

        _hist_choice_df.rename(columns={'hist_ac_prefix': 'Activity', 'hist_rl_prefix': 'Role', 'hist_tm_prefix': 'Time', 'hist_pred_prefix':'Choice'}, inplace=True)

        st.dataframe(_hist_choice_df)
        st.markdown("""---""")
        if parms['variant'] in ['multi_pred']:

            multipreddict = ModelPredictor.dashboard_multiprediction_columns(parms)

            with st.container():
                colstm = st.columns(1)
                with colstm[0]:
                    st.subheader('⌛ Predicted Time Duration of Predictions')
                    st.write(results_dash[["tm_pred"]].rename(columns={"tm_pred": 'Expected'}, inplace=False).iloc[-1:].T, use_column_width=True)
            cols = st.columns(parms['multiprednum'])

            for kz in range(parms['multiprednum']):
                    with cols[kz]:
                        ModelPredictor.dashboard_nextprediction_write_acrl(results_dash, parms, multipreddict, kz)
            #st.markdown("""---""")

        else:
            st.header("🤔 Max Probability Prediction")
            cols1, cols2, cols3, cols4 = st.columns([2, 2, 1, 0.5])
            with cols1:
                st.subheader('🏋️ Activity')
                # writes Activity and it's respective confidence on the dashboard with the renamed coulumns name but not modified in the dataframe
                st.write(results_dash[["ac_pred", "ac_prob"]].rename(
                    columns={"ac_pred": 'Predicted', "ac_prob": 'Confidence'}, inplace=False).iloc[-1:])
            with cols2:
                st.subheader('👨‍💻 Role')
                # writes Role and it's respective confidence on the dashboard with the renamed coulumns name but not modified in the dataframe
                st.write(results_dash[["rl_pred", "rl_prob"]].rename(
                    columns={"rl_pred": 'Predicted', "rl_prob": 'Confidence'}, inplace=False).iloc[-1:])
            with cols3:
                st.subheader('⌛ Time')
                st.write(results_dash[["tm_pred"]].rename(columns={"tm_pred": 'Predicted'}, inplace=False).iloc[-1:])
            with cols4:
                st.subheader('🏷️ Label')
                ModelPredictor.dashboard_label_decider(sum(results_dash[["ac_pred"]].values.tolist(), []), parms)
            st.markdown("""---""")

        #--Predictions of Predictions
        # if st.session_state['multi_pred_ss']['ss_multipredict1']["multiverse_predict1"]['ac_pred'] != []:
        #     ModelPredictor.dashboard_nextprediction_whatif_multiverse(parms)

    @staticmethod
    def dashboard_nextprediction_whatif_multiverse(parms):
        st.info("Predictions of what could happen for the respective above predictions if selected")
        for lk in range(parms['multiprednum']):
            st.header("📜 " + nw(lk + 1, lang="en", to="ordinal_num") + " Prediction " + " Historical Behaviour ")

            _hist_columns = ['pos_ac_ss', 'pos_rl_ss', 'pos_tm_ss']  # selecting the columns
            _hist_multiverse_dict = dict([(k, st.session_state['initial_prediction']['ss_initpredict' + str(lk + 1)][k]) for k in _hist_columns])  # constructing new dict fromm sessionstate

            # --Manuplation to see the Value properly
            _hist_multiverse_dict['pos_tm_ss'] = sum(_hist_multiverse_dict['pos_tm_ss'], [])  # flattening of time

            _hist_multiverse_dict['pos_tm_ss'] = [ModelPredictor.rescale(x, parms, parms['scale_args']) for x in _hist_multiverse_dict['pos_tm_ss']]  # Normalizing back to original value

            _hist_multiverse_dict = {k: _hist_multiverse_dict[k][1:] for k in _hist_multiverse_dict}  # removing first item in each key in a dictionary
            _hist_predicted_df = pd.DataFrame.from_dict(_hist_multiverse_dict)
            # Replacing from Dictionary Values to it's original name
            _hist_predicted_df['pos_ac_ss'] = _hist_predicted_df.pos_ac_ss.replace(parms['index_ac'])
            _hist_predicted_df['pos_rl_ss'] = _hist_predicted_df.pos_rl_ss.replace(parms['index_rl'])
            _hist_predicted_df.rename(columns={'pos_ac_ss': 'Activity', 'pos_rl_ss': 'Role', "pos_tm_ss": 'Time'}, inplace=True)
            # _hist_predicted_df = _hist_predicted_df.iloc[1:]
            st.dataframe(_hist_predicted_df.iloc[:-1])
            st.markdown("""---""")
            _multi_columns = ['ac_pred', 'ac_prob', 'rl_pred', 'rl_prob', 'tm_pred']

            # cols = st.columns(parms['multiprednum'])
            #

            for _lz in range(parms['multiprednum']):
                _multi_multiverse_dict = dict([(k, st.session_state['multi_pred_ss']['ss_multipredict' + str(lk + 1)]['multiverse_predict' + str(_lz+1)][k]) for k in _multi_columns])
                _multi_multiverse_dict['tm_pred'] = sum(_multi_multiverse_dict['tm_pred'], [])  # flattening of time
                _multi_multiverse_dict['tm_pred'] = [ModelPredictor.rescale(x, parms, parms['scale_args']) for x in _multi_multiverse_dict['tm_pred']]  # Normalizing back to original value
                _multi_multiverse_df = pd.DataFrame.from_dict(_multi_multiverse_dict)
                _multi_multiverse_df = ModelPredictor.dashboard_maxprediction(_multi_multiverse_df, parms)
                _multi_multiverse_df.index = _multi_multiverse_df.index + 1  # to match with the index value of the main prediction
                if parms['variant'] in ['multi_pred']:
                    if _lz == 0:
                        st.subheader('⌛ Predicted Time Duration of Predictions')
                        st.write(_multi_multiverse_df[["tm_pred"]].rename(columns={"tm_pred": 'Expected'},
                                                                          inplace=False).iloc[-1:].T, use_column_width=True)
                        cols = st.columns(parms['multiprednum'])

                    with cols[_lz]:
                        st.subheader("🔮 " + nw(_lz + 1, lang="en", to="ordinal_num") + " Prediction of Prediction")
                    #----------------------------------------------------------------------------------------------------------------------------------------
                        st.subheader('🏋️ Activity')
                        # writes Activity and it's respective confidence on the dashboard with the renamed coulumns name but not modified in the dataframe
                        st.write(_multi_multiverse_df[["ac_pred"] +
                                             ["ac_prob"]].iloc[-1:].rename(columns={"ac_pred": 'Predicted', "ac_prob": 'Confidence'}))
                        st.markdown("""---""")

                        st.subheader('👨‍💻 Role')
                        # writes Role and it's respective confidence on the dashboard with the renamed coulumns name but not modified in the dataframe
                        st.write(_multi_multiverse_df[["rl_pred"] +
                                                      ["rl_prob"]].iloc[-1:].rename(columns={"rl_pred": 'Predicted', "rl_prob": 'Confidence'}))
                        st.markdown("""---""")

                        st.subheader('🏷️ Label')
                        _lbkey = [sum(_hist_predicted_df[['Activity']].values.tolist(), [])[0]] + sum(_multi_multiverse_df[["ac_pred"]].values.tolist(), [])
                        ModelPredictor.dashboard_label_decider(_lbkey, parms)
                        st.markdown("""---""")
                else:
                    st.header("🔮 Max Probability Prediction")
                    cols1, cols2, cols3, cols4 = st.columns([2, 2, 1, 0.5])
                    with cols1:
                        st.subheader('🏋️ Activity')
                        # writes Activity and it's respective confidence on the dashboard with the renamed coulumns name but not modified in the dataframe
                        st.write(_multi_multiverse_df[["ac_pred", "ac_prob"]].rename(
                            columns={"ac_pred": 'Predicted', "ac_prob": 'Confidence'}, inplace=False).iloc[-1:])
                    with cols2:
                        st.subheader('👨‍💻 Role')
                        # writes Role and it's respective confidence on the dashboard with the renamed coulumns name but not modified in the dataframe
                        st.write(_multi_multiverse_df[["rl_pred", "rl_prob"]].rename(
                            columns={"rl_pred": 'Predicted', "rl_prob": 'Confidence'}, inplace=False).iloc[-1:])
                    with cols3:
                        st.subheader('⌛ Time')
                        st.write(_multi_multiverse_df[["tm_pred"]].rename(columns={"tm_pred": 'Predicted'}, inplace=False).iloc[-1:])
                    with cols4:
                        st.subheader('🏷️ Label')
                        _lbkey = [sum(_hist_predicted_df[['Activity']].values.tolist(), [])[0]] + sum(_multi_multiverse_df[["ac_pred"]].values.tolist(), [])
                        ModelPredictor.dashboard_label_decider(_lbkey, parms)
                    st.markdown("""---""")

                # st.subheader('🏋️ Activity')
                # # writes Activity and it's respective confidence on the dashboard with the renamed coulumns name but not modified in the dataframe
                # st.write(_multi_multiverse_df[["ac_pred" + str(_lz + 1)] +
                #                      ["ac_prob" + str(_lz + 1)]].iloc[-1:].rename(columns={"ac_pred" + str(_lz + 1): 'Predicted', "ac_prob" + str(_lz + 1): 'Confidence'}))
                # st.markdown("""---""")
                #
                # st.subheader('👨‍💻 Role')
                # # writes Role and it's respective confidence on the dashboard with the renamed coulumns name but not modified in the dataframe
                # st.write(_multi_multiverse_df[["rl_pred" + str(_lz + 1)] +
                #                               ["rl_prob" + str(_lz + 1)]].iloc[-1:].rename(columns={"rl_pred" + str(_lz + 1): 'Predicted', "rl_prob" + str(_lz + 1): 'Confidence'}))
                # st.markdown("""---""")
                #
                # st.subheader('🏷️ Label')
                # # print("Historical data : ", [sum(_hist_predicted_df[['Activity']].values.tolist(), [])[0]])
                # # print("Activity : ", sum(_multi_multiverse_df[["ac_pred" + str(_lz + 1)]].values.tolist(), []))
                # # print("Length of Activity :", len(sum(_multi_multiverse_df[["ac_pred" + str(_lz + 1)]].values.tolist(), [])))
                # _lbkey = [sum(_hist_predicted_df[['Activity']].values.tolist(), [])[0]] + sum(_multi_multiverse_df[["ac_pred" + str(_lz + 1)]].values.tolist(), [])
                # ModelPredictor.dashboard_label_decider(_lbkey, parms)
                # st.markdown("""---""")
                #-------------------------------------------------------
                #     multipreddict = ModelPredictor.dashboard_multiprediction_columns(parms)
                #
                #     with st.container():
                #         colstm = st.columns(1)
                #         with colstm[0]:
                #             st.subheader('⌛ Predicted Time Duration of Predictions')
                #             st.write(_multi_multiverse_df[["tm_pred"]].rename(columns={"tm_pred": 'Expected'}, inplace=False).iloc[-1:].T, use_column_width=True)
                #
                #     with cols[_lz]:
                #         ModelPredictor.dashboard_nextprediction_write_acrl(_multi_multiverse_df, parms, multipreddict, _lz)
                #     st.markdown("""---""")
                #
                # else:
                #     st.subheader("🤔 Max Probability Prediction of " + nw(_lz + 1, lang="en", to="ordinal_num") + " Prediction")
                #     cols1, cols2, cols3, cols4 = st.columns([2, 2, 2, 1])
                #     with cols1:
                #         st.subheader('🏋️ Activity')
                #         # writes Activity and it's respective confidence on the dashboard with the renamed coulumns name but not modified in the dataframe
                #         st.write(_multi_multiverse_df[["ac_pred", "ac_prob"]].rename(
                #             columns={"ac_pred": 'Predicted', "ac_prob": 'Confidence'}, inplace=False).iloc[-1:])
                #     with cols2:
                #         st.subheader('👨‍💻 Role')
                #         # writes Role and it's respective confidence on the dashboard with the renamed coulumns name but not modified in the dataframe
                #         st.write(_multi_multiverse_df[["rl_pred", "rl_prob"]].rename(
                #             columns={"rl_pred": 'Predicted', "rl_prob": 'Confidence'}, inplace=False).iloc[-1:])
                #     with cols4:
                #         st.subheader('⌛ Time')
                #         st.write(_multi_multiverse_df[["tm_pred"]].rename(columns={"tm_pred": 'Predicted'}, inplace=False).iloc[-1:])
                #     st.markdown("""---""")

        #------Supporting Functions


    @staticmethod
    def dashboard_nextprediction_write_acrl(results_dash, parms, multipreddict, kz):
        if kz <= parms['multiprednum']:
            if parms['next_mode'] == 'next_action': #Evaluation Mode
                #st.subheader("🤔", nw(kz + 1, lang="en", to="ordinal_num") + "Prediction")
                if kz == 0:
                    st.header("👨‍💼 SME Predictions")
                else:
                    if parms['variant'] == 'arg_max':
                        st.header("🤔 Max Probability Prediction")
                    else:
                        st.header("🤔 " + nw(kz, lang="en", to="ordinal_num") + " Prediction")
                # print("results_dash : ", multipreddict["ac_pred"][kz])
                # print("Dataframe : ", results_dash)
                st.subheader('🏋️ Activity')
                # writes Activity and it's respective confidence on the dashboard with the renamed coulumns name but not modified in the dataframe
                # st.write(results_dash[[multipreddict["ac_pred"][kz]] + [multipreddict["ac_prob"][kz]]].rename(
                #     columns={multipreddict["ac_pred"][kz]: 'Predicted', multipreddict["ac_prob"][kz]: 'Confidence'},
                #     inplace=False))
                st.write(results_dash[[multipreddict["ac_pred"][kz]] + [multipreddict["ac_prob"][kz]]][0:1].rename(
                    columns={multipreddict["ac_pred"][kz]: 'Predicted', multipreddict["ac_prob"][kz]: 'Confidence'},
                    inplace=False))
                with st.expander('ℹ️'):
                    st.info("Suffix of Predicted Activity")
                    st.write(results_dash[[multipreddict["ac_pred"][kz]] + [multipreddict["ac_prob"][kz]]][1:].rename(
                        columns={multipreddict["ac_pred"][kz]: 'Predicted', multipreddict["ac_prob"][kz]: 'Confidence'},
                        inplace=False))
                st.markdown("""---""")
                st.subheader('👨‍💻 Role')
                # writes Role and it's respective confidence on the dashboard with the renamed coulumns name but not modified in the dataframe
                st.write(results_dash[[multipreddict["rl_pred"][kz]] + [multipreddict["rl_prob"][kz]]][0:1].rename(
                    columns={multipreddict["rl_pred"][kz]: 'Predicted', multipreddict["rl_prob"][kz]: 'Confidence'},
                    inplace=False))
                with st.expander('ℹ️'):
                    st.info("Suffix of Predicted Role")
                    st.write(results_dash[[multipreddict["rl_pred"][kz]] + [multipreddict["rl_prob"][kz]]][1:].rename(
                        columns={multipreddict["rl_pred"][kz]: 'Predicted', multipreddict["rl_prob"][kz]: 'Confidence'},
                        inplace=False))
                st.markdown("""---""")

                st.subheader('⌛ Time')
                # st.write(results_dash[["tm_pred"]].rename(columns={"tm_pred": 'Predicted'}, inplace=False))
                st.write(results_dash[[multipreddict["tm_pred"][kz]]][0:1].rename(
                    columns={multipreddict["tm_pred"][kz]: 'Predicted'}, inplace=False))
                with st.expander('ℹ️'):
                    st.info("Suffix of Predicted Time")
                    st.write(results_dash[[multipreddict["tm_pred"][kz]]][1:].rename(
                        columns={multipreddict["tm_pred"][kz]: 'Predicted'}, inplace=False))

                st.markdown("""---""")

                st.subheader('🏷️ Label')
                ModelPredictor.dashboard_label_decider(sum(results_dash[[multipreddict["ac_pred"][kz]]].values.tolist(), []), parms)
                st.markdown("""---""")
            elif parms['next_mode'] in ['history_with_next', 'what_if']: #Execution Mode
                # st.header("🤔 Prediction " + str(kz + 1))
                st.header("🤔 " + nw(kz + 1, lang="en", to="ordinal_num") + " Prediction")
                # with st.expander('ℹ️'):
                #     st.info("Predicted Events")
                st.subheader('🏋️ Activity')
                # writes Activity and it's respective confidence on the dashboard with the renamed coulumns name but not modified in the dataframe
                st.write(results_dash[[multipreddict["ac_pred"][kz]] + [multipreddict["ac_prob"][kz]]].rename(
                    columns={multipreddict["ac_pred"][kz]: 'Predicted', multipreddict["ac_prob"][kz]: 'Confidence'},
                    inplace=False).iloc[-1:])
                st.markdown("""---""")
                st.subheader('👨‍💻 Role')
                # writes Role and it's respective confidence on the dashboard with the renamed coulumns name but not modified in the dataframe
                st.write(results_dash[[multipreddict["rl_pred"][kz]] + [multipreddict["rl_prob"][kz]]].rename(
                    columns={multipreddict["rl_pred"][kz]: 'Predicted', multipreddict["rl_prob"][kz]: 'Confidence'},
                    inplace=False).iloc[-1:])
                st.markdown("""---""")
                st.subheader('🏷️ Label')
                # print("Label Paramters : ", sum(results_dash[[multipreddict["ac_pred"][kz]]].values.tolist(), []))
                ModelPredictor.dashboard_label_decider(sum(results_dash[[multipreddict["ac_pred"][kz]]].values.tolist(), []), parms)
                st.markdown("""---""")
                # st.subheader('⌛ Time')
                # st.write(results_dash[["tm_pred"]].rename(columns={"tm_pred": 'Predicted'}, inplace=False))

    # ------Supporting Functions
    @staticmethod
    @st.cache(persist=True)
    def dashboard_multiprediction_columns(parms):
        # Initialize list for multi pred
        ac_pred_lst = []
        ac_prob_lst = []
        rl_pred_lst = []
        rl_prob_lst = []

        if parms['next_mode'] == 'next_action' or (parms['mode'] == 'batch' and parms['batchpredchoice'] in ['Prediction', 'Generative'] and parms['batch_mode'] == 'pre_prefix' and parms[
                    'variant'] in ['multi_pred', 'multi_pred_rand']):
            tm_pred_lst = []
            if parms['next_mode'] == 'next_action':
                _numofcols = parms['multiprednum'] + 1
            else:
                _numofcols = parms['multiprednum']
        else :
            _numofcols = parms['multiprednum']

        multipreddict = {}

        for zx in range(_numofcols):
            zx += 1
            ac_pred_lst.append("ac_pred" + str(zx))
            ac_prob_lst.append("ac_prob" + str(zx))
            rl_pred_lst.append("rl_pred" + str(zx))
            rl_prob_lst.append("rl_prob" + str(zx))
            if parms['next_mode'] == 'next_action' or (parms['mode'] == 'batch' and parms['batchpredchoice'] in ['Prediction', 'Generative'] and parms['batch_mode'] == 'pre_prefix' and parms[
                    'variant'] in ['multi_pred', 'multi_pred_rand']):
                tm_pred_lst.append("tm_pred" + str(zx))
        multipreddict["ac_pred"] = ac_pred_lst
        multipreddict["ac_prob"] = ac_prob_lst
        multipreddict["rl_pred"] = rl_pred_lst
        multipreddict["rl_prob"] = rl_prob_lst
        if parms['next_mode'] == 'next_action'or (parms['mode'] == 'batch' and parms['batchpredchoice'] in ['Prediction', 'Generative'] and parms['batch_mode'] == 'pre_prefix' and parms[
                    'variant'] in ['multi_pred', 'multi_pred_rand']):
            multipreddict["tm_pred"] = tm_pred_lst

        return multipreddict

    # ------Supporting Functions
    @staticmethod
    @st.cache(persist=True)
    def rescale(value, parms, scale_args):
        if parms['norm_method'] == 'lognorm':
            max_value = scale_args['max_value']
            min_value = scale_args['min_value']
            value = (value * (max_value - min_value)) + min_value
            value = np.expm1(value)
        elif parms['norm_method'] == 'normal':
            max_value = scale_args['max_value']
            min_value = scale_args['min_value']
            value = (value * (max_value - min_value)) + min_value
        elif parms['norm_method'] == 'standard':
            mean = scale_args['mean']
            std = scale_args['std']
            value = (value * std) + mean
        elif parms['norm_method'] == 'max':
            max_value = scale_args['max_value']
            value = np.rint(value * max_value)
        elif parms['norm_method'] is None:
            value = value
        else:
            raise ValueError(parms['norm_method'])
        return value

    # @staticmethod
    # @st.cache(persist=True)
    # def dashboard_history_label():
    #     colstm = st.columns(2)
    #     with colstm[0]:
    #         process_history_behaviour = st.empty()
    #     with colstm[1]:
    #         label_status = st.empty()
    #     return process_history_behaviour, label_status

    @staticmethod
    def dashboard_label_decider(check_ac_list, parms):
        pos_label = "### Deviant"
        neg_label = "### Regular"
        na_label = "### Not Decided"
        if len(check_ac_list) >= parms['label_check_event']:
            if parms['label_activity'] in check_ac_list:
                st.markdown(pos_label)
            elif parms['label_activity'] not in check_ac_list:
                st.markdown(neg_label)
        else:
            st.markdown(na_label)

    # def _export_results(self, output_path) -> None:
    #     # Save results
    #     pd.DataFrame(self.sim_values).to_csv(
    #         os.path.join(self.output_route, sup.file_id(prefix='SE_')),
    #         index=False)
    #     # Save logs
    #     log_test = self.log[~self.log.task.isin(['Start', 'End'])]
    #     log_test.to_csv(
    #         os.path.join(self.output_route, 'tst_'+
    #                      self.parms['model_file'].split('.')[0]+'.csv'),
    #         index=False)

class EvaluateTask():

    def evaluate(self, parms, data):
        sampler = self._get_evaluator(parms['activity'], parms['mode'])
        # print("Evaluate Data:", data)
        return sampler(data, parms)

    def _get_evaluator(self, activity, mode):
        if activity == 'predict_next' and mode == 'next':
            return self._evaluate_predict_next
        elif activity == 'predict_next' and mode == 'batch':
            return self._evaluate_predict_batch
        else:
            raise ValueError(activity)

    def _evaluate_predict_next(self, data, parms):
        evaluator = ev.Evaluator(parms['one_timestamp'], parms['variant'], parms['next_mode'], parms['mode'])

        if parms['variant'] in ['multi_pred']:
            _data = data.copy()
            _data['ac_expect'] = data.ac_expect.replace(parms['index_ac'])
            _data['rl_expect'] = data.rl_expect.replace(parms['index_rl'])
            ac_sim = evaluator.measure('accuracy', _data, 'ac')
            rl_sim = evaluator.measure('accuracy', _data, 'rl')
        else:
            ac_sim = evaluator.measure('accuracy', data, 'ac')
            rl_sim = evaluator.measure('accuracy', data, 'rl')

        mean_ac = ac_sim.accuracy.mean()
        mean_rl = rl_sim.accuracy.mean()

        st.sidebar.write("Activity Prediction Accuracy : ", round((mean_ac * 100), 2), " %")
        st.sidebar.write("Role Prediction Accuracy : ", round((mean_rl * 100), 2), " %")

        if parms['one_timestamp']:
            tm_mae = evaluator.measure('mae_next', data, 'tm')
            st.sidebar.write("Time MAE : ", round(tm_mae['mae'][0], 2))
        else:
            dur_mae = evaluator.measure('mae_next', data, 'dur')
            wait_mae = evaluator.measure('mae_next', data, 'wait')

    # def _evaluate_pred_sfx(self, data, parms):
    #
    #     exp_desc = self.clean_parameters(parms.copy())
    #     evaluator = ev.Evaluator(parms['one_timestamp'], parms['variant'], parms['next_mode'])
    #     ac_sim = evaluator.measure('similarity', data, 'ac')
    #     rl_sim = evaluator.measure('similarity', data, 'rl')
    #     mean_sim = ac_sim['mean'].mean()
    #     exp_desc = pd.DataFrame([exp_desc])
    #     exp_desc = pd.concat([exp_desc] * len(ac_sim), ignore_index=True)
    #     ac_sim = pd.concat([ac_sim, exp_desc], axis=1).to_dict('records')
    #     rl_sim = pd.concat([rl_sim, exp_desc], axis=1).to_dict('records')
    #     self.save_results(ac_sim, 'ac', parms)
    #     self.save_results(rl_sim, 'rl', parms)
    #     if parms['one_timestamp']:
    #         tm_mae = evaluator.measure('mae_suffix', data, 'tm')
    #         tm_mae = pd.concat([tm_mae, exp_desc], axis=1).to_dict('records')
    #         self.save_results(tm_mae, 'tm', parms)
    #     else:
    #         dur_mae = evaluator.measure('mae_suffix', data, 'dur')
    #         wait_mae = evaluator.measure('mae_suffix', data, 'wait')
    #         dur_mae = pd.concat([dur_mae, exp_desc], axis=1).to_dict('records')
    #         wait_mae = pd.concat([wait_mae, exp_desc], axis=1).to_dict('records')
    #         self.save_results(dur_mae, 'dur', parms)
    #         self.save_results(wait_mae, 'wait', parms)
    #     return mean_sim
    #
    # def _evaluate_predict_log(self, data, parms):
    #
    #     exp_desc = self.clean_parameters(parms.copy())
    #     evaluator = ev.Evaluator(parms['one_timestamp'], parms['variant'], parms['next_mode'])
    #     dl = evaluator.measure('dl', data)
    #     els = evaluator.measure('els', data)
    #     mean_els = els.els.mean()
    #     mae = evaluator.measure('mae_log', data)
    #     exp_desc = pd.DataFrame([exp_desc])
    #     exp_desc = pd.concat([exp_desc] * len(dl), ignore_index=True)
    #     # exp_desc = pd.concat([exp_desc]*len(els), ignore_index=True)
    #     dl = pd.concat([dl, exp_desc], axis=1).to_dict('records')
    #     els = pd.concat([els, exp_desc], axis=1).to_dict('records')
    #     mae = pd.concat([mae, exp_desc], axis=1).to_dict('records')
    #     self.save_results(dl, 'dl', parms)
    #     self.save_results(els, 'els', parms)
    #     self.save_results(mae, 'mae', parms)
    #     return mean_els


    def _evaluate_predict_batch(self, _data, parms):

        eval_data = _data.copy()
        req_columns = ['caseid', 'ac_expect', 'ac_pred', 'rl_expect', 'rl_pred', 'tm_expect', 'tm_pred',
                   'pref_size', 'run_num', 'implementation']
        eval_data = eval_data[req_columns]

        if parms['variant'] in ['multi_pred', 'multi_pred_rand']:
            eval_ac_index = {v: k for k, v in parms['index_ac'].items()}
            eval_rl_index = {v: k for k, v in parms['index_rl'].items()}

            for ix in range(len(eval_data['ac_pred'])):
                for jx in range(len(eval_data['ac_pred'][ix])):
                    # replacing the value from the parms dictionary
                    eval_data['ac_pred'][ix].append(eval_ac_index[eval_data.ac_pred[ix][jx]])
                # poping out the values from the list
                ln = int(len(eval_data['ac_pred'][ix]) / 2)
                del eval_data['ac_pred'][ix][:ln]

            for ix in range(len(eval_data['rl_pred'])):
                for jx in range(len(eval_data['rl_pred'][ix])):
                    # replacing the value from the parms dictionary
                    eval_data['rl_pred'][ix].append(eval_rl_index[eval_data.rl_pred[ix][jx]])
                # poping out the values from the list
                ln = int(len(eval_data['rl_pred'][ix]) / 2)
                del eval_data['rl_pred'][ix][:ln]

            evaluationdf_list = list()
            similarity_measure_list = list()
            for i in range(parms['multiprednum']):
                data = eval_data.copy()
                # print("Data Before")
                # print(data)
                data['ac_pred'] = eval_data['ac_pred'].str[i]
                data['rl_pred'] = eval_data['rl_pred'].str[i]
                if parms['batchpredchoice'] in ['Prediction', 'Generative'] and parms['batch_mode'] == 'pre_prefix' and parms[
                    'variant'] in ['multi_pred', 'multi_pred_rand']:
                    data['tm_pred'] = eval_data['tm_pred'].str[i]
                # print("Data After")
                # print(data)
                evaluationdf, similarity_measure = self._evaluate_predict_batch_subprocess(data, parms)

                evaluationdf.insert(0, "Pred_Number", "Prediction " + str(i+1))
                similarity_measure.insert(0, "Pred_Number", "Prediction " + str(i+1))

                evaluationdf = evaluationdf.reset_index().groupby(["Pred_Number", 'Parameter'])['Mean Value'].aggregate(
                    'first').unstack()
                evaluationdf_list.append(evaluationdf)
                similarity_measure_list.append(similarity_measure)
            # st.subheader("📊 Evaluation of " + nw(i + 1, lang="en", to="ordinal_num") + " Prediction")
            final_similarity_measure = pd.concat(similarity_measure_list) #recreating dataframe
            st.subheader("📊 Evaluation of Prediction")
            final_evaluationdf = pd.concat(evaluationdf_list)
            final_evaluationdf = final_evaluationdf[['Activity Accuracy', 'Activity Similarity', 'Role Accuracy',
                                                     'Role Similarity','Time MAE', 'Time MAE Similarity',
                                                     'Control-Flow Log Similarity', 'Event Log Similarity']]
            st.write(final_evaluationdf)
            final_evaluationdf.reset_index(level=0, inplace=True)
            final_evaluationdf_w = final_evaluationdf.to_dict('records')
            # self.save_results(final_evaluationdf_w, 'all', 'evaluation', parms)
            #--Similarity Measure
            final_similarity_measure_w = final_similarity_measure.to_dict('records')
            # self.save_results(final_similarity_measure_w, 'all', 'similarity', parms)
            # x_cols = list(set(final_similarity_measure.columns) - set(['Pred_Number', 'run_num', 'Attributes', 'implementation', 'mean']))  # to get the sequence of columns
            # y_cols = final_similarity_measure['Pred_Number'].unique().tolist()
            #--Dataframe with just activity
            final_similarity_measure_ac = final_similarity_measure.loc[final_similarity_measure['Attributes'] == 'Activity']
            final_similarity_measure_ac = final_similarity_measure_ac.reset_index(drop=True)
            final_similarity_measure_ac = final_similarity_measure_ac.drop(columns=['run_num', 'Attributes', 'implementation', 'mean'])
            final_similarity_measure_ac = final_similarity_measure_ac.set_index('Pred_Number').T.rename_axis('Event_Num').rename_axis(None, axis=1).reset_index()
            final_similarity_measure_ac.set_index('Event_Num', inplace=True)
            # final_similarity_measure_ac = final_similarity_measure_ac.drop(columns=['Varient'])

            #--Dataframe with just role
            final_similarity_measure_rl = final_similarity_measure.loc[final_similarity_measure['Attributes'] == 'Role']
            final_similarity_measure_rl = final_similarity_measure_rl.reset_index(drop=True)
            final_similarity_measure_rl = final_similarity_measure_rl.drop(columns=['run_num', 'Attributes', 'implementation', 'mean'])
            final_similarity_measure_rl = final_similarity_measure_rl.set_index('Pred_Number').T.rename_axis('Event_Num').rename_axis(None, axis=1).reset_index()
            final_similarity_measure_rl.set_index('Event_Num', inplace=True)
            # final_similarity_measure_rl = final_similarity_measure_rl.drop(columns=['Varient'])

            #--Dataframe with just time
            final_similarity_measure_tm = final_similarity_measure.loc[final_similarity_measure['Attributes'] == 'Time']
            final_similarity_measure_tm = final_similarity_measure_tm.reset_index(drop=True)
            final_similarity_measure_tm = final_similarity_measure_tm.drop(columns=['run_num', 'Attributes', 'implementation', 'mean'])
            # final_similarity_measure_tm = final_similarity_measure_tm[x_cols].mean(axis=0)
            final_similarity_measure_tm = final_similarity_measure_tm.iloc[[0]]
            final_similarity_measure_tm = final_similarity_measure_tm.set_index('Pred_Number').T.rename_axis('Event_Num').rename_axis(None, axis=1).reset_index()
            # final_similarity_measure_tm = final_similarity_measure_tm.drop(columns=['Varient'])
            final_similarity_measure_tm.set_index('Event_Num', inplace=True)
            # final_similarity_measure_tm = final_similarity_measure_tm.rename_axis('Event_Num', inplace=True)
            final_similarity_measure_tm = final_similarity_measure_tm.rename(columns={final_similarity_measure_tm.columns[0]: 'MAE'})
            # print(final_similarity_measure_tm)

            st.subheader("Time : Predictions vs Event Number")
            # st.area_chart(final_similarity_measure_tm)
            st.line_chart(final_similarity_measure_tm)

            figcols1, figcols2 = st.columns([2,2])
            with figcols1:
                #--Activity Accuracy
                fig1 = plt.figure()
                plt.plot(final_evaluationdf['Pred_Number'], final_evaluationdf['Activity Accuracy'], color='tab:cyan', marker='o')
                plt.title('Activity Accuracy Comparison', fontsize=14)
                # plt.xlabel('Predictions', fontsize=10)
                plt.ylabel('Accuracy', fontsize=10)
                st.write(fig1);

            with figcols2:
                #--Activity Similarity
                fig4 = plt.figure()
                plt.plot(final_evaluationdf['Pred_Number'], final_evaluationdf['Activity Similarity'], color='tab:cyan', marker='o')
                plt.title('Activity Similarity Comparison', fontsize=14)
                # plt.xlabel('Predictions', fontsize=10)
                plt.ylabel('Mean Value', fontsize=10)
                st.write(fig4);

            st.subheader("Activity : Predictions vs Event Number")
            # st.area_chart(final_similarity_measure_ac)
            st.bar_chart(final_similarity_measure_ac)
            # st.line_chart(final_similarity_measure_ac)
            figcols3, figcols4 = st.columns([2, 2])

            with figcols3:
                #--Role Accuracy
                fig2 = plt.figure()
                plt.plot(final_evaluationdf['Pred_Number'], final_evaluationdf['Role Accuracy'], color='royalblue', marker='o')
                plt.title('Role Accuracy Comparison', fontsize=14)
                # plt.xlabel('Predictions', fontsize=10)
                plt.ylabel('Accuracy', fontsize=10)
                st.write(fig2);


            with figcols4:
                #--Role Similarity
                fig5 = plt.figure()
                plt.plot(final_evaluationdf['Pred_Number'], final_evaluationdf['Role Similarity'], color='royalblue', marker='o')
                plt.title('Role Similarity Comparison', fontsize=14)
                # plt.xlabel('Predictions', fontsize=10)
                plt.ylabel('Mean Value', fontsize=10)
                st.write(fig5);

            st.subheader("Role : Predictions vs Event Number")
            # st.area_chart(final_similarity_measure_rl)
            st.bar_chart(final_similarity_measure_rl)
            # st.line_chart(final_similarity_measure_rl)
            figcols5, figcols6 = st.columns([2, 2])

            with figcols5:
                #--Control-Flow Log Similarity
                fig6 = plt.figure()
                plt.plot(final_evaluationdf['Pred_Number'], final_evaluationdf['Control-Flow Log Similarity'], color='dodgerblue', marker='o')
                plt.title('Control-Flow Log (Damerau-Levenshtein) Similarity Comparison', fontsize=12)
                # plt.xlabel('Predictions', fontsize=10)
                plt.ylabel('Mean Value', fontsize=10)
                st.write(fig6);

            with figcols6:
                #--Event Log Similartiy
                fig3 = plt.figure()
                plt.plot(final_evaluationdf['Pred_Number'], final_evaluationdf['Event Log Similarity'], color='dodgerblue', marker='o')
                plt.title('Event Log Similarity (Business Process Trace Distance) Comparison', fontsize=12)
                # plt.xlabel('Predictions', fontsize=10)
                plt.ylabel('Mean Value', fontsize=10)
                st.write(fig3);

            # print(final_similarity_measure_ac)

            #https://towardsdatascience.com/a-guide-to-pandas-and-matplotlib-for-data-exploration-56fad95f951c
            # colour_list = ['blue', 'red', 'green', 'pink']
            # fig7 = plt.figure()
            # # for i in range(len(y_cols)):
            # plt.plot.area(final_similarity_measure_ac['Varient'], final_similarity_measure_ac[y_cols])
            # plt.title('Activity Varient for respective Prediction', fontsize=14)
            # plt.xlabel('Variants')
            # plt.legend(y_cols, loc='upper left')
            # st.write(fig7);

        else:
            st.subheader("📊 Evaluation of Prediction")
            evaluationdf, similarity_measure = self._evaluate_predict_batch_subprocess(eval_data, parms)
            evaluationdf = evaluationdf.set_index('Parameter').T.rename_axis(None, axis=1)
            # evaluationdf.index = [""] * len(evaluationdf)
            st.write(evaluationdf)
            final_similarity_measure = similarity_measure.copy()
            similarity_measure = similarity_measure.to_dict('records')
            # self.save_results(similarity_measure, 'max', 'similarity', parms)
            #--Dataframe with just activity
            final_similarity_measure_ac = final_similarity_measure.loc[final_similarity_measure['Attributes'] == 'Activity']
            final_similarity_measure_ac = final_similarity_measure_ac.reset_index(drop=True)
            final_similarity_measure_ac = final_similarity_measure_ac.drop(columns=['run_num', 'implementation', 'mean'])
            final_similarity_measure_ac = final_similarity_measure_ac.set_index('Attributes').T.rename_axis('Event_Num').rename_axis(None, axis=1).reset_index()
            # final_similarity_measure_ac = final_similarity_measure_ac.drop(columns=['Varient'])
            final_similarity_measure_ac.set_index('Event_Num', inplace=True)
            final_similarity_measure_ac.rename(columns={'Activity': 'mean'}, inplace=True)
            #
            # #--Dataframe with just role
            final_similarity_measure_rl = final_similarity_measure.loc[final_similarity_measure['Attributes'] == 'Role']
            final_similarity_measure_rl = final_similarity_measure_rl.reset_index(drop=True)
            final_similarity_measure_rl = final_similarity_measure_rl.drop(columns=['run_num', 'implementation', 'mean'])
            final_similarity_measure_rl = final_similarity_measure_rl.set_index('Attributes').T.rename_axis('Event_Num').rename_axis(None, axis=1).reset_index()
            # final_similarity_measure_rl = final_similarity_measure_rl.drop(columns=['Varient'])
            final_similarity_measure_rl.set_index('Event_Num', inplace=True)
            final_similarity_measure_rl.rename(columns={'Role': 'mean'}, inplace=True)
            #
            #--Dataframe with just time
            final_similarity_measure_tm = final_similarity_measure.loc[final_similarity_measure['Attributes'] == 'Time']
            final_similarity_measure_tm = final_similarity_measure_tm.reset_index(drop=True)
            final_similarity_measure_tm = final_similarity_measure_tm.drop(columns=['run_num', 'implementation', 'mean'])
            final_similarity_measure_tm = final_similarity_measure_tm.iloc[[0]]
            final_similarity_measure_tm = final_similarity_measure_tm.set_index('Attributes').T.rename_axis('Event_Num').rename_axis(None, axis=1).reset_index()
            # final_similarity_measure_tm = final_similarity_measure_tm.drop(columns=['Varient'])
            final_similarity_measure_tm.set_index('Event_Num', inplace=True)
            final_similarity_measure_tm.rename(columns={'Time': 'MAE'}, inplace=True)

            st.subheader("Activity : Predictions vs Event Number")
            st.bar_chart(final_similarity_measure_ac)
            st.subheader("Role : Predictions vs Event Number")
            st.bar_chart(final_similarity_measure_rl)
            st.subheader("Time : Predictions vs Event Number")
            st.area_chart(final_similarity_measure_tm)

    def _evaluate_predict_batch_subprocess(self, data, parms):
        evaluator = ev.Evaluator(parms['one_timestamp'], parms['variant'], parms['next_mode'], parms['mode'])
        #--Accuracy Measurement
        ac_acc = evaluator.measure('accuracy', data, 'ac')
        rl_acc = evaluator.measure('accuracy', data, 'rl')

        mean_acc_ac = ac_acc.accuracy.mean()
        mean_acc_rl = rl_acc.accuracy.mean()

        if parms['one_timestamp']:
            tm_mae_acc = evaluator.measure('mae_next', data, 'tm')
        else:
            dur_mae = evaluator.measure('mae_next', data, 'dur')
            wait_mae = evaluator.measure('mae_next', data, 'wait')

        #---Similarity Measure (Control-Flow Log Similarity (CFLS) using Damerau-Levenshtein distance)
        ac_sim = evaluator.measure('similarity', data, 'ac')
        ac_sim.insert(1, "Attributes", 'Activity')
        rl_sim = evaluator.measure('similarity', data, 'rl')
        rl_sim.insert(1, "Attributes", 'Role')
        if parms['one_timestamp']:
            tm_mae = evaluator.measure('mae_suffix', data, 'tm')
            tm_mae.insert(1, "Attributes", 'Time')
            similarity_measure = pd.concat([ac_sim, rl_sim, tm_mae], ignore_index=True)
            # similarity_measure = pd.concat([similarity_measure] * max[len(ac_sim), len(rl_sim), len(tm_mae)], ignore_index=True)
            # similarity_measure.insert(1, "Attributes", ['Activity', 'Role', 'Time'])
        else:
            dur_mae = evaluator.measure('mae_suffix', data, 'dur')
            wait_mae = evaluator.measure('mae_suffix', data, 'wait')
            similarity_measure = pd.concat([ac_sim, rl_sim, dur_mae, wait_mae], ignore_index=True)
            similarity_measure.insert(0, "Attributes", ['Activity', 'Role', 'Duration', 'Wait'])
        mean_sim_ac = ac_sim['mean'].mean()
        mean_sim_rl = rl_sim['mean'].mean()
        mean_sim_tm = tm_mae['mean'].mean()

        #--Evaluate Predict Log
        preddata = self.modify_log_pred(data, parms)
        dl = evaluator.measure('dl', preddata)
        els = evaluator.measure('els', preddata)
        mean_dl = dl.dl.mean() #--Control-Flow Log Similarity
        mean_els = els.els.mean() #--Event Log Similarity

        # els = evaluator.measure('mae_log', preddata)

        evaluation = {'Parameter' : ["Activity Accuracy", "Role Accuracy", "Time MAE",
                                     "Activity Similarity", "Role Similarity", "Time MAE Similarity",
                                     "Control-Flow Log Similarity", "Event Log Similarity"],
                      'Mean Value' : [mean_acc_ac, mean_acc_rl, round(tm_mae_acc['mae'][0], 2),
                                      mean_sim_ac, mean_sim_rl, mean_sim_tm,
                                      mean_dl, mean_els]}

        evaluationdf = pd.DataFrame(evaluation)

        return evaluationdf, similarity_measure


    @staticmethod
    @st.cache(persist=True)
    def modify_log_pred(_predictions, parms):

        source_log = _predictions[['caseid', 'ac_expect', 'tm_expect', 'rl_expect', 'pref_size']]
        source_predictions = _predictions[['caseid', 'ac_pred', 'tm_pred', 'rl_pred', 'pref_size', 'run_num', 'implementation']]

        log = source_log.copy()

        log.rename(
            columns={'caseid': 'caseid', 'ac_expect': 'task',
                     'rl_expect': 'role', 'tm_expect': 'dur', 'pref_size': 'event_nr'}, inplace=True)

        predictions = source_predictions.copy()

        predictions.rename(
            columns={'caseid': 'caseid', 'ac_pred': 'task',
                     'rl_pred': 'role', 'tm_pred': 'dur', 'pref_size': 'event_nr'}, inplace=True)

        log['run_num'] = 0
        log['implementation'] = 'log'

        return log.append(predictions, ignore_index=True)


    @staticmethod
    def clean_parameters(parms):
        exp_desc = parms.copy()
        exp_desc.pop('activity', None)
        exp_desc.pop('read_options', None)
        exp_desc.pop('column_names', None)
        exp_desc.pop('one_timestamp', None)
        exp_desc.pop('reorder', None)
        exp_desc.pop('index_ac', None)
        exp_desc.pop('index_rl', None)
        exp_desc.pop('dim', None)
        exp_desc.pop('max_dur', None)
        exp_desc.pop('variants', None)
        exp_desc.pop('is_single_exec', None)
        exp_desc.pop('caseid', None)
        exp_desc.pop('start_time', None)
        exp_desc.pop('additional_columns', None)
        exp_desc.pop('mode', None)
        exp_desc.pop('variant', None)
        exp_desc.pop('next_mode', None)
        exp_desc.pop('predchoice', None)
        exp_desc.pop('scale_args', None)
        exp_desc.pop('imp', None)
        return exp_desc

    # @staticmethod
    # def save_results(measurements, feature, evaltype, parms):
    #     if measurements:
    #         if parms['is_single_exec']:
    #             output_route = os.path.join('output_files',
    #                                         parms['folder'],
    #                                         'results')
    #             model_name, _ = os.path.splitext(parms['model_file'])
    #             sup.create_csv_file_header(
    #                 measurements,
    #                 os.path.join(
    #                     output_route,
    #                     model_name + '_' + feature + '_' + evaltype + '_' + parms['mode'] + '.csv'))
    #             # print("output_route : ", output_route)
    #         else:
    #             if os.path.exists(os.path.join(
    #                     'output_files', feature + '_' + parms['activity'] + '.csv')):
    #                 sup.create_csv_file(
    #                     measurements,
    #                     os.path.join('output_files',
    #                                  feature + '_' + parms['activity'] + '.csv'),
    #                     mode='a')
    #             else:
    #                 sup.create_csv_file_header(
    #                     measurements,
    #                     os.path.join('output_files',
    #                                  feature + '_' + parms['activity'] + '.csv'))
