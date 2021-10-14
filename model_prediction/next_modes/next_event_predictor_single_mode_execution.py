# -*- coding: utf-8 -*-

import numpy as np
import streamlit as st


from support_modules import support as sup

class NextEventPredictor():

    def __init__(self):
        """constructor"""
        self.model = None
        self.spl = dict()
        self.imp = 'arg_max'
        #-----------------------------------------

    #parms : parameters
    #model : .h5 file
    #spl : prefix or the event log
    #imp : Value is 1 by default, basically signifies how many times the prediction has to be made on same event
    #vectorizer : 'basic' value by default


    def predict(self, params, model, spl, imp, vectorizer):
        self.model = model
        #print("spl :", spl)
        self.spl = spl
        self.imp = imp
        if params['mode'] == 'next':
            fltr_idx = params['nextcaseid_attr']["filter_index"]
            # spl_df_prefx = pd.DataFrame(self.spl['prefixes'])[fltr_idx:]
            # spl_df_next = pd.DataFrame(self.spl['next_evt'])[fltr_idx:]
        self.nx = params['multiprednum']
        predictor = self._get_predictor(params['model_type'], params['mode'], params['next_mode'])
        sup.print_performed_task('Predicting next events')

        return predictor(params, vectorizer)

    def _get_predictor(self, model_type, mode, next_mode):
        # OJO: This is an extension point just incase
        # a different predictor being neccesary
        if mode == 'next':
            if next_mode == 'history_with_next':
                return self._predict_next_event_shared_cat_next  # Predicts what would come next and after

    def _predict_next_event_shared_cat_next(self, parameters, vectorizer):
        """Generate business process suffixes using a keras trained model.
        Args:
            model (keras model): keras trained model.
            prefixes (list): list of prefixes.
            ac_index (dict): index of activities.
            rl_index (dict): index of roles.
            imp (str): method of next event selection.
        """
        # Generation of predictions
        pred_fltr_idx = parameters['nextcaseid_attr']["filter_index"] + 1

        #prediction of prediction
        if 'multi_pred_ss' not in st.session_state:
            st.session_state['multi_pred_ss'] = dict()
            for _zx in range(parameters['multiprednum']):
                st.session_state['multi_pred_ss']["ss_multipredict{0}".format(_zx + 1)] = {'ac_pred': [], 'ac_prob': [],
                                                                                          'rl_pred': [], 'rl_prob': [],
                                                                                          'tm_pred': []}

        # --Time
        if 'pos_tm_ss' in st.session_state:

            serie_predict_tm = [np.array(st.session_state['pos_tm_ss'][:idx])
                                for idx in range(1, pred_fltr_idx + 1)]  # range starts with 1 to avoid start

            y_serie_predict_tm = [x[-1] for x in
                                  serie_predict_tm]  # selecting the last value from each list of list

            # print("y-serie time:", y_serie_predict_tm)

        elif 'pos_tm_ss' not in st.session_state:
            st.session_state['pos_tm_ss'] = [[0]]

        if 'initial_prediction' in st.session_state:

            for _ih in range(parameters['multiprednum']):

                if "ss_initpredict"+str(_ih+1) in st.session_state['initial_prediction']:

                    serie_predict_ac = [st.session_state['initial_prediction']["ss_initpredict"+str(_ih+1)]["pos_ac_ss"][:idx]
                                        for idx in range(1, pred_fltr_idx + 1)]  # range starts with 1 to avoid start

                    y_serie_predict_ac = [x[-1] for x in
                                          serie_predict_ac]  # selecting the last value from each list of list

                    serie_predict_rl = [st.session_state['initial_prediction']["ss_initpredict"+str(_ih+1)]["pos_rl_ss"][:idx]
                                        for idx in range(1, pred_fltr_idx + 1)]  # range starts with 1 to avoid start

                    y_serie_predict_rl = [x[-1] for x in
                                          serie_predict_rl]  # selecting the last value from each list of list

                #----Check the Vector length is same or not
                    if (len(self.spl['prefixes']['activities'][:pred_fltr_idx]) == len(serie_predict_ac)) and (
                            len(self.spl['prefixes']['roles'][:pred_fltr_idx]) == len(serie_predict_rl)) and (
                            len(self.spl['prefixes']['times'][:pred_fltr_idx]) == len(serie_predict_tm)) and (
                            'multi_pred_ss' in st.session_state):

                        # print("--------------Input to Prediction",(_ih+ 1), "--------------------")
                        # print("Activity Prefixes :", serie_predict_ac)
                        # print("Role Prefixes :", serie_predict_rl)
                        # print("Time Prefixes :", serie_predict_tm)

                        self._predict_next_event_shared_cat_pred(parameters, vectorizer, serie_predict_ac, serie_predict_rl, serie_predict_tm, _ih)

            # first prediction
        if 'initial_prediction' not in st.session_state:
            st.session_state['initial_prediction'] = dict()
            for _lx in range(parameters['multiprednum']):
                st.session_state['initial_prediction']["ss_initpredict{0}".format(_lx + 1)] = {'pos_ac_ss': [0],
                                                                                               'pos_rl_ss': [0]}

        results = list()
        # print("Initial Session STate : ", st.session_state)
        for i, _ in enumerate(self.spl['prefixes']['activities'][:pred_fltr_idx]):

            x_ac_ngram = (np.append(
                    np.zeros(parameters['dim']['time_dim']),
                    np.array(self.spl['prefixes']['activities'][:pred_fltr_idx][i]),
                    axis=0)[-parameters['dim']['time_dim']:]
                .reshape((1, parameters['dim']['time_dim'])))

            x_rl_ngram = (np.append(
                    np.zeros(parameters['dim']['time_dim']),
                    np.array(self.spl['prefixes']['roles'][:pred_fltr_idx][i]),
                    axis=0)[-parameters['dim']['time_dim']:]
                .reshape((1, parameters['dim']['time_dim'])))

            # times input shape(1,5,1)
            times_attr_num = (self.spl['prefixes']['times'][:pred_fltr_idx][i].shape[1])
            x_t_ngram = np.array(
                [np.append(np.zeros(
                    (parameters['dim']['time_dim'], times_attr_num)),
                    self.spl['prefixes']['times'][:pred_fltr_idx][i], axis=0)
                    [-parameters['dim']['time_dim']:]
                    .reshape((parameters['dim']['time_dim'], times_attr_num))])
            # add intercase features if necessary
            if vectorizer in ['basic']:
                inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram]
            elif vectorizer in ['inter']:
                inter_attr_num = (self.spl['prefixes']['inter_attr'][:pred_fltr_idx][i].shape[1])
                x_inter_ngram = np.array(
                    [np.append(np.zeros((
                        parameters['dim']['time_dim'], inter_attr_num)),
                        self.spl['prefixes']['inter_attr'][:pred_fltr_idx][i], axis=0)
                        [-parameters['dim']['time_dim']:]
                        .reshape((parameters['dim']['time_dim'], inter_attr_num))])
                inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram, x_inter_ngram]
            # predict
            preds = self.model.predict(inputs)

            if self.imp == 'random_choice':
                # Use this to get a random choice following as PDF
                pos = np.random.choice(np.arange(0, len(preds[0][0])),
                                       p=preds[0][0])
                pos_prob = preds[0][0][pos]
                pos1 = np.random.choice(np.arange(0, len(preds[1][0])),
                                        p=preds[1][0])
                pos1_prob = preds[1][0][pos1]


            elif self.imp == 'arg_max':
                # Use this to get the max prediction

                pos = np.argmax(preds[0][0])
                pos_prob = preds[0][0][pos]

                pos1 = np.argmax(preds[1][0])
                pos1_prob = preds[1][0][pos1]

            elif self.imp == 'multi_pred':

                #changing array to numpy
                acx = np.array(preds[0][0])
                rlx = np.array(preds[1][0])

                pos = (-acx).argsort()[:self.nx].tolist()
                pos1 = (-rlx).argsort()[:self.nx].tolist()

                pos_prob = []
                pos1_prob = []

                for ix in range(len(pos)):
                    # probability of activity
                    pos_prob.append(acx[pos[ix]])
                for jx in range(len(pos1)):
                    # probability of role
                    pos1_prob.append(rlx[pos1[jx]])

            # save results
            predictions = [pos, pos1, preds[2][0][0], pos_prob, pos1_prob]

            if i == pred_fltr_idx - 1:
                for _ik in range(parameters['multiprednum']):
                    if  parameters['multiprednum'] > 1:
                        # Selecting the Choice of Prediction for each run, int(parameters['predchoice'][-1])
                        # takes the last value of the choice and coverts it into integer, then to match with
                        # index -1 is subtracted, similarly we need only one value so, value after colon to
                        # match the position
                        #-Activity
                        st.session_state['initial_prediction']['ss_initpredict' + str(_ik + 1)]['pos_ac_ss'].extend(
                            pos[_ik:_ik + 1])

                        #-Role
                        st.session_state['initial_prediction']['ss_initpredict' + str(_ik + 1)]['pos_rl_ss'].extend(
                            pos1[_ik:_ik + 1])

                    elif parameters['multiprednum'] == 1:
                        #When the prediction is selected as the Max probability)
                        st.session_state['initial_prediction']['ss_initpredict' + str(_ik + 1)]['pos_ac_ss'].extend(
                            [pos])
                        st.session_state['initial_prediction']['ss_initpredict' + str(_ik + 1)]['pos_rl_ss'].extend(
                            [pos1])

                # print("State of Dictionary for the iteration", _ik, " : ", st.session_state['initial_prediction'])
                # -Time
                st.session_state['pos_tm_ss'].extend([[preds[2][0][0]]])

            if not parameters['one_timestamp']:
                predictions.extend([preds[2][0][1]])
            results.append(self._create_result_record_next(i, self.spl, predictions, parameters))
            print("Execution Mode Predictions")
            print(results)
        sup.print_done_task()
        return results

    def _create_result_record_next(self, index, spl, preds, parms):
        _fltr_idx = parms['nextcaseid_attr']["filter_index"] + 1
        record = dict()
        record['ac_prefix'] = spl['prefixes']['activities'][:_fltr_idx][index]
        record['ac_expect'] = spl['next_evt']['activities'][:_fltr_idx][index]
        record['ac_pred'] = preds[0]
        record['ac_prob'] = preds[3]
        record['rl_prefix'] = spl['prefixes']['roles'][:_fltr_idx][index]
        record['rl_expect'] = spl['next_evt']['roles'][:_fltr_idx][index]
        record['rl_pred'] = preds[1]
        record['rl_prob'] = preds[4]



        if parms['one_timestamp']:

            record['tm_prefix'] = [self.rescale(
               x, parms, parms['scale_args'])
               for x in spl['prefixes']['times'][:_fltr_idx][index]]
            record['tm_expect'] = self.rescale(
                spl['next_evt']['times'][:_fltr_idx][index][0],
                parms, parms['scale_args'])
            record['tm_pred'] = self.rescale(
                preds[2], parms, parms['scale_args'])

        else:
            # Duration
            record['dur_prefix'] = [self.rescale(
                x[0], parms, parms['scale_args']['dur'])
                for x in spl['prefixes']['times'][:_fltr_idx][index]]
            record['dur_expect'] = self.rescale(
                spl['next_evt']['times'][:_fltr_idx][index][0], parms,
                parms['scale_args']['dur'])
            record['dur_pred'] = self.rescale(
                preds[2], parms, parms['scale_args']['dur'])
            # Waiting
            record['wait_prefix'] = [self.rescale(
                x[1], parms, parms['scale_args']['wait'])
                for x in spl['prefixes']['times'][:_fltr_idx][index]]
            record['wait_expect'] = self.rescale(
                spl['next_evt']['times'][_fltr_idx][index][1], parms,
                parms['scale_args']['wait'])
            record['wait_pred'] = self.rescale(
                preds[3], parms, parms['scale_args']['wait'])
        return record


    def _predict_next_event_shared_cat_pred(self, parameters, vectorizer, serie_predict_ac, serie_predict_rl, serie_predict_tm, index):
        """Generate business process suffixes using a keras trained model.
        Args:
            model (keras model): keras trained model.
            prefixes (list): list of prefixes.
            ac_index (dict): index of activities.
            rl_index (dict): index of roles.
            imp (str): method of next event selection.
        """
        # print("Starting of Multi Dimention Prediction : ", index + 1)
        # Generation of predictions
        pred_fltr_idx = parameters['nextcaseid_attr']["filter_index"] + 1

        for i, _ in enumerate(serie_predict_ac):

            x_ac_ngram = (np.append(
                    np.zeros(parameters['dim']['time_dim']),
                    np.array(serie_predict_ac[i]),
                    axis=0)[-parameters['dim']['time_dim']:]
                .reshape((1, parameters['dim']['time_dim'])))
            #----------------------------------------------------------------------------------
            x_rl_ngram = (np.append(
                    np.zeros(parameters['dim']['time_dim']),
                    np.array(serie_predict_rl[i]),
                    axis=0)[-parameters['dim']['time_dim']:]
                .reshape((1, parameters['dim']['time_dim'])))
            # ----------------------------------------------------------------------------------
            # ----------------------------------------------------------------------------------
            # times input shape(1,5,1)
            times_attr_num = (serie_predict_tm[i].shape[1])
            x_t_ngram = np.array(
                [np.append(np.zeros(
                    (parameters['dim']['time_dim'], times_attr_num)),
                    serie_predict_tm[i], axis=0)
                    [-parameters['dim']['time_dim']:]
                    .reshape((parameters['dim']['time_dim'], times_attr_num))])
            # add intercase features if necessary
            # ----------------------------------------------------------------------------------
            if vectorizer in ['basic']:
                inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram]
            elif vectorizer in ['inter']:
                inter_attr_num = (self.spl['prefixes']['inter_attr'][:pred_fltr_idx][i].shape[1])
                x_inter_ngram = np.array(
                    [np.append(np.zeros((
                        parameters['dim']['time_dim'], inter_attr_num)),
                        self.spl['prefixes']['inter_attr'][:pred_fltr_idx][i], axis=0)
                        [-parameters['dim']['time_dim']:]
                        .reshape((parameters['dim']['time_dim'], inter_attr_num))])
                inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram, x_inter_ngram]
            # predict
            preds = self.model.predict(inputs)

            #---Selecting the Arg Max
            pos = np.argmax(preds[0][0])
            pos_prob = preds[0][0][pos]

            pos1 = np.argmax(preds[1][0])
            pos1_prob = preds[1][0][pos1]

            if i == pred_fltr_idx-1:

                st.session_state['multi_pred_ss']["ss_multipredict" + str(index + 1)]['ac_pred'].extend([pos])
                st.session_state['multi_pred_ss']["ss_multipredict" + str(index + 1)]['ac_prob'].extend([pos_prob])

                st.session_state['multi_pred_ss']["ss_multipredict" + str(index + 1)]['rl_pred'].extend([pos1])
                st.session_state['multi_pred_ss']["ss_multipredict" + str(index + 1)]['rl_prob'].extend([pos1_prob])

                st.session_state['multi_pred_ss']["ss_multipredict" + str(index + 1)]['tm_pred'].extend(
                    [[preds[2][0][0]]])

    @staticmethod
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
