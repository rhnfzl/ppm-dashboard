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
            if next_mode == 'what_if':
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
        # print("In the New What-if")
        # print("Index Value :", pred_fltr_idx)

        # print("State of the Sate Space Initially: ", st.session_state)
        # print("-------------------------------------------------------")

        if 'multi_pred_ss' not in st.session_state:
            st.session_state['multi_pred_ss'] = dict()
            for _zx in range(parameters['multiprednum']):
                st.session_state['multi_pred_ss']["ss_multipredict{0}".format(_zx + 1)] = {}
                for _ux in range(parameters['multiprednum']):
                    st.session_state['multi_pred_ss']["ss_multipredict" + str(_zx + 1)][
                        "multiverse_predict{0}".format(_ux + 1)] = {'ac_pred': [], 'ac_prob': [], 'rl_pred': [],
                                                                    'rl_prob': [], 'tm_pred': []}

        # --History of choice
        if 'prediction_choice_idx' not in st.session_state:

            st.session_state['prediction_choice_idx'] = []

        if 'prediction_choice_name' not in st.session_state:

            st.session_state['prediction_choice_name'] = []

        if 'history_of_choice' not in st.session_state:

            st.session_state['history_of_choice'] = dict()

            st.session_state['history_of_choice'] = {'hist_ac_prefix': [],
                                                     'hist_rl_prefix': [],
                                                     'hist_tm_prefix': [],
                                                     'hist_pred_prefix': []}

        #-- Choice of prediction
        # SME Mode assumes that Functional Knowledge person is taking action
        if parameters['predchoice'] == 'SME':
            _selectpredidx = 0
        else :
            # int(parameters['predchoice'][-1]) takes the last value of the choice and coverts it into integer
            # i.e Prediction 1 to 1
            _selectpredidx = int(parameters['predchoice'][-1])

        # print("State of the Sate Space Second: ", st.session_state)
        # print("-------------------------------------------------------")

        if 'initial_prediction' in st.session_state:

            st.session_state['prediction_choice_idx'].extend([_selectpredidx])
            st.session_state['prediction_choice_name'].extend([parameters['predchoice']])

            #-- Multiverse logic
            # for _ih in range(parameters['multiprednum']):
            #
            #     if "ss_initpredict"+str(_ih+1) in st.session_state['initial_prediction']:
            #
            #         _serie_predict_ac = [st.session_state['initial_prediction']["ss_initpredict"+str(_ih+1)]["pos_ac_ss"][:idx]
            #                             for idx in range(1, pred_fltr_idx + 1)]  # range starts with 1 to avoid start
            #         _serie_predict_rl = [st.session_state['initial_prediction']["ss_initpredict"+str(_ih+1)]["pos_rl_ss"][:idx]
            #                             for idx in range(1, pred_fltr_idx + 1)]  # range starts with 1 to avoid start
            #         _serie_predict_tm = [np.array(st.session_state['initial_prediction']["ss_initpredict"+str(_ih+1)]['pos_tm_ss'][:idx])
            #                             for idx in range(1, pred_fltr_idx + 1)]  # range starts with 1 to avoid start
            #
            #     #----Check the Vector length is same or not
            #         if (len(self.spl['prefixes']['activities'][:pred_fltr_idx]) == len(_serie_predict_ac)) and (
            #                 len(self.spl['prefixes']['roles'][:pred_fltr_idx]) == len(_serie_predict_rl)) and (
            #                 len(self.spl['prefixes']['times'][:pred_fltr_idx]) == len(_serie_predict_tm)) and (
            #                 'multi_pred_ss' in st.session_state):
            #
            #             # print("--------------Input to Multiverse Prediction",(_ih+ 1), "--------------------")
            #             # print("Activity Prefixes :", _serie_predict_ac)
            #             # print("Role Prefixes :", _serie_predict_rl)
            #             # print("Time Prefixes :", _serie_predict_tm)
            #
            #             self._predict_next_event_shared_cat_pred(parameters, vectorizer, _serie_predict_ac, _serie_predict_rl, _serie_predict_tm, _ih) #Control the future pred

            if parameters['predchoice'] != 'SME':

                if "ss_initpredict" + str(_selectpredidx) in st.session_state['initial_prediction']:

                    #-- Saving the order of prediction choice

                    #--- Activity
                    serie_predict_ac = [st.session_state['initial_prediction']["ss_initpredict"+str(_selectpredidx)]["pos_ac_ss"][:idx]
                                        for idx in range(1, pred_fltr_idx + 1)]  # range starts with 1 to avoid start

                    y_serie_predict_ac = [x[-1] for x in
                                          serie_predict_ac]  # selecting the last value from each list of list

                    # --- Role
                    serie_predict_rl = [st.session_state['initial_prediction']["ss_initpredict"+str(_selectpredidx)]["pos_rl_ss"][:idx]
                                        for idx in range(1, pred_fltr_idx + 1)]  # range starts with 1 to avoid start

                    y_serie_predict_rl = [x[-1] for x in
                                          serie_predict_rl]  # selecting the last value from each list of list

                    # --- Time
                    serie_predict_tm = [np.array(st.session_state['initial_prediction']["ss_initpredict"+str(_selectpredidx)]['pos_tm_ss'][:idx])
                                        for idx in range(1, pred_fltr_idx + 1)]  # range starts with 1 to avoid start

                    y_serie_predict_tm = [x[-1] for x in
                                          serie_predict_tm]  # selecting the last value from each list of list

                #----Check the Vector length is same or not
                    if (len(self.spl['prefixes']['activities'][:pred_fltr_idx]) == len(serie_predict_ac)) and (
                            len(self.spl['prefixes']['roles'][:pred_fltr_idx]) == len(serie_predict_rl)) and (
                            len(self.spl['prefixes']['times'][:pred_fltr_idx]) == len(serie_predict_tm)):

                        # -Saving Histroy of the prediction
                        if 'history_of_choice' in st.session_state:
                            # print("Prediction Choice : ", parameters['predchoice'])
                            st.session_state['history_of_choice']['hist_ac_prefix'].extend([y_serie_predict_ac[-1]])
                            st.session_state['history_of_choice']['hist_rl_prefix'].extend([y_serie_predict_rl[-1]])
                            st.session_state['history_of_choice']['hist_tm_prefix'].extend([y_serie_predict_tm[-1].tolist()])
                            st.session_state['history_of_choice']['hist_pred_prefix'].extend([parameters['predchoice']])

                            self.spl['prefixes']['roles'][:pred_fltr_idx] = serie_predict_ac
                            self.spl['prefixes']['roles'][:pred_fltr_idx] = serie_predict_rl
                            self.spl['prefixes']['times'][:pred_fltr_idx] = serie_predict_tm

            elif parameters['predchoice'] == 'SME':
                y_serie_predict_ac = [x[-1] for x in
                                      self.spl['prefixes']['activities'][:pred_fltr_idx]]  # selecting the last value from each list of list
                y_serie_predict_rl = [x[-1] for x in
                                      self.spl['prefixes']['roles'][:pred_fltr_idx]]  # selecting the last value from each list of list
                y_serie_predict_tm = [x[-1] for x in
                                      self.spl['prefixes']['times'][:pred_fltr_idx]]  # selecting the last value from each list of list

                if 'history_of_choice' in st.session_state:
                    print("Prediction Choice : ", parameters['predchoice'])
                    st.session_state['history_of_choice']['hist_ac_prefix'].extend([y_serie_predict_ac[-1]])
                    st.session_state['history_of_choice']['hist_rl_prefix'].extend([y_serie_predict_rl[-1]])
                    st.session_state['history_of_choice']['hist_tm_prefix'].extend([y_serie_predict_tm[-1].tolist()])
                    st.session_state['history_of_choice']['hist_pred_prefix'].extend([parameters['predchoice']])

            # print("--------------Input to Prediction", "--------------------")
            # print("Activity Prefixes :", y_serie_predict_ac)
            # print("Role Prefixes :", y_serie_predict_rl)
            # print("Time Prefixes :", y_serie_predict_tm[-1].tolist())

        # Creating Dictionary
        if 'initial_prediction' not in st.session_state:
            st.session_state['initial_prediction'] = dict()
            for _lx in range(parameters['multiprednum']):
                st.session_state['initial_prediction']["ss_initpredict{0}".format(_lx+1)] = {'pos_ac_ss': [0],
                                                                                            'pos_rl_ss': [0],
                                                                                            'pos_tm_ss': [[0]]}


        #print("Prediction Activity :", self.spl['prefixes']['activities'][pred_fltr_idx], type(self.spl['prefixes']['activities']))
        results = list()
        for i, _ in enumerate(self.spl['prefixes']['activities'][:pred_fltr_idx]):

            # print("First Activity Input : ", self.spl['prefixes']['activities'][:pred_fltr_idx])
            # print("First Role Input : ", self.spl['prefixes']['roles'][:pred_fltr_idx])
            # print("First Time Input : ", self.spl['prefixes']['times'][:pred_fltr_idx])

            x_ac_ngram = (np.append(
                    np.zeros(parameters['dim']['time_dim']),
                    np.array(self.spl['prefixes']['activities'][:pred_fltr_idx][i]),
                    axis=0)[-parameters['dim']['time_dim']:]
                .reshape((1, parameters['dim']['time_dim'])))
            #print("x_ac_ngram :", x_ac_ngram)
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

            #SME Mode assumes that Functional Knowledge person is taking action
            if i == pred_fltr_idx-1:
                for _ik in range(parameters['multiprednum']):
                #-Time
                # st.session_state['pos_tm_ss'].extend([[preds[2][0][0]]])
                    if  parameters['multiprednum'] > 1:
                        # -Activity
                        st.session_state['initial_prediction']['ss_initpredict' + str(_ik + 1)]['pos_ac_ss'].extend(
                            pos[_ik:_ik + 1])

                        # -Role
                        st.session_state['initial_prediction']['ss_initpredict' + str(_ik + 1)]['pos_rl_ss'].extend(
                            pos1[_ik:_ik + 1])

                        # st.session_state['pos_tm_ss'].extend([[preds[2][0][0]]])


                    elif parameters['multiprednum'] == 1:
                        #When the prediction is selected as the Max probability
                        st.session_state['initial_prediction']['ss_initpredict' + str(_ik + 1)]['pos_ac_ss'].extend(
                            [pos])
                        st.session_state['initial_prediction']['ss_initpredict' + str(_ik + 1)]['pos_rl_ss'].extend(
                            [pos1])

                     # -Time
                    st.session_state['initial_prediction']['ss_initpredict' + str(_ik + 1)]['pos_tm_ss'].extend(
                        [[preds[2][0][0]]])

                #--Multiverse
                for _ih in range(parameters['multiprednum']):

                    if "ss_initpredict" + str(_ih + 1) in st.session_state['initial_prediction']:
                        # print("Inital Prediction : ", st.session_state['initial_prediction'])
                        # print("pred_fltr_idx : ", pred_fltr_idx)

                        _serie_predict_ac = [
                            st.session_state['initial_prediction']["ss_initpredict" + str(_ih + 1)]["pos_ac_ss"][:idx]
                            for idx in range(1, pred_fltr_idx + 2)]  # range starts with 1 to avoid start
                        _serie_predict_rl = [
                            st.session_state['initial_prediction']["ss_initpredict" + str(_ih + 1)]["pos_rl_ss"][:idx]
                            for idx in range(1, pred_fltr_idx + 2)] # range starts with 1 to avoid start
                        _serie_predict_tm = [np.array(
                            st.session_state['initial_prediction']["ss_initpredict" + str(_ih + 1)]['pos_tm_ss'][:idx])
                                             for idx in
                                             range(1, pred_fltr_idx + 2)] # range starts with 1 to avoid start

                        # _test_serie_predict_ac = [
                        #     st.session_state['initial_prediction']["ss_initpredict" + str(_ih + 1)]["pos_ac_ss"][:idx]
                        #     for idx in range(1, pred_fltr_idx + 2)]  # range starts with 1 to avoid start
                        #
                        # print("Test Series : ", _test_serie_predict_ac)
                        # ----Check the Vector length is same or not
                        # if (len(self.spl['prefixes']['activities'][:pred_fltr_idx]) == len(_serie_predict_ac)) and (
                        #         len(self.spl['prefixes']['roles'][:pred_fltr_idx]) == len(_serie_predict_rl)) and (
                        #         len(self.spl['prefixes']['times'][:pred_fltr_idx]) == len(_serie_predict_tm)) and (
                        #         'multi_pred_ss' in st.session_state):
                        if 'multi_pred_ss' in st.session_state:
                            # print("--------------Input to Multiverse Prediction",(_ih+ 1), "--------------------")
                            # print("Activity Prefixes :", _serie_predict_ac)
                            # print("Role Prefixes :", _serie_predict_rl)
                            # print("Time Prefixes :", _serie_predict_tm)

                            self._predict_next_event_shared_cat_pred(parameters, vectorizer, _serie_predict_ac,
                                                                     _serie_predict_rl, _serie_predict_tm,
                                                                     _ih)  # Control the future pred

            if not parameters['one_timestamp']:
                predictions.extend([preds[2][0][1]])
            results.append(self._create_result_record_next(i, self.spl, predictions, parameters))
        sup.print_done_task()
        return results

    def _create_result_record_next(self, index, spl, preds, parms):
        _fltr_idx = parms['nextcaseid_attr']["filter_index"] + 1
        record = dict()
        #print("Preds under result :", preds)
        #record['caseid'] = parms['caseid'][_fltr_idx][index]
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
        # print("Multi Activity Input : ", serie_predict_ac)
        # print("Multi Role Input : ", serie_predict_rl)
        # print("Multi Time Input : ", serie_predict_tm)
        # Generation of predictions
        pred_fltr_idx = parameters['nextcaseid_attr']["filter_index"] + 1
        # print("Index : ", pred_fltr_idx)

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
                inter_attr_num = (self.spl['prefixes']['inter_attr'][:pred_fltr_idx + i][i].shape[1])
                x_inter_ngram = np.array(
                    [np.append(np.zeros((
                        parameters['dim']['time_dim'], inter_attr_num)),
                        self.spl['prefixes']['inter_attr'][:pred_fltr_idx + i][i], axis=0)
                        [-parameters['dim']['time_dim']:]
                        .reshape((parameters['dim']['time_dim'], inter_attr_num))])
                inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram, x_inter_ngram]
            # predict
            preds = self.model.predict(inputs)

            if self.imp == 'arg_max':
                # Use this to get the max prediction

                pos = np.argmax(preds[0][0])
                pos_prob = preds[0][0][pos]

                pos1 = np.argmax(preds[1][0])
                pos1_prob = preds[1][0][pos1]

            elif self.imp == 'multi_pred':

                #---Selecting the multipred
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



            if i == pred_fltr_idx-1:

                for _iz in range(parameters['multiprednum']):
                    if parameters['multiprednum'] > 1:

                        st.session_state['multi_pred_ss']["ss_multipredict" + str(index + 1)]["multiverse_predict" + str(_iz + 1)]['ac_pred'].extend(pos[_iz:_iz + 1])
                        st.session_state['multi_pred_ss']["ss_multipredict" + str(index + 1)]["multiverse_predict" + str(_iz + 1)]['ac_prob'].extend(pos_prob[_iz:_iz + 1])

                        st.session_state['multi_pred_ss']["ss_multipredict" + str(index + 1)]["multiverse_predict" + str(_iz + 1)]['rl_pred'].extend(pos1[_iz:_iz + 1])
                        st.session_state['multi_pred_ss']["ss_multipredict" + str(index + 1)]["multiverse_predict" + str(_iz + 1)]['rl_prob'].extend(pos1_prob[_iz:_iz + 1])

                        st.session_state['multi_pred_ss']["ss_multipredict" + str(index + 1)]["multiverse_predict" + str(_iz + 1)]['tm_pred'].extend([[preds[2][0][0]]])

                    elif parameters['multiprednum'] == 1:

                        st.session_state['multi_pred_ss']["ss_multipredict" + str(index + 1)]["multiverse_predict" + str(_iz + 1)]['ac_pred'].extend([pos])
                        st.session_state['multi_pred_ss']["ss_multipredict" + str(index + 1)]["multiverse_predict" + str(_iz + 1)]['ac_prob'].extend([pos_prob])

                        st.session_state['multi_pred_ss']["ss_multipredict" + str(index + 1)]["multiverse_predict" + str(_iz + 1)]['rl_pred'].extend([pos1])
                        st.session_state['multi_pred_ss']["ss_multipredict" + str(index + 1)]["multiverse_predict" + str(_iz + 1)]['rl_prob'].extend([pos1_prob])

                        st.session_state['multi_pred_ss']["ss_multipredict" + str(index + 1)]["multiverse_predict" + str(_iz + 1)]['tm_pred'].extend([[preds[2][0][0]]])
                # print("Multiverse Predict : ", st.session_state['multi_pred_ss'])



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
