# -*- coding: utf-8 -*-
import numpy as np

from datetime import timedelta
from support_modules import support as sup


class NextEventPredictor():

    def __init__(self):
        """constructor"""
        self.model = None
        self.spl = dict()
        self.imp = 'arg_max'
        #self.nx = 2

    def predict(self, params, model, spl, imp, vectorizer):
        self.model = model
        #print("spl :", spl)
        self.spl = spl
        # print("What is Imp :", imp)
        self.imp = imp
        # if params['mode'] == 'next':
        #     fltr_idx = params['nextcaseid_attr']["filter_index"]
        #     spl_df_prefx = pd.DataFrame(self.spl['prefixes'])[fltr_idx:]
        #     spl_df_next = pd.DataFrame(self.spl['next_evt'])[fltr_idx:]
        #     st.subheader("Prefixes")
        #     st.table(spl_df_prefx)
        #     st.subheader("Next Event")
        #     st.table(spl_df_next)
            #print("spl :", spl_df)
        print("Batch Prefix Mode")
        print("Variant : ", self.imp)
        if params['variant'] in ['multi_pred', 'multi_pred_rand']:
            self.nx = params['multiprednum']
        predictor = self._get_predictor(params['model_type'], params['mode'], params['next_mode'])
        sup.print_performed_task('Predicting next events')

        return predictor(params, vectorizer)

    def _get_predictor(self, model_type, mode, next_mode):
        # OJO: This is an extension point just incase
        # a different predictor being neccesary
        if mode == 'batch':
            return self._predict_next_event_shared_cat_batch

    def _predict_next_event_shared_cat_batch(self, parameters, vectorizer):
        """Generate business process suffixes using a keras trained model.
        Args:
            model (keras model): keras trained model.
            prefixes (list): list of prefixes.
            ac_index (dict): index of activities.
            rl_index (dict): index of roles.
            imp (str): method of next event selection.
        """
        results = list()
        #-----------------SME Mode----------------------------------------
        if parameters['batchpredchoice'] == 'SME':
            for i, _ in enumerate(self.spl['prefixes']['activities']):
                # Activities and roles input shape(1,5)
                x_ac_ngram = (np.append(
                        np.zeros(parameters['dim']['time_dim']),
                        np.array(self.spl['prefixes']['activities'][i]),
                        axis=0)[-parameters['dim']['time_dim']:]
                    .reshape((1, parameters['dim']['time_dim'])))

                x_rl_ngram = (np.append(
                        np.zeros(parameters['dim']['time_dim']),
                        np.array(self.spl['prefixes']['roles'][i]),
                        axis=0)[-parameters['dim']['time_dim']:]
                    .reshape((1, parameters['dim']['time_dim'])))

                # times input shape(1,5,1)
                times_attr_num = (self.spl['prefixes']['times'][i].shape[1])
                x_t_ngram = np.array(
                    [np.append(np.zeros(
                        (parameters['dim']['time_dim'], times_attr_num)),
                        self.spl['prefixes']['times'][i], axis=0)
                        [-parameters['dim']['time_dim']:]
                        .reshape((parameters['dim']['time_dim'], times_attr_num))]
                    )

                # intercase features if necessary
                if vectorizer in ['basic']:
                    inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram]
                elif vectorizer in ['inter']:
                    inter_attr_num = (self.spl['prefixes']['inter_attr'][i].shape[1])
                    x_inter_ngram = np.array(
                        [np.append(np.zeros((
                            parameters['dim']['time_dim'], inter_attr_num)),
                            self.spl['prefixes']['inter_attr'][i], axis=0)
                            [-parameters['dim']['time_dim']:]
                            .reshape((parameters['dim']['time_dim'], inter_attr_num))]
                        )
                    inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram, x_inter_ngram]

                pref_size = len(self.spl['prefixes']['activities'][i])
                #print("input_time : ", x_t_ngram)
                # predict
                preds = self.model.predict(inputs)
                #print("Predictions :", preds)
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

                    # print("Executing Max Random")

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

                elif self.imp == 'multi_pred_rand':

                    # print("Executing Multi Random")

                    acx = np.array(preds[0][0])
                    rlx = np.array(preds[1][0])

                    pos = np.random.choice(np.arange(0, len(preds[0][0])), self.nx, replace=False,
                                           p=preds[0][0]).tolist()

                    pos1 = np.random.choice(np.arange(0, len(preds[1][0])), self.nx, replace=False,
                                            p=preds[1][0]).tolist()

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

                if not parameters['one_timestamp']:
                    predictions.extend([preds[2][0][1]])
                results.append(self.create_result_record_batch(i, self.spl, predictions, parameters, pref_size, results))
            sup.print_done_task()

        # -----------------Prediction Mode----------------------------------------
        elif parameters['batchpredchoice'] == 'Prediction':

            # self._predict_next_event_shared_cat_batch_prediction(self, parameters, results,  self.spl['prefixes']['activities'], self.spl['prefixes']['roles'], self.spl['prefixes']['times'], self.spl['prefixes']['inter_attr'])

            if vectorizer in ['basic']:
                inter_case_vector = []
                results = self._predict_next_event_shared_cat_batch_prediction(parameters, results,
                                                                               self.spl['prefixes']['activities'],
                                                                               self.spl['prefixes']['roles'],
                                                                               self.spl['prefixes']['times'],
                                                                               inter_case_vector, vectorizer)
            elif vectorizer in ['inter']: #with intercase features
                results = self._predict_next_event_shared_cat_batch_prediction(parameters, results,
                                                                     self.spl['prefixes']['activities'],
                                                                     self.spl['prefixes']['roles'],
                                                                     self.spl['prefixes']['times'],
                                                                     self.spl['prefixes']['inter_attr'], vectorizer)

        # -----------------Generative Mode----------------------------------------
        elif parameters['batchpredchoice'] == 'Generative':

            # self._predict_next_event_shared_cat_batch_prediction(self, parameters, results,  self.spl['prefixes']['activities'], self.spl['prefixes']['roles'], self.spl['prefixes']['times'], self.spl['prefixes']['inter_attr'])

            if vectorizer in ['basic']:
                inter_case_vector = []
                results = self._predict_next_event_shared_cat_batch_generative(parameters, results,
                                                                               self.spl['prefixes']['activities'],
                                                                               self.spl['prefixes']['roles'],
                                                                               self.spl['prefixes']['times'],
                                                                               inter_case_vector, vectorizer)
            elif vectorizer in ['inter']: #with intercase features
                results = self._predict_next_event_shared_cat_batch_generative(parameters, results,
                                                                     self.spl['prefixes']['activities'],
                                                                     self.spl['prefixes']['roles'],
                                                                     self.spl['prefixes']['times'],
                                                                     self.spl['prefixes']['inter_attr'], vectorizer)

        return results

    def create_result_record_batch(self, index, spl, preds, parms, pref_size, results):
        record = dict()

        record['caseid'] = parms['caseid'][index]
        record['ac_prefix'] = spl['prefixes']['activities'][index]
        record['ac_expect'] = spl['next_evt']['activities'][index]
        record['ac_pred'] = preds[0]
        record['ac_prob'] = preds[3]
        record['rl_prefix'] = spl['prefixes']['roles'][index]
        record['rl_expect'] = spl['next_evt']['roles'][index]
        record['rl_pred'] = preds[1]
        record['rl_prob'] = preds[4]
        record['pref_size'] = pref_size

        # print("ac_pred : ", record['ac_pred'])
        # print('ac_prob : ', record['ac_prob'])
        # print('rl_pred : ', record['rl_pred'])
        # print('rl_prob :', record['rl_prob'])

        if parms['one_timestamp']:
            record['tm_prefix'] = [self.rescale(
               x, parms, parms['scale_args'])
               for x in spl['prefixes']['times'][index]]
            record['tm_expect'] = self.rescale(
                spl['next_evt']['times'][index][0],
                parms, parms['scale_args'])
            if parms['batchpredchoice'] in ['Prediction', 'Generative'] and parms['batch_mode'] == 'pre_prefix' and parms['variant'] in ['multi_pred', 'multi_pred_rand']:
                record['tm_pred'] = [self.rescale(x, parms, parms['scale_args'])
                                     for x in preds[2]]
            else:
                #--Converting back to original timestamp for expect
                if (len(results) == 0) or record['caseid'] != results[index - 1]['caseid']: #first time
                        f_time = [d['end_timestamp'] for d in parms['min_time'] if d['caseid'] == record['caseid']][0]
                        f_time.strftime(parms['read_options']['timeformat'])
                        _time = f_time
                elif record['caseid'] == results[index - 1]['caseid']:
                    _time = (results[index - 1]["end_timestamp_expected"] + timedelta(seconds=record['tm_expect']))
                record["end_timestamp_expected"] = _time
                record['tm_pred'] = self.rescale(
                    preds[2], parms, parms['scale_args'])
                #--Converting back to original timestamp for predict
                if (len(results) == 0) or record['caseid'] != results[index - 1]['caseid']: #first time
                        f_time = [d['end_timestamp'] for d in parms['min_time'] if d['caseid'] == record['caseid']][0]
                        f_time.strftime(parms['read_options']['timeformat'])
                        time = f_time
                elif record['caseid'] == results[index - 1]['caseid']:
                    time = (results[index - 1]["end_timestamp_pred"] + timedelta(seconds=record['tm_pred']))

                record["end_timestamp_pred"] = time

        else:
            # Duration
            record['dur_prefix'] = [self.rescale(
                x[0], parms, parms['scale_args']['dur'])
                for x in spl['prefixes']['times'][index]]
            record['dur_expect'] = self.rescale(
                spl['next_evt']['times'][index][0], parms,
                parms['scale_args']['dur'])
            record['dur_pred'] = self.rescale(
                preds[2], parms, parms['scale_args']['dur'])
            # Waiting
            record['wait_prefix'] = [self.rescale(
                x[1], parms, parms['scale_args']['wait'])
                for x in spl['prefixes']['times'][index]]
            record['wait_expect'] = self.rescale(
                spl['next_evt']['times'][index][1], parms,
                parms['scale_args']['wait'])
            record['wait_pred'] = self.rescale(
                preds[3], parms, parms['scale_args']['wait'])
        # print("record")
        # print(record)
        return record


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

    def _predict_next_event_shared_cat_batch_prediction(self, parameters, results,  _preac, _prerl, _pretm, _inter, vectorizer):
        # print("parefix number :", parameters['batchprefixnum'])
        for i, enm in enumerate(self.spl['prefixes']['activities']):

            # print("----------interation-------------------- : ", i)
            # print("enumerate input tm : ", _pretm[i])

            if (enm == [0] and len(enm) == 1) or (parameters['batchprefixnum'] + 1 == len(enm)): #first condition is for without prefix  other one is for with prefix for the first time
                # Activities and roles input shape(1,5)
                x_ac_ngram = (np.append(
                    np.zeros(parameters['dim']['time_dim']),
                    np.array(_preac[i]),
                    axis=0)[-parameters['dim']['time_dim']:]
                              .reshape((1, parameters['dim']['time_dim'])))

                x_rl_ngram = (np.append(
                    np.zeros(parameters['dim']['time_dim']),
                    np.array(_prerl[i]),
                    axis=0)[-parameters['dim']['time_dim']:]
                              .reshape((1, parameters['dim']['time_dim'])))

                # times input shape(1,5,1)
                times_attr_num = (_pretm[i].shape[1])
                x_t_ngram = np.array(
                    [np.append(np.zeros(
                        (parameters['dim']['time_dim'], times_attr_num)),
                        _pretm[i], axis=0)
                     [-parameters['dim']['time_dim']:]
                         .reshape((parameters['dim']['time_dim'], times_attr_num))]
                )

                if vectorizer in ['basic']:
                    inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram]
                elif vectorizer in ['inter']:
                    inter_attr_num = (_inter[i].shape[1])
                    x_inter_ngram = np.array(
                        [np.append(np.zeros((
                            parameters['dim']['time_dim'], inter_attr_num)),
                            _inter[i], axis=0)
                         [-parameters['dim']['time_dim']:]
                             .reshape((parameters['dim']['time_dim'], inter_attr_num))]
                    )
                    inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram, x_inter_ngram]

                pref_size = len(self.spl['prefixes']['activities'][i])

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

                    # changing array to numpy
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

                elif self.imp == 'multi_pred_rand':

                    # print("Executing Multi Random")

                    acx = np.array(preds[0][0])
                    rlx = np.array(preds[1][0])

                    pos = np.random.choice(np.arange(0, len(preds[0][0])), self.nx, replace=False,
                                           p=preds[0][0]).tolist()

                    pos1 = np.random.choice(np.arange(0, len(preds[1][0])), self.nx, replace=False,
                                            p=preds[1][0]).tolist()

                    pos_prob = []
                    pos1_prob = []

                    for ix in range(len(pos)):
                        # probability of activity
                        pos_prob.append(acx[pos[ix]])
                    for jx in range(len(pos1)):
                        # probability of role
                        pos1_prob.append(rlx[pos1[jx]])

                # save results
                if self.imp in ['multi_pred', 'multi_pred_rand']:
                    predictions = [pos, pos1, [preds[2][0][0]] * parameters['multiprednum'], pos_prob, pos1_prob]

                elif self.imp in ['arg_max', 'random_choice']:
                    predictions = [pos, pos1, preds[2][0][0], pos_prob, pos1_prob]

                #-------
                _pos = pos
                _pos1 = pos1
                _preds = preds[2][0][0]
                #-------
                # print("predicted time--0-- :", _preds)

                if not parameters['one_timestamp']:
                    predictions.extend([preds[2][0][1]])
                results.append(self.create_result_record_batch(i, self.spl, predictions, parameters, pref_size, results))

            # elif (enm != [0] and len(enm) != 1) or (parameters['batchprefixnum'] + 1 < len(enm)):
            elif (enm != [0] and parameters['batchprefixnum'] + 1 < len(enm)):
            # else: # for iteration >0

                _ac = _preac[i] #--Activity
                _rl = _prerl[i] #--Role
                _tm = _pretm[i] #--Time
                # print("predicted time :", _preds)

                for lk in range(parameters['multiprednum']):

                    if self.imp in ['multi_pred', 'multi_pred_rand']:
                        _ac = np.append(_ac[:-1], float(_pos[lk]))
                        _rl = np.append(_rl[:-1], float(_pos1[lk]))
                        # if i == 1:
                        # if len(enm) == 2:
                        if len(enm) == parameters['batchprefixnum'] + 2:
                            _tm = np.concatenate((_tm[:-1], np.array([[_preds]])), axis=0)
                        elif len(enm) > parameters['batchprefixnum'] + 2:
                            _tm = np.concatenate((_tm[:-1], np.array([[_preds[lk]]])), axis=0)
                    elif self.imp in ['arg_max', 'random_choice']:
                        _ac = np.append(_ac[:-1], float(_pos))
                        _rl = np.append(_rl[:-1], float(_pos1))
                        _tm = np.concatenate((_tm[:-1], np.array([[_preds]])), axis=0)

                    # print("As the Input : ", _ac)
                    # print("_rl : ", _rl)
                    # print("As the Input : : ", _tm)

                    # Activities and roles input shape(1,5)
                    x_ac_ngram = (np.append(
                        np.zeros(parameters['dim']['time_dim']),
                        np.array(_ac),
                        axis=0)[-parameters['dim']['time_dim']:]
                                  .reshape((1, parameters['dim']['time_dim'])))

                    x_rl_ngram = (np.append(
                        np.zeros(parameters['dim']['time_dim']),
                        np.array(_rl),
                        axis=0)[-parameters['dim']['time_dim']:]
                                  .reshape((1, parameters['dim']['time_dim'])))

                    # times input shape(1,5,1)
                    times_attr_num = (_tm.shape[1])
                    x_t_ngram = np.array(
                        [np.append(np.zeros(
                            (parameters['dim']['time_dim'], times_attr_num)),
                            _tm, axis=0)
                         [-parameters['dim']['time_dim']:]
                             .reshape((parameters['dim']['time_dim'], times_attr_num))]
                    )

                    if vectorizer in ['basic']:
                        inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram]
                    elif vectorizer in ['inter']:
                        inter_attr_num = (_inter[i].shape[1])
                        x_inter_ngram = np.array(
                            [np.append(np.zeros((
                                parameters['dim']['time_dim'], inter_attr_num)),
                                _inter[i], axis=0)
                             [-parameters['dim']['time_dim']:]
                                 .reshape((parameters['dim']['time_dim'], inter_attr_num))]
                        )
                        inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram, x_inter_ngram]

                    pref_size = len(self.spl['prefixes']['activities'][i])

                    preds = self.model.predict(inputs)

                    # print("Predictions :", preds)
                    if self.imp == 'random_choice':
                        # Use this to get a random choice following as PDF
                        _arr_pos = np.random.choice(np.arange(0, len(preds[0][0])),
                                               p=preds[0][0])
                        _arr_pos_prob = preds[0][0][_arr_pos]
                        _arr_pos1 = np.random.choice(np.arange(0, len(preds[1][0])),
                                                p=preds[1][0])
                        _arr_pos1_prob = preds[1][0][_arr_pos1]

                    elif self.imp == 'arg_max':

                        _arr_pos = np.argmax(preds[0][0])
                        _arr_pos_prob = preds[0][0][_arr_pos]

                        _arr_pos1 = np.argmax(preds[1][0])
                        _arr_pos1_prob = preds[1][0][_arr_pos1]

                    elif self.imp == 'multi_pred':

                        if lk == 0:
                            _arr_pos = []
                            _arr_pos_prob = []
                            _arr_pos1 = []
                            _arr_pos1_prob = []
                            _arr_pred = []

                        # Use this to get the max prediction in multipred
                        _arr_pos.append(np.argmax(preds[0][0]))
                        _arr_pos_prob.append(preds[0][0][_arr_pos[lk]])

                        _arr_pos1.append(np.argmax(preds[1][0]))
                        _arr_pos1_prob.append(preds[1][0][_arr_pos1[lk]])

                        _arr_pred.append(preds[2][0][0])

                    elif self.imp == 'multi_pred_rand':

                        if lk == 0:
                            _arr_pos = []
                            _arr_pos_prob = []
                            _arr_pos1 = []
                            _arr_pos1_prob = []
                            _arr_pred = []

                        _arr_pos.append(np.random.choice(np.arange(0, len(preds[0][0])), p=preds[0][0]))
                        _arr_pos_prob.append(preds[0][0][_arr_pos[lk]])
                        _arr_pos1.append(np.random.choice(np.arange(0, len(preds[1][0])), p=preds[1][0]))
                        _arr_pos1_prob.append(preds[1][0][_arr_pos1[lk]])

                        _arr_pred.append(preds[2][0][0])


                _pos = _arr_pos
                _pos1 = _arr_pos1

                if self.imp in ['multi_pred', 'multi_pred_rand']:
                    # _preds = sum(_arr_pred)/len(_arr_pred)
                    _preds = _arr_pred
                elif self.imp in ['arg_max', 'random_choice']:
                    _preds = preds[2][0][0]

                predictions = [_pos, _pos1, _preds, _arr_pos_prob, _arr_pos1_prob]

                if not parameters['one_timestamp']:
                    predictions.extend([preds[2][0][1]])
                results.append(self.create_result_record_batch(i, self.spl, predictions, parameters, pref_size, results))
        sup.print_done_task()
        return results

    def _predict_next_event_shared_cat_batch_generative(self, parameters, results,  _preac, _prerl, _pretm, _inter, vectorizer):
        # print("multi pred : ", parameters['multiprednum'])
        for i, enm in enumerate(self.spl['prefixes']['activities']):

            # print("------------------ith Number:--------------------------", i)
            # print("_ac - in the test log : ", _preac[i])

            # if enm == [0] and len(enm) == 1:
            if (enm == [0] and len(enm) == 1) or (parameters['batchprefixnum'] + 1 == len(enm)):  # first condition is for without prefix  other one is for with prefix for the first time

                preds_prefix = list() # reinitiate the list

                # Activities and roles input shape(1,5)
                x_ac_ngram = (np.append(
                    np.zeros(parameters['dim']['time_dim']),
                    np.array(_preac[i]),
                    axis=0)[-parameters['dim']['time_dim']:]
                              .reshape((1, parameters['dim']['time_dim'])))

                x_rl_ngram = (np.append(
                    np.zeros(parameters['dim']['time_dim']),
                    np.array(_prerl[i]),
                    axis=0)[-parameters['dim']['time_dim']:]
                              .reshape((1, parameters['dim']['time_dim'])))

                # times input shape(1,5,1)
                times_attr_num = (_pretm[i].shape[1])
                x_t_ngram = np.array(
                    [np.append(np.zeros(
                        (parameters['dim']['time_dim'], times_attr_num)),
                        _pretm[i], axis=0)
                     [-parameters['dim']['time_dim']:]
                         .reshape((parameters['dim']['time_dim'], times_attr_num))]
                )

                if vectorizer in ['basic']:
                    inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram]
                elif vectorizer in ['inter']:
                    inter_attr_num = (_inter[i].shape[1])
                    x_inter_ngram = np.array(
                        [np.append(np.zeros((
                            parameters['dim']['time_dim'], inter_attr_num)),
                            _inter[i], axis=0)
                         [-parameters['dim']['time_dim']:]
                             .reshape((parameters['dim']['time_dim'], inter_attr_num))]
                    )
                    inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram, x_inter_ngram]

                pref_size = len(self.spl['prefixes']['activities'][i])

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

                    # changing array to numpy
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

                elif self.imp == 'multi_pred_rand':

                    acx = np.array(preds[0][0])
                    rlx = np.array(preds[1][0])

                    pos = np.random.choice(np.arange(0, len(preds[0][0])), self.nx, replace=False,
                                           p=preds[0][0]).tolist()

                    pos1 = np.random.choice(np.arange(0, len(preds[1][0])), self.nx, replace=False,
                                            p=preds[1][0]).tolist()

                    pos_prob = []
                    pos1_prob = []

                    for ix in range(len(pos)):
                        # probability of activity
                        pos_prob.append(acx[pos[ix]])
                    for jx in range(len(pos1)):
                        # probability of role
                        pos1_prob.append(rlx[pos1[jx]])

                # save results
                if self.imp in ['multi_pred', 'multi_pred_rand']:

                    predictions = [pos, pos1, [preds[2][0][0]] * parameters['multiprednum'], pos_prob, pos1_prob]

                    preds_prefix.append([pos, pos1, [preds[2][0][0]] * parameters['multiprednum']])

                elif self.imp in ['arg_max', 'random_choice']:

                    predictions = [pos, pos1, preds[2][0][0], pos_prob, pos1_prob]

                    preds_prefix.append([pos, pos1, preds[2][0][0]])

                #-------
                _pos = pos
                _pos1 = pos1
                _preds = preds[2][0][0]
                #-------

                # print("Prediction of the iteration  : ", preds_prefix)

                if not parameters['one_timestamp']:
                    predictions.extend([preds[2][0][1]])
                results.append(self.create_result_record_batch(i, self.spl, predictions, parameters, pref_size, results))

            # elif enm != [0] and len(enm) != 1:
            elif (enm != [0] and parameters['batchprefixnum'] + 1 < len(enm)):
            # else: # for iteration >0
            #     print("Prediction of the iteration  : ", preds_prefix)
                # print("_rl0 : ", _prerl)
                # print("_tm0 : ", _pretm)

                _ac = _preac[i] #--Activity
                _rl = _prerl[i] #--Role
                _tm = _pretm[i] #--Time

                for lk in range(parameters['multiprednum']):

                    _temp_ac = list()
                    _temp_rl = list()
                    _temp_tm = list()

                    if self.imp in ['multi_pred', 'multi_pred_rand']:

                        for gk in range(len(preds_prefix)):
                            _temp_ac.append(float(preds_prefix[gk][0][lk]))
                            _temp_rl.append(float(preds_prefix[gk][1][lk]))
                            _temp_tm.append(float(preds_prefix[gk][2][lk]))

                        # _temp_ac = np.array(_ac[:1] + _temp_ac)
                        # _temp_rl = np.array(_rl[:1] + _temp_rl)
                        # _temp_tm = np.concatenate((_tm[:1], np.dstack([_temp_tm])[0]), axis=0)

                    elif self.imp in ['arg_max', 'random_choice']:
                        # _ac = np.append(_ac[:-1], float(_pos))
                        # _rl = np.append(_rl[:-1], float(_pos1))
                        # _tm = np.concatenate((_tm[:-1], np.array([[_preds]])), axis=0)
                        #
                        # _temp_ac = list()
                        # _temp_rl = list()
                        # _temp_tm = list()

                        for gk in range(len(preds_prefix)):
                            _temp_ac.append(float(preds_prefix[gk][0]))
                            _temp_rl.append(float(preds_prefix[gk][1]))
                            _temp_tm.append(float(preds_prefix[gk][2]))

                    # _temp_ac = np.array(_ac[:1] + _temp_ac)
                    # _temp_rl = np.array(_rl[:1] + _temp_rl)
                    # _temp_tm = np.concatenate((_tm[:1], np.dstack([_temp_tm])[0]), axis=0)

                    _temp_ac = np.array(_ac[:parameters['batchprefixnum'] + 1] + _temp_ac)
                    _temp_rl = np.array(_rl[:parameters['batchprefixnum'] + 1] + _temp_rl)
                    _temp_tm = np.concatenate((_tm[:parameters['batchprefixnum'] + 1], np.dstack([_temp_tm])[0]), axis=0)

                    # print("Input _ac : ", _temp_ac)
                    # print("Input _rl : ", _temp_rl)
                    # print("Input _tm : ", _temp_tm)

                    # Activities and roles input shape(1,5)
                    x_ac_ngram = (np.append(
                        np.zeros(parameters['dim']['time_dim']),
                        np.array(_temp_ac),
                        axis=0)[-parameters['dim']['time_dim']:]
                                  .reshape((1, parameters['dim']['time_dim'])))

                    x_rl_ngram = (np.append(
                        np.zeros(parameters['dim']['time_dim']),
                        np.array(_temp_rl),
                        axis=0)[-parameters['dim']['time_dim']:]
                                  .reshape((1, parameters['dim']['time_dim'])))

                    # times input shape(1,5,1)
                    times_attr_num = (_temp_tm.shape[1])
                    x_t_ngram = np.array(
                        [np.append(np.zeros(
                            (parameters['dim']['time_dim'], times_attr_num)),
                            _temp_tm, axis=0)
                         [-parameters['dim']['time_dim']:]
                             .reshape((parameters['dim']['time_dim'], times_attr_num))]
                    )

                    if vectorizer in ['basic']:
                        inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram]
                    elif vectorizer in ['inter']:
                        inter_attr_num = (_inter[i].shape[1])
                        x_inter_ngram = np.array(
                            [np.append(np.zeros((
                                parameters['dim']['time_dim'], inter_attr_num)),
                                _inter[i], axis=0)
                             [-parameters['dim']['time_dim']:]
                                 .reshape((parameters['dim']['time_dim'], inter_attr_num))]
                        )
                        inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram, x_inter_ngram]

                    pref_size = len(self.spl['prefixes']['activities'][i])

                    preds = self.model.predict(inputs)

                    if self.imp == 'random_choice':
                        # Use this to get a random choice following as PDF
                        _arr_pos = np.random.choice(np.arange(0, len(preds[0][0])),
                                               p=preds[0][0])
                        _arr_pos_prob = preds[0][0][_arr_pos]
                        _arr_pos1 = np.random.choice(np.arange(0, len(preds[1][0])),
                                                p=preds[1][0])
                        _arr_pos1_prob = preds[1][0][_arr_pos1]

                    elif self.imp == 'arg_max':

                        _arr_pos = np.argmax(preds[0][0])
                        _arr_pos_prob = preds[0][0][_arr_pos]

                        _arr_pos1 = np.argmax(preds[1][0])
                        _arr_pos1_prob = preds[1][0][_arr_pos1]

                    elif self.imp == 'multi_pred':

                        if lk == 0:
                            _arr_pos = []
                            _arr_pos_prob = []
                            _arr_pos1 = []
                            _arr_pos1_prob = []
                            _arr_pred = []

                        # Use this to get the max prediction in multipred
                        _arr_pos.append(np.argmax(preds[0][0]))
                        _arr_pos_prob.append(preds[0][0][_arr_pos[lk]])

                        _arr_pos1.append(np.argmax(preds[1][0]))
                        _arr_pos1_prob.append(preds[1][0][_arr_pos1[lk]])

                        _arr_pred.append(preds[2][0][0])

                    elif self.imp == 'multi_pred_rand':

                        if lk == 0:
                            _arr_pos = []
                            _arr_pos_prob = []
                            _arr_pos1 = []
                            _arr_pos1_prob = []
                            _arr_pred = []

                        _arr_pos.append(np.random.choice(np.arange(0, len(preds[0][0])), p=preds[0][0]))
                        _arr_pos_prob.append(preds[0][0][_arr_pos[lk]])
                        _arr_pos1.append(np.random.choice(np.arange(0, len(preds[1][0])), p=preds[1][0]))
                        _arr_pos1_prob.append(preds[1][0][_arr_pos1[lk]])

                        _arr_pred.append(preds[2][0][0])

                _pos = _arr_pos
                _pos1 = _arr_pos1

                if self.imp in ['multi_pred', 'multi_pred_rand']:
                    # _preds = sum(_arr_pred)/len(_arr_pred)
                    _preds = _arr_pred
                elif self.imp in ['arg_max', 'random_choice']:
                    _preds = preds[2][0][0]

                # save results
                predictions = [_pos, _pos1, _preds, _arr_pos_prob, _arr_pos1_prob]

                preds_prefix.append([_pos, _pos1, _preds])

                if not parameters['one_timestamp']:
                    predictions.extend([preds[2][0][1]])
                results.append(self.create_result_record_batch(i, self.spl, predictions, parameters, pref_size, results))
        sup.print_done_task()
        return results
