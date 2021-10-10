# -*- coding: utf-8 -*-
from model_prediction import next_event_samples_creator as nesc

from model_prediction.next_modes import next_event_predictor_batch_prefix_mode as nepe, \
    next_event_predictor_single_mode_execution as nepsmexe, next_event_predictor_batch_base_mode as nepb, \
    next_event_predictor_single_mode_evaluation as nepsmeva, next_event_predictor_single_mode_whatif as nepsmwia


class SamplesCreator:
    def create(self, predictor, activity):
        sampler = self._get_samples_creator(activity)
        predictor.sampling(sampler)

    def _get_samples_creator(self, activity):
        if activity == 'predict_next':
            return nesc.NextEventSamplesCreator()
        else:
            raise ValueError(activity)


class PredictionTasksExecutioner:
    def predict(self, predictor, activity, mode, next_mode, batch_option):
        executioner = self._get_predictor(activity, mode, next_mode, batch_option)
        predictor.predict(executioner, mode)

    def _get_predictor(self, activity, mode, next_mode, batch_option):
        if activity == 'predict_next':
            if mode == 'next':
                if next_mode == 'history_with_next':
                    return nepsmexe.NextEventPredictor()
                elif next_mode == 'next_action':
                    return nepsmeva.NextEventPredictor()
                elif next_mode == 'what_if':
                    return nepsmwia.NextEventPredictor()
            elif mode == 'batch':
                if batch_option == 'base_batch':
                    return nepb.NextEventPredictor()
                elif batch_option == 'pre_prefix':
                    return nepe.NextEventPredictor()
        else:
            raise ValueError(activity)
