# -*- coding: utf-8 -*-
import datetime
import gzip
import itertools as it
import os
import zipfile as zf
from operator import itemgetter

import pandas as pd

from support_modules import support as sup


class LogReader(object):
    """
    This class reads and parse the elements of a given event-log
    expected format .xes or .csv
    """

    def __init__(self, input, settings):
        """constructor"""
        #print("log reader input :", self.input)
        # print("log reader settings :", settings)
        self.input = input
        self.file_name, self.file_extension = self.define_ftype()
        self.timeformat = settings['timeformat']
        self.column_names = settings['column_names']
        self.one_timestamp = settings['one_timestamp']
        self.filter_d_attrib = settings['filter_d_attrib']
        self.ns_include = settings['ns_include']

        self.data = list()
        self.raw_data = list()
        self.load_data_from_file()

    def load_data_from_file(self):
        """
        reads all the data from the log depending
        the extension of the file
        """
        if self.file_extension == '.csv':
            self.get_csv_events_data()

# =============================================================================
# csv methods
# =============================================================================
    def get_csv_events_data(self):
        """
        reads and parse all the events information from a csv file
        """
        sup.print_performed_task('Reading log traces ')
        log = pd.read_csv(self.input, dtype={'user': str})
        if self.one_timestamp:
            self.column_names['Complete Timestamp'] = 'end_timestamp'
            log = log.rename(columns=self.column_names)
            log = log.astype({'caseid': object})
            log = (log[(log.task != 'Start') & (log.task != 'End')]
                   .reset_index(drop=True))
            if self.filter_d_attrib:
                log = log[['caseid', 'task', 'user', 'end_timestamp']]
            log['end_timestamp'] = pd.to_datetime(log['end_timestamp'],
                                                  format=self.timeformat)
        else:
            self.column_names['Start Timestamp'] = 'start_timestamp'
            self.column_names['Complete Timestamp'] = 'end_timestamp'
            log = log.rename(columns=self.column_names)
            log = log.astype({'caseid': object})
            log = (log[(log.task != 'Start') & (log.task != 'End')]
                   .reset_index(drop=True))
            if self.filter_d_attrib:
                log = log[['caseid', 'task', 'user',
                           'start_timestamp', 'end_timestamp']]
            log['start_timestamp'] = pd.to_datetime(log['start_timestamp'],
                                                    format=self.timeformat)
            log['end_timestamp'] = pd.to_datetime(log['end_timestamp'],
                                                  format=self.timeformat)

        log['user'].fillna('SYS', inplace=True)
        self.data = log.to_dict('records')
        self.append_csv_start_end()
        self.split_event_transitions()
        sup.print_done_task()

    def split_event_transitions(self):
        temp_raw = list()
        if self.one_timestamp:
            for event in self.data:
                temp_event = event.copy()
                temp_event['timestamp'] = temp_event.pop('end_timestamp')
                temp_event['event_type'] = 'complete'
                temp_raw.append(temp_event)
        else:
            for event in self.data:
                start_event = event.copy()
                complete_event = event.copy()
                start_event.pop('end_timestamp')
                complete_event.pop('start_timestamp')
                start_event['timestamp'] = start_event.pop('start_timestamp')
                complete_event['timestamp'] = complete_event.pop('end_timestamp')
                start_event['event_type'] = 'start'
                complete_event['event_type'] = 'complete'
                temp_raw.append(start_event)
                temp_raw.append(complete_event)
        self.raw_data = temp_raw

    def append_csv_start_end(self):
        new_data = list()
        data = sorted(self.data, key=lambda x: x['caseid'])
        for key, group in it.groupby(data, key=lambda x: x['caseid']):
            trace = list(group)
            for new_event in ['Start', 'End']:
                idx = 0 if new_event == 'Start' else -1
                t_key = 'end_timestamp'
                if not self.one_timestamp and new_event == 'Start':
                    t_key = 'start_timestamp'
                temp_event = dict()
                temp_event['caseid'] = trace[idx]['caseid']
                temp_event['task'] = new_event
                temp_event['user'] = new_event
                temp_event['end_timestamp'] = trace[idx][t_key]
                if not self.one_timestamp:
                    temp_event['start_timestamp'] = trace[idx][t_key]
                if new_event == 'Start':
                    trace.insert(0, temp_event)
                else:
                    trace.append(temp_event)
            new_data.extend(trace)
        self.data = new_data

# =============================================================================
# Accesssor methods
# =============================================================================
    def get_traces(self):
        """
        returns the data splitted by caseid and ordered by start_timestamp
        """
        cases = list(set([x['caseid'] for x in self.data]))
        traces = list()
        for case in cases:
            order_key = 'end_timestamp' if self.one_timestamp else 'start_timestamp'
            trace = sorted(
                list(filter(lambda x: (x['caseid'] == case), self.data)),
                key=itemgetter(order_key))
            traces.append(trace)
        return traces

    def get_raw_traces(self):
        """
        returns the raw data splitted by caseid and ordered by timestamp
        """
        cases = list(set([c['caseid'] for c in self.raw_data]))
        traces = list()
        for case in cases:
            trace = sorted(
                list(filter(lambda x: (x['caseid'] == case), self.raw_data)),
                key=itemgetter('timestamp'))
            traces.append(trace)
        return traces

    def set_data(self, data):
        """
        seting method for the data attribute
        """
        self.data = data

# =============================================================================
# Support Method
# =============================================================================
    def define_ftype(self):
        filename, file_extension = os.path.splitext(self.input)
        # if file_extension in ['.xes', '.csv', '.mxml']:
        if file_extension in ['.xes', '.csv']:
            filename = filename + file_extension
            file_extension = file_extension
        elif file_extension == '.gz':
            outFileName = filename
            filename, file_extension = self.decompress_file_gzip(outFileName)
        elif file_extension == '.zip':
            filename, file_extension = self.decompress_file_zip(filename)
        else:
            raise IOError('file type not supported')
        return filename, file_extension

    # Decompress .gz files
    def decompress_file_gzip(self, outFileName):
        inFile = gzip.open(self.input, 'rb')
        outFile = open(outFileName, 'wb')
        outFile.write(inFile.read())
        inFile.close()
        outFile.close()
        _, fileExtension = os.path.splitext(outFileName)
        return outFileName, fileExtension

    # Decompress .zip files
    def decompress_file_zip(self, outfilename):
        with zf.ZipFile(self.input, "r") as zip_ref:
            zip_ref.extractall("../inputs/")
        _, fileExtension = os.path.splitext(outfilename)
        return outfilename, fileExtension
