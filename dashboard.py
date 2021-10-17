import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import sys
import json
import time
import getopt
import pandas as pd
import streamlit as st

from model_prediction import model_predictor as pr

# ----------------------------------------------------------------------------------
# --                         Dashboard Global View                             -- #
# ----------------------------------------------------------------------------------

# -- Page Config
st.set_page_config(layout="wide", initial_sidebar_state="auto", page_title='NxEventPred', page_icon="🌌")

# -- Page Customization
max_width_str = f"max-width: 1500px;"
st.markdown(f""" <style>
    .reportview-container .main .block-container {{{max_width_str}
    }} </style> """, unsafe_allow_html=True)

# -- Condense the layout
padding = 1
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)

# -- hides both menu and footer watermark

# #Hide the menu button
# st.markdown(""" <style>
# #MainMenu {visibility: hidden;}
# footer {visibility: hidden;}
# </style> """, unsafe_allow_html=True)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        """

# -- hides footer watermark
# hide_menu_style = """
#         <style>
#         footer {visibility: hidden;}
#         </style>
#         """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# -- Control Panel Name
st.sidebar.title("🎛️ App Control Menu")  # Adding the header to the sidebar as well as the sidebar


def catch_parameter(opt):
    """Change the captured parameters names"""
    switch = {'-h': 'help', '-o': 'one_timestamp', '-a': 'activity',
              '-f': 'file_name', '-i': 'imp', '-l': 'lstm_act',
              '-d': 'dense_act', '-p': 'optim', '-n': 'norm_method',
              '-m': 'model_type', '-z': 'n_size', '-y': 'l_size',
              '-c': 'folder', '-b': 'model_file', '-x': 'is_single_exec',
              '-t': 'max_trace_size', '-e': 'splits', '-g': 'sub_group',
              '-v': 'variant', '-r': 'rep'}
    try:
        return switch[opt]
    except:
        raise Exception('Invalid option ' + opt)


def clear_cache():
    session_state_key_list = ['multi_pred_ss', 'pos_tm_ss', 'initial_prediction',
                              'prediction_choice_idx', 'prediction_choice_name', 'history_of_choice']
    _temp_sessionstate_keys = [*st.session_state]
    for key in _temp_sessionstate_keys:
        if key in session_state_key_list:
            del st.session_state[key]


def main(argv):
    parameters = dict()

    column_names = {'Case ID': 'caseid',
                    'Activity': 'task',
                    'lifecycle:transition': 'event_type',
                    'Resource': 'user'}

    if not argv:

        parameters['one_timestamp'] = True  # Only one timestamp

        parameters['read_options'] = {
            'timeformat': '%Y-%m-%dT%H:%M:%S.%f',
            'column_names': column_names,
            'one_timestamp': parameters['one_timestamp'],
            'ns_include': True}

        # -- Navigate Between Different Views
        navigate_page = st.sidebar.selectbox("🔗 Navigate", ["Predictive-Process-Monitoring", "About"], key="navigation_page",
                                             on_change=clear_cache)

        # ----------------------------------------------------------------------------------
        # --                         About Section                                      -- #
        # ----------------------------------------------------------------------------------

        body =  '''
                ## Dashboard for Predictive Process Monitoring using LSTM

                #### The original development repository is : [Predictive Process Monitoring Dashboard for Emulating LSTM Models over Business Processes and its Qualitative Evaluation](https://github.com/rhnfzl/business-process-dashboard-for-lstm).

                #### It is being hosted for the pourpose of demonstration of it's capabilities only on [Sepsis Event Log](https://data.4tu.nl/articles/dataset/Sepsis_Cases_-_Event_Log/12707639) with [limited functionalities due to framework](https://github.com/rhnfzl/streamlit-predictive-process-monitoring-dashboard-using-lstm#known-issues) it has been built upon. It might be incorporated with [Future enhancements](https://github.com/rhnfzl/ppm-dashboard#future-enhancement).

                #### A LSTM model has been trained to showcase different mode of predictions. The base LSTM architecture has been taken from [Camargo et al.](https://link.springer.com/chapter/10.1007/978-3-030-26619-6_19) and has been enhanced to incorporate intra and inter case features.

                #### The dashboard offers multiple options to predict the next activity and also let to play with What-If scenaios (Currently not working).

                You can reach out to me : [Twitter](https://twitter.com/rhnfzl) or [LinkedIn](https://www.linkedin.com/in/rhnfzl/).
                '''


        if navigate_page == "About":

            st.caption('About Project')

            st.markdown(body)


        # ----------------------------------------------------------------------------------
        # --                 Dashboard Home (Simulation) Control View                   -- #
        # ----------------------------------------------------------------------------------

        elif navigate_page == "Predictive-Process-Monitoring":

            # Dashboard Name
            html_temp = """
            <div style="background-color:tomato;padding:1.5px">
            <h1 style="color:white;text-align:center;">⏭️Next Event Prediction Dashboard </h1>
            </div><br>"""
            st.markdown(html_temp, unsafe_allow_html=True)

            parameters['activity'] = 'predict_next'

            # --Hard Coded Value
            parameters['is_single_exec'] = True
            parameters['rep'] = 1

            # --- Common Functions Used Across different sub-page

            @st.cache(persist=True)
            def _list2dictConvert(_a):
                _it = iter(_a)
                _res_dct = dict(zip(_it, _it))
                return _res_dct

            def common_batch_varient(actcount, rolecount):
                with st.sidebar.expander('Variant'):
                    st.info("Select **Max Probability** for the most probable events,"
                            " **Multiple Prediction** for the prediction of the multiple events, "
                            "and **Random Prediction** for prediction of random recommendation of events")
                    variant_opt = st.selectbox("", (
                        'Max Probability', 'Multiple Prediction', 'Random Prediction',
                        'Multiple Random Prediction'),
                                               key='variant_opt', on_change=clear_cache)

                if variant_opt == 'Max Probability':
                    variant_opt = 'arg_max'
                    slider = 1
                elif variant_opt == 'Random Prediction':
                    variant_opt = 'random_choice'
                    slider = 1
                elif variant_opt in ['Multiple Prediction', 'Multiple Random Prediction']:
                    if variant_opt == 'Multiple Prediction':
                        variant_opt = 'multi_pred'
                    elif variant_opt == 'Multiple Random Prediction':
                        variant_opt = 'multi_pred_rand'
                    # variant_opt = 'multi_pred'

                    if actcount > rolecount:
                        _maxmulti = rolecount
                    elif rolecount > actcount:
                        _maxmulti = actcount
                    else:
                        _maxmulti = actcount

                    with st.sidebar.expander('Number of Predictions'):
                        st.info(
                            "**Multiple Predictions : ** minimum 2 predictions and maximum value based on slider")
                        slider = st.slider(
                            label='', min_value=2,
                            max_value=_maxmulti, key='my_number_prediction_slider_batch', on_change=clear_cache)
                return variant_opt, slider

            @st.cache(persist=True)
            def read_next_testlog():
                input_file = os.path.join('output_files', parameters['folder'], 'parameters', 'test_log.csv')
                parameter_file = os.path.join('output_files', parameters['folder'], 'parameters',
                                              'model_parameters.json')
                filter_log = pd.read_csv(input_file, dtype={'user': str})
                # Standard Code based on log_reader
                filter_log = filter_log.rename(columns=column_names)
                filter_log = filter_log.astype({'caseid': object})
                filter_log = (
                    filter_log[(filter_log.task != 'Start') & (filter_log.task != 'End')].reset_index(drop=True))
                filter_log_columns = filter_log.columns
                with open(parameter_file) as pfile:
                    parameter_data = json.load(pfile)
                    file_name = parameter_data["file_name"]
                    pfile.close()
                return filter_log, filter_log_columns, file_name

            def next_columns(filter_log, display_columns, index=None, flag=None):
                # Dashboard selection of Case ID
                if index == None:
                    filter_caseid = st.selectbox("🆔 Select Case ID", filter_log["caseid"].unique(),
                                                 key="caseid_select", on_change=clear_cache)
                    filter_caseid_index = filter_log["caseid"].unique().tolist().index(filter_caseid)
                else:
                    if flag != None:
                        filter_caseid = st.selectbox("🆔 Select Case ID", filter_log["caseid"].unique()[index],
                                                     key="caseid_select", on_change=clear_cache)
                        filter_caseid_index = filter_log["caseid"].unique().tolist().index(filter_caseid)
                        flag = None
                filter_caseid_attr_df = filter_log.loc[filter_log["caseid"].isin([filter_caseid])]
                filter_attr_display = filter_caseid_attr_df[display_columns]
                return filter_attr_display, filter_caseid, filter_caseid_attr_df, filter_caseid_index

            def num_predictions(_df):
                if _df['task'].nunique() > _df['role'].nunique():
                    _dfmax = _df['role'].nunique()
                elif _df['role'].nunique() > _df['task'].nunique():
                    _dfmax = _df['task'].nunique()
                else:
                    _dfmax = _df['task'].nunique()

                with st.sidebar.expander('Variant'):
                    st.info("Select the slider value **equal to 1** for **Max Probability** "
                            "otherwise choose the number **greater than 1** for "
                            "**Multiple Predictions** sorted in decreasing order based on probability of it's ocurrance")
                    slider = st.slider(
                        label='', min_value=1,
                        max_value=_dfmax, key='my_number_prediction_slider', on_change=clear_cache)
                return slider

            def label_identifier(file_name):
                # --- Logic to be added for other dataset
                with st.sidebar.expander('Label Checker'):
                    if "sepsis" in file_name:
                        st.info("Select the respective labeling mode for Sepsis Patient Condition")
                        label_indicator = ["Returns to Emergency Room", "Admitted to Intensive Care",
                                           "Discharged for other Reason"]
                        _labelsel = st.radio("Labelling Indicator", label_indicator, key="radio_select_label",
                                             on_change=clear_cache)
                        if _labelsel == "Returns to Emergency Room":
                            parameters['label_activity'] = "Return ER"
                        elif _labelsel == "Admitted to Intensive Care":
                            parameters['label_activity'] = "Admission IC"
                        elif _labelsel == "Discharged for other Reason":
                            parameters['label_activity'] = "Release A"
                        parameters['label_check_event'] = st.number_input("Check after number of events",
                                                                          min_value=1, max_value=25, value=5,
                                                                          # These are estimate numbers
                                                                          step=1, key="radio_number_label",
                                                                          on_change=clear_cache)
                    else:
                        st.error("Label Logic for dataset is not defined")

            # ----------------------------------------------------

            # Folder picker / provide name from output_files
            if parameters['activity'] in ['predict_next']:
                # with st.sidebar.expander('Folder Name'):
                #     st.info("Provide the folder for which the prediction has to be simulated")
                #     _folder_name = st.text_input('', key="folder_name", on_change=clear_cache)
                # parameters['folder'] = _folder_name
                st.sidebar.info('Sepsis Dataset Simulation')
                _folder_name = '20210926_42671FEF_0DDC_4E55_82C8_0653FEC85037'
                parameters['folder'] = _folder_name

                if parameters['folder'] != "":
                    # --Selecting Model
                    _model_directory = os.path.join('output_files', _folder_name)
                    _model_name = []
                    for file in os.listdir(_model_directory):
                        # check the files which are end with specific extention
                        if file.endswith(".h5"):
                            _model_name.append(file)
                    parameters['model_file'] = _model_name[-1]

                    # --Selecting Mode
                    with st.sidebar.expander('Type of Prediction Processing Mode'):
                        st.info("Select **Batch Processing** for the prediction of eintire log file,"
                                " **Single Processing** to simulate the prediction for each Case Id individually")
                        # next_option = st.sidebar.radio('', ['Execution Mode', 'Evaluation Mode'])
                        # st.sidebar.subheader("Choose Mode of Prediction")
                        _mode_sel = st.radio('Processing', ['Batch Processing', 'Single Event Processing'],
                                             key="processing_type", on_change=clear_cache)
                    # st.sidebar.markdown("""---""")

                    if _mode_sel == 'Batch Processing':
                        _mode_sel = 'batch'
                    elif _mode_sel == 'Single Event Processing':
                        _mode_sel = 'next'

                    parameters['mode'] = _mode_sel

                    # if parameters['folder'] != "":
                    if parameters['activity'] in ['predict_next']:

                        # ----------------------------------------------------------------------------------
                        # --             Dashboard Simulation Single Mode Control View                  -- #
                        # ----------------------------------------------------------------------------------

                        if parameters['mode'] in ['next']:
                            # ---batch default values
                            parameters['batch_mode'] = ''
                            parameters['batchprefixnum'] = ''
                            parameters['batchpredchoice'] = ''
                            parameters['batchlogrange'] = ''
                            #   Saves the result in the URL in the Next mode
                            app_state = st.experimental_get_query_params()
                            if "my_saved_result" in app_state:
                                saved_result = app_state["my_saved_result"][0]
                                nxt_button_idx = int(saved_result)
                                # st.write("Here is your result", saved_result)
                            else:
                                # st.write("No result to display, compute a value first.")
                                nxt_button_idx = 0

                            # next_option = st.sidebar.selectbox("Type of Single Event Processing", ('Prediction of Next Event', 'Prediction of Next Events with Suffix'), key='next_dropdown_opt')
                            with st.sidebar.expander('Type of Single Event Processing'):
                                st.info(
                                    "Select **Execution Mode** for simulating the dashboard for the Users, **Evaluation Mode** to judge the trustworthiness of the ML model prediction")
                                next_option = st.radio('', ['Execution Mode', 'Evaluation Mode', 'What-If Mode'], index=1,
                                                       key="single_event_processing", on_change=clear_cache)
                            # st.sidebar.markdown("""---""")

                            if next_option == 'Execution Mode':
                                next_option = 'history_with_next'
                            elif next_option == 'Evaluation Mode':
                                next_option = 'next_action'
                            elif next_option == 'What-If Mode':
                                next_option = 'what_if'
                            # Read the Test Log
                            filter_log, filter_log_columns, file_name = read_next_testlog()
                            essential_columns = ['task', 'role', 'end_timestamp']
                            norm_cols = [col for col in filter_log.columns if '_norm' in col]
                            index_cols = [col for col in filter_log.columns if '_index' in col]
                            ord_cols = [col for col in filter_log.columns if '_ord' in col]
                            ohe_cols = [col for col in filter_log.columns if '_ohe' in col]
                            extra_columns = ['caseid', 'label', 'dur', 'acc_cycle', 'daytime', 'user',
                                             # 'dur_norm', 'ac_index', 'rl_index', 'label_index',
                                             # 'wait_norm', 'open_cases_norm', 'daytime_norm',
                                             'acc_cycle']  # Add the Columns here which you don't want to display
                            display_columns = list(set(filter_log_columns) - set(
                                essential_columns + extra_columns + norm_cols + index_cols + ord_cols + ohe_cols))
                            filter_attr_display, filter_caseid, filter_caseid_attr_df, filter_caseid_index = next_columns(
                                filter_log, display_columns)
                            parameters['nextcaseid'] = filter_caseid

                            st.subheader('🔦 State of the Process')
                            statecols, _, buttoncols = st.columns([2, 0.25, 0.2])
                            with statecols:
                                state_of_theprocess = st.empty()
                            with buttoncols:
                                button_of_theprocess = st.empty()

                            parameters['multiprednum'] = num_predictions(filter_caseid_attr_df)

                            if parameters['multiprednum'] == 1:
                                variant_opt = 'arg_max'
                            elif parameters['multiprednum'] > 1:
                                variant_opt = 'multi_pred'

                            parameters['variant'] = variant_opt
                            parameters['next_mode'] = next_option

                            filter_caseid_attr_df = filter_caseid_attr_df[essential_columns].values.tolist()

                            # ---Logic to select the label
                            label_identifier(file_name)

                            # ----------------------------------------------------------------------------------
                            # --             Dashboard Simulation Evaluation Sub-Control View               -- #
                            # ----------------------------------------------------------------------------------

                            if next_option == 'next_action':

                                # with st.sidebar.expander('Choose Prefix Source'):
                                #     st.info("Select **SME** for simulating the input to prediction using the log, "
                                #             "**Prediction** to use the respective prediction as the input for subsequent prediction")
                                #     parameters['nextactpredchoice'] = st.radio('', ['SME', 'Prediction'], key="radio_select_pred_next_act", on_change=clear_cache)

                                filter_caseid_attr_list = st.select_slider("Choose [Activity, User, Time]",
                                                                           options=filter_caseid_attr_df,
                                                                           key="caseid_attr_slider",
                                                                           on_change=clear_cache)

                                _idx = filter_caseid_attr_df.index(filter_caseid_attr_list)
                                filter_caseid_attr_list.append(_idx)

                                # Selected suffix key, Converting list to dictionary
                                filter_key_attr = ["filter_acitivity", "filter_role", "filter_time", "filter_index"]
                                filter_key_pos = [0, 1, 2, 3]
                                assert (len(filter_key_attr) == len(filter_key_pos))
                                _acc_val = 0
                                for i in range(len(filter_key_attr)):
                                    filter_caseid_attr_list.insert(filter_key_pos[i] + _acc_val, filter_key_attr[i])
                                    _acc_val += 1
                                filter_caseid_attr_dict = _list2dictConvert(filter_caseid_attr_list)
                                # print("Value of Slider :", filter_caseid_attr_dict)

                                # Passing the respective paramter to Parameters
                                parameters['nextcaseid_attr'] = filter_caseid_attr_dict
                                if (_idx + 1) < len(
                                        filter_caseid_attr_df):  # Index starts from 0 so added 1 to equate with length value
                                    _filterdf = filter_attr_display.iloc[[_idx]]
                                    _filterdf.index = [""] * len(_filterdf)
                                    state_of_theprocess.dataframe(_filterdf)
                                    st.sidebar.markdown("""---""")
                                    # if st.sidebar.button("Process", key='next_process'):
                                    if button_of_theprocess.button("Process", key='next_process'):
                                        with st.spinner(text='In progress'):
                                            predictor = pr.ModelPredictor(parameters)
                                            # print("predictor : ", predictor.acc)
                                        st.success('Done')
                                else:
                                    st.error('Reselect the Suffix to a lower Value')

                            # ----------------------------------------------------------------------------------
                            # --             Dashboard Simulation Execution Sub-Control View                -- #
                            # ----------------------------------------------------------------------------------

                            elif next_option == 'history_with_next':
                                st.experimental_set_query_params(my_saved_caseid=filter_caseid)

                                st.sidebar.markdown("""---""")
                                # next_button = st.sidebar.button("Process", key='next_process_action')
                                next_button = button_of_theprocess.button("Process", key='next_process_action')

                                _filterdf = filter_attr_display.iloc[[nxt_button_idx]]
                                _filterdf.index = [""] * len(_filterdf)
                                state_of_theprocess.dataframe(_filterdf)
                                if (next_button) and ((nxt_button_idx) < len(filter_caseid_attr_df) + 1):

                                    nxt_button_idx += 1
                                    st.experimental_set_query_params(my_saved_result=nxt_button_idx,
                                                                     my_saved_caseid=filter_caseid)  # Save value

                                    filter_caseid_attr_list = [nxt_button_idx - 1]

                                    filter_key_attr = ["filter_index"]
                                    filter_key_pos = [0]
                                    assert (len(filter_key_attr) == len(filter_key_pos))
                                    _acc_val = 0
                                    for i in range(len(filter_key_attr)):
                                        filter_caseid_attr_list.insert(filter_key_pos[i] + _acc_val, filter_key_attr[i])
                                        _acc_val += 1
                                    filter_caseid_attr_dict = _list2dictConvert(filter_caseid_attr_list)

                                    parameters['nextcaseid_attr'] = filter_caseid_attr_dict

                                    with st.spinner(text='In progress'):
                                        predictor = pr.ModelPredictor(parameters)
                                        # print("predictor : ", predictor.acc)
                                    st.success('Done')
                                    if (nxt_button_idx) >= len(filter_caseid_attr_df) + 1:
                                        # next_button.enabled = False
                                        st.experimental_set_query_params(my_saved_result=0)
                                        # ---Adding logic to skip to next caseid (Not Working)
                                        # filter_caseid_index = filter_caseid_index + 1
                                        # next_case_id_flag = 1
                                        # filter_attr_display, filter_caseid, filter_caseid_attr_df, filter_caseid_index = next_columns(filter_log, display_columns, filter_caseid_index, next_case_id_flag)
                                        st.error('End of Current Case Id, Select the Next Case ID')
                                elif ((nxt_button_idx) >= len(filter_caseid_attr_df) + 1):
                                    st.experimental_set_query_params(my_saved_result=0)  # reset value
                                    st.error('End of Current Case Id, Select the Next Case ID')

                            # ----------------------------------------------------------------------------------
                            # --                Dashboard Simulation What-If Sub-Control View               -- #
                            # ----------------------------------------------------------------------------------

                            elif next_option == 'what_if':
                                form = st.form(key="my_whatif_form")
                                form.subheader("What-IF Prediction Choose Box")
                                # # --------------------------------------------------------------------------
                                # Creating prediction selection radio button
                                choose_pred_lst = ['SME']
                                for _dx in range(parameters['multiprednum']):
                                    _dx += 1
                                    choose_pred_lst.append("Prediction " + str(_dx))
                                st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>',
                                         unsafe_allow_html=True)
                                # # --------------------------------------------------------------------------
                                # with st.expander('Choice of Prediction'):
                                form.info(
                                    "Choose the Prediction according to which System generates the next prediction, "
                                    "**SME** (Subject Matter Expert): decision solely based on users instinct and knowledge of business process about the process, "
                                    "**Prediction n** : decision solely based on the respective ranked confidence of the process")
                                wcols1, _, wcols2 = form.columns(
                                    [2, 0.25, 0.2])  # --To add prediction radio button and process button in a line
                                parameters['predchoice'] = wcols1.radio('', choose_pred_lst, key="radio_select_whatif")
                                # # --------------------------------------------------------------------------
                                st.experimental_set_query_params(my_saved_caseid=filter_caseid)
                                st.sidebar.markdown("""---""")
                                next_button = wcols2.form_submit_button("Process")

                                _filterdf = filter_attr_display.iloc[[nxt_button_idx]]
                                _filterdf.index = [""] * len(_filterdf)
                                state_of_theprocess.dataframe(_filterdf)
                                # print("Prediction Choice : ", parameters['predchoice'])

                                if (next_button) and ((nxt_button_idx) < len(filter_caseid_attr_df)):

                                    nxt_button_idx += 1

                                    st.experimental_set_query_params(my_saved_result=nxt_button_idx,
                                                                     my_saved_caseid=filter_caseid)  # Save value

                                    filter_caseid_attr_list = [nxt_button_idx - 1]

                                    filter_key_attr = ["filter_index"]
                                    filter_key_pos = [0]
                                    assert (len(filter_key_attr) == len(filter_key_pos))
                                    _acc_val = 0
                                    for i in range(len(filter_key_attr)):
                                        filter_caseid_attr_list.insert(filter_key_pos[i] + _acc_val, filter_key_attr[i])
                                        _acc_val += 1
                                    filter_caseid_attr_dict = _list2dictConvert(filter_caseid_attr_list)

                                    parameters['nextcaseid_attr'] = filter_caseid_attr_dict
                                    with st.spinner(text='In progress'):
                                        predictor = pr.ModelPredictor(parameters)
                                        # print("predictor : ", predictor.acc)
                                    st.success('Done')
                                    if (nxt_button_idx) >= len(filter_caseid_attr_df):
                                        # next_button.enabled = False
                                        st.experimental_set_query_params(my_saved_result=0)
                                        st.error('End of Current Case Id, Select the Next Case ID')
                                elif ((nxt_button_idx) >= len(filter_caseid_attr_df)):
                                    st.experimental_set_query_params(my_saved_result=0)  # reset value
                                    st.error('End of Current Case Id, Select the Next Case ID')

                        # ----------------------------------------------------------------------------------
                        # --             Dashboard Simulation Batch Mode Control View                  -- #
                        # ----------------------------------------------------------------------------------

                        elif parameters['mode'] in ['batch']:

                            _log, _, _ = read_next_testlog()
                            rolecount = min(_log.groupby('caseid')['role'].nunique())
                            actcount = min(_log.groupby('caseid')['task'].nunique())

                            # logic to count the max number of suffix which can be used for the eintire log
                            _count = _log['caseid'].value_counts().to_frame()
                            _count.reset_index(level=0, inplace=True)
                            _count.sort_values(by=['caseid'], inplace=True)
                            _log_event_select = list(set(_count['caseid'].values.tolist()))
                            # _count = (_count['caseid'].iloc[:1].values.tolist()[0]) - 1

                            with st.sidebar.expander('Select Range of Number of Events'):
                                st.info("Select the range of Number of events need to be evaluated")
                                range_event_number = st.slider('', int(min(_log_event_select)),
                                                               int(max(_log_event_select)),
                                                               (int(min(_log_event_select)),
                                                                int(max(_log_event_select))),
                                                               key="range_of_event_number", on_change=clear_cache)
                                # range_event_number = st.select_slider('', min(_log_event_select), max(_log_event_select),
                                #                                _log_event_select, key="range_of_event_number", on_change=clear_cache)

                            _log_check = _log.groupby("caseid").filter(
                                lambda x: len(x) >= min(range_event_number) and len(x) <= max(range_event_number))

                            if _log_check.empty is True:
                                st.error("The Range of Event Number doesn't have any Case Id's")
                                raise ValueError(range_event_number)
                            else:
                                parameters['batchlogrange'] = range_event_number

                                with st.sidebar.expander('Type of Batch Event Processing'):
                                    st.info(
                                        "Select **Execution Mode** for simulating the dashboard for the Users, **Evaluation Mode** to judge the trustworthiness of the ML model prediction")
                                    batch_option = st.radio('', ['Base Mode', 'Pre-Select Prefix Mode'],
                                                            key="batch_event_processing", on_change=clear_cache)

                                if batch_option == 'Base Mode':
                                    batch_option = 'base_batch'

                                elif batch_option == 'Pre-Select Prefix Mode':
                                    batch_option = 'pre_prefix'

                                if batch_option == 'base_batch':
                                    variant_opt, slider = common_batch_varient(actcount, rolecount)
                                    parameters['batchpredchoice'] = ''
                                    parameters['batchprefixnum'] = ''

                                elif batch_option == 'pre_prefix':

                                    with st.sidebar.expander('Choose Prefix Source'):
                                        st.info("Select **SME** for simulating the input to prediction using the log, "
                                                "**Prediction** to use the respective prediction as the input for subsequent prediction, "
                                                "**Generative** to generate the model predictions")
                                        parameters['batchpredchoice'] = st.radio('', ['SME', 'Prediction', 'Generative'],
                                                                                 key="radio_select_pred_batch",
                                                                                 on_change=clear_cache)

                                    with st.sidebar.expander('Select the Number of Prefix'):
                                        st.info("Prefix for each caseid in the batch mode")

                                        # -- Dynamic Slider based on number of event number range
                                        prefix_slider = st.slider(
                                            label='', min_value=0,
                                            max_value=(min(range_event_number) - 1), value=0, step=1,
                                            key='my_number_prefix_slider_batch', on_change=clear_cache)

                                        # -- Dynamic Slider Logic based on entire event log : Issue with automatic update of prefix
                                        # prefix_slider = st.select_slider(
                                        #     label='', options=list(range(min(range_event_number))),
                                        #     value=min(list(range(min(range_event_number)))),
                                        #     key='my_number_prefix_select_slider_batch', on_change=clear_cache)

                                    parameters['batchprefixnum'] = prefix_slider

                                    variant_opt, slider = common_batch_varient(actcount, rolecount)

                                parameters['multiprednum'] = slider  #
                                parameters[
                                    'variant'] = variant_opt  # random_choice, arg_max for variants and repetitions to be tested
                                parameters['batch_mode'] = batch_option
                                parameters['next_mode'] = ''
                                parameters['predchoice'] = ''
                                st.sidebar.markdown("""---""")
                                if st.sidebar.button("Process", key='batch_process'):
                                    start = time.time()
                                    with st.spinner(text='In progress'):
                                        predictor = pr.ModelPredictor(parameters)
                                    end = time.time()
                                    st.sidebar.write("Elapsed Time (in minutes) : ", (end - start) / 60)
                                    st.success('Done')

    else:
        # Catch parameters
        try:
            opts, _ = getopt.getopt(
                argv,
                "ho:a:f:i:l:d:p:n:m:z:y:c:b:x:t:e:v:r:",
                ['one_timestamp=', 'activity=',
                 'file_name=', 'imp=', 'lstm_act=',
                 'dense_act=', 'optim=', 'norm_method=',
                 'model_type=', 'n_size=', 'l_size=',
                 'folder=', 'model_file=', 'is_single_exec=',
                 'max_trace_size=', 'splits=', 'sub_group=',
                 'variant=', 'rep='])
            for opt, arg in opts:
                key = catch_parameter(opt)
                if arg in ['None', 'none']:
                    parameters[key] = None
                elif key in ['is_single_exec', 'one_timestamp']:
                    parameters[key] = arg in ['True', 'true', 1]
                elif key in ['imp', 'n_size', 'l_size',
                             'max_trace_size', 'splits', 'rep']:
                    parameters[key] = int(arg)
                else:
                    parameters[key] = arg
            parameters['read_options'] = {'timeformat': '%Y-%m-%dT%H:%M:%S.%f',
                                          'column_names': column_names,
                                          'one_timestamp':
                                              parameters['one_timestamp'],
                                          'ns_include': True}
        except getopt.GetoptError:
            print('Invalid option')
            sys.exit(2)


if __name__ == "__main__":
    main(sys.argv[1:])

# streamlit run dashboard.py
