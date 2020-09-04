#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 20:05:55 2020

@author: biran
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import xgboost as xgb
from xgboost import plot_importance
from numpy import inf
import math
from datetime import datetime
from scipy.optimize import fsolve


class TalentAttribute(object):

    def __init__(self, data_config):
        self.data_config = data_config

    def loaddata(self):
        return pd.read_csv(self.data_config['talent_attributes_file'])

    @staticmethod
    def if_manager(row, order, data_config):
        '''
        Check if the employee is a manager or IC by manager key words. In some transition period, number of direct reports cannot be used as the only indicator of if an employee is a manager or IC.
        '''
        mgr_words = data_config['mgr_words']
        # prev_job_title_nm_flag = data_config['time_dict'][str(order)][4]
        # is_manager_flag = data_config['time_dict'][str(order)][3]
        prev_job_title_nm_flag = data_config['time_dict'][order][4]
        is_manager_flag = data_config['time_dict'][order][3]
        is_manager_indicator = row[is_manager_flag]
        if any(ele in row[prev_job_title_nm_flag].lower() for ele in mgr_words):
            is_manager_indicator = 1
        else:
            is_manager_indicator = is_manager_indicator
        return is_manager_indicator

    def data_prep(self, data):
        data['current_year'] = self.data_config['current_year']
        data['current_year'] = data['current_year'].astype(int)

        data.loc[data['direct_reports'].isna(), 'direct_reports'] = 0
        data.loc[data['direct_reports'] == 0, 'org_size'] = 0

        for column in self.data_config['pipe_line_features']:
            data.loc[data[column] == 'N', column] = 0
            data.loc[data[column] == 'Y', column] = 1
            data[column] = data[column].astype(float)

        data['is_manager'] = 0
        data.loc[data.direct_reports > 0, 'is_manager'] = 1
        data['is_manager'] = data.apply(self.if_manager, args=(0, self.data_config,), axis=1)

        data['prior_level_promotion_spd'] = np.where(data.prior_job_lvl_dwell_months.isnull(), 1.0,
                                                     data.avg_prior_job_lvl_dwell_months / data.prior_job_lvl_dwell_months)
        data.loc[np.isinf(data['prior_level_promotion_spd']), 'prior_level_promotion_spd'] = max(data.loc[data['prior_level_promotion_spd'] != inf, 'prior_level_promotion_spd'])  # replace inf value with the max value
        data['promotion_spd_rank_in_level'] = data.groupby('job_lvl_nm').prior_level_promotion_spd.rank(pct=True)
        # data['promotion_spd_rank_in_level'] = data['promotion_spd_rank_in_level'] * 100.0

        data['initial_lv'] = data[['prev_job_lvl_nm_1', 'prev_job_lvl_nm_2', 'prev_job_lvl_nm_3', 'job_lvl_nm']].min(
            axis=1)

        data['group_key'] = data.job_lvl_nm.astype(str) + '-' + data.is_manager.astype(
            str) + '-' + data.job_family_nm.astype(str)

        for column in ['ov2010','ov2011','ov2012','ov2013','ov2014','ov2015','ov2016','ov2017','ov2018','ov2019','ov2020']:
             data.loc[data[column].isnull(), column] = 'Unknown'

        # for column in ['ov2010', 'ov2011', 'ov2012', 'ov2013', 'ov2014', 'ov2015', 'ov2016', 'ov2017', 'ov2018',
        #                'ov2019', 'ov2020']:
        #     talent_data.loc[talent_data[column].isnull(), column] = 'Unknown'
        group_population = data.groupby('group_key').count().iloc[:,0]
        group_population = pd.DataFrame(group_population)
        group_population.columns = ['number_of_people']
        data = data.join(group_population, on = ['group_key'], how = 'left')

        return data

    def assign_missing_prevjob_info(self, data):
        for order in range(1, 4, 1):
            # prev_job_family_nm = self.data_config['time_dict'][str(order)][2]
            # ahead_prev_job_family_nm = self.data_config['time_dict'][str(order-1)][2]
            prev_job_family_nm = self.data_config['time_dict'][order][2]
            ahead_prev_job_family_nm = self.data_config['time_dict'][order-1][2]

            data.loc[data[prev_job_family_nm].isnull(), prev_job_family_nm] = 'Unknown'

            data.loc[data[prev_job_family_nm] == 'M&A', prev_job_family_nm] = data.loc[
            data[prev_job_family_nm] == 'M&A', ahead_prev_job_family_nm]

        for column in ['tenure', 'prev_job_cd_dwelltime_months_1', 'prev_job_cd_dwelltime_months_2',
                       'prev_job_cd_dwelltime_months_3']:
            data.loc[data[column].isna(), column] = 0
        data['time_in_current_role_month'] = data['tenure'] - data['prev_job_cd_dwelltime_months_1'] - data[
            'prev_job_cd_dwelltime_months_2'] - data['prev_job_cd_dwelltime_months_3']

        data['time_in_current_role_month'] = data[['time_in_current_role_month', 'time_in_level']].min(axis=1)

    def if_previous_manager(self, data):
        for order in range(1,4,1):
            prev_job_order_nm = self.data_config['time_dict'][order][1]
            data.loc[data[prev_job_order_nm].isnull(), prev_job_order_nm] = 99

            prev_job_title = self.data_config['time_dict'][order][4]
            data.loc[data[prev_job_title].isnull(), prev_job_title] = 'Unknown'

            prev_is_manager = self.data_config['time_dict'][order][3]
            data[prev_is_manager] = 0

            is_manager_flag = self.data_config['time_dict'][order][3]
            data[is_manager_flag] = data.apply(self.if_manager, args = (order, self.data_config,), axis=1)

    @staticmethod
    def fill_missing_time_level(row):
        if math.isnan(row['time_in_level']):
            if row['prev_job_lvl_nm_3'] == row['job_lvl_nm']:
                time_in_level = row['prev_job_cd_dwelltime_months_3'] + \
                                                  row['prev_job_cd_dwelltime_months_2'] + \
                                                  row['prev_job_cd_dwelltime_months_1']
            elif row['prev_job_lvl_nm_2'] == row['job_lvl_nm']:
                time_in_level = row['prev_job_cd_dwelltime_months_2'] + \
                                                  row['prev_job_cd_dwelltime_months_1']
            elif row['prev_job_lvl_nm_1'] == row['job_lvl_nm']:
                time_in_level = row['prev_job_cd_dwelltime_months_1']
            else:
                time_in_level = row['tenure']
        else:
            time_in_level = row['time_in_level']
        return time_in_level

    def fill_time_in_position(self, data):
        data['months_since_last_transfer'] = (datetime.now() - pd.to_datetime(
            data.most_rencent_transfer_date)) / np.timedelta64(1, 'M')

        data['months_in_current_position'] = data['time_in_level']

        data.loc[~data['months_since_last_transfer'].isnull(), 'months_in_current_position'] = \
        data.loc[~data['months_since_last_transfer'].isnull()][['months_since_last_transfer', 'time_in_level']].min(
            axis=1)

    def mapping_old_jobfamily(self, data):
        '''
        Map the old job family to the corresponding new job family.
        '''
        mapping_dict = self.data_config['old_jobfamily_mapping_dict']
        for ele in mapping_dict:
            for order in range(1,4,1):
                prev_job_family_nm = self.data_config['time_dict'][order][2]
                data.loc[data[prev_job_family_nm] == ele, prev_job_family_nm] = mapping_dict[ele]

    @staticmethod
    def process_spd(data):
        '''
        Since the previous job dwell time is highly skewed, so we take the log value, and then scale it as the previous promotion speed.
        '''
        mean_prior_job_lvl_dwell_months = np.mean(data.loc[~data['prior_job_lvl_dwell_months'].isnull(),'prior_job_lvl_dwell_months'])
        data.loc[data['avg_prior_job_lvl_dwell_months'].isnull(), 'avg_prior_job_lvl_dwell_months'] = mean_prior_job_lvl_dwell_months
        data.loc[data['prior_job_lvl_dwell_months'].isnull(),'prior_job_lvl_dwell_months'] = data.loc[data['prior_job_lvl_dwell_months'].isnull(),'avg_prior_job_lvl_dwell_months']
        data['log_time'] = np.log(data['prior_job_lvl_dwell_months'] + 1)
        min_max_table = data[['group_key','log_time']].groupby(['group_key']).agg([('min_logtime' , 'min'), ('max_logtime', 'max')])
        min_max_table.columns = ['min_logtime','max_logtime']
        min_max_table['log_time_delta'] = min_max_table['max_logtime'] - min_max_table['min_logtime']
        data = data.set_index(['group_key'])
        data = data.join(min_max_table, on = (['group_key']), how = 'left')
        data['promotion_spd_scaled'] = 1-(data['log_time'] - data['min_logtime'])/data['log_time_delta']
        data.loc[data['promotion_spd_scaled'].isnull(), 'promotion_spd_scaled'] = 0.5
        data = data.reset_index(drop=False)
        return data

    def process_talent_attribute(self):
        data = self.loaddata()
        data = self.data_prep(data)
        self.if_previous_manager(data)
        data['time_in_level'] = data.apply(self.fill_missing_time_level, axis =1)
        self.assign_missing_prevjob_info(data)
        self.fill_time_in_position(data)
        self.mapping_old_jobfamily(data)
        data = self.process_spd(data)
        return data