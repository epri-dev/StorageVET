"""
Library.py

Library of helper functions used in StorageVET.
"""

__author__ = 'Halley Nathwani, Micah Botkin-Levy, Miles Evans and Evan Giarta'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Micah Botkin-Levy']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Evan Giarta', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'egiarta@epri.com', 'mevans@epri.com']
__version__ = '2.1.1.1'

import numpy as np
import pandas as pd
import copy

BOUND_NAMES = ['ch_max', 'ch_min', 'dis_max', 'dis_min', 'ene_max', 'ene_min']
# bound_names = ['ch_max', 'ch_min', 'dis_max', 'dis_min', 'ene_max', 'ene_min']
fr_obj_names = ['regu_d_cap', 'regu_c_cap', 'regd_d_cap', 'regd_c_cap', 'regu_d_ene', 'regu_c_ene', 'regd_d_ene', 'regd_c_ene']


def update_df(df1, df2):
    """ Helper function: Updates elements of df1 based on df2. Will add new columns if not in df1 or insert elements at
    the corresponding index if existing column

    Args:
        df1 (Data Frame): original data frame to be editted
        df2 (Data Frame): data frame to be added

    Returns:
        df1 (Data Frame)
    """

    old_col = set(df2.columns).intersection(set(df1.columns))
    df1 = df1.join(df2[list(set(df2.columns).difference(old_col))], how='left')  # join new columns
    df1.update(df2[list(old_col)])  # update old columns
    return df1


def disagg_col(df, group, col):
    """ Helper function: Adds a disaggregated column of 'col' based on the count of group
    TEMP FUNCTION: assumes that column is merged into disagg dataframe at original resolution = Bad approach

    Args:
        df (Data Frame): original data frame to be
        group (list): columns to group on
        col (string): column to disagg

    Returns:
        df (Data Frame)


    """
    # TODO this is a temp helper function until a more robust disagg function is built

    count_df = df.groupby(by=group).size()
    count_df.name = 'counts'
    df = df.reset_index().merge(count_df.reset_index(), on=group, how='left').set_index(df.index.names)
    df[col+'_disagg'] = df[col] / df['counts']
    return df


def apply_growth(source, rate, source_year, yr, freq):
    """ Applies linear growth rate to determine data for future year

    Args:
        source (Series): given data
        rate (float): yearly growth rate (%)
        source_year (Period): given data year
        yr (Period): future year to get data for
        freq (str): simulation time step frequency

    Returns:
        new (Series)
    """
    years = yr.year - source_year.year  # difference in years between source and desired yea
    new = source*(1+rate/100)**years  # apply growth rate to source data
    # new.index = new.index + pd.DateOffset(years=1)
    # deal with leap years
    source_leap = is_leap_yr(source_year.year)
    new_leap = is_leap_yr(yr.year)

    if (not source_leap) and new_leap:   # need to add leap day
        # if source is not leap year but desired year is, copy data from previous day
        new.index = new.index + pd.DateOffset(years=years)
        leap_ind = pd.date_range(start='02/29/'+str(yr), end='03/01/'+str(yr), freq=freq, closed='left')
        leap = pd.Series(new[leap_ind - pd.DateOffset(days=1)].values, index=leap_ind, name=new.name)
        new = pd.concat([new, leap])
        new = new.sort_index()
    elif source_leap and (not new_leap):  # need to remove leap day
        leap_ind = pd.date_range(start='02/29/'+str(source_year), end='03/01/'+str(source_year), freq=freq, closed='left')
        new = new[~new.index.isin(leap_ind)]
        new.index = new.index + pd.DateOffset(years=years)
    else:
        new.index = new.index + pd.DateOffset(years=years)
    return new


def create_timeseries_index(years, frequency):
    """ Creates the template for the timeseries index internal to the program that is Hour Begining

    Args:
        years (list): list of years that should be included in the returned Index
        frequency (str): the pandas frequency in string representation -- required to create dateTime range

    Returns: an empty DataFrame with the index beginning at hour 0

    """
    temp_master_df = pd.DataFrame()
    years = np.sort(years)
    for year in years:
        new_index = pd.date_range(start='1/1/' + str(year), end='1/1/' + str(year + 1), freq=frequency, closed='left')
        temp_df = pd.DataFrame(index=new_index)

        # add new year to original data frame
        temp_master_df = pd.concat([temp_master_df, temp_df], sort=True)
    temp_master_df.index.name = 'Start Datetime (hb)'
    return temp_master_df.index


def fill_extra_data(df, years_need_data_for, growth_rate, frequency):
    """ Extends timeseries data with missing years of data estimated with given GROWTH_RATE

    Args:
        df (DataFrame): DataFrame for which to apply the following
        years_need_data_for (list): years that need to be in the index of the DF
        growth_rate (float, int): the rate at which the data growths as time goes on
        frequency (str): the pandas frequency in string representation -- required to create dateTime range

    Returns:

    """
    data_year = df.iloc[1:, :].index.year.unique()  # grab all but the first index
    # which years do we not have data for
    no_data_year = {pd.Period(year) for year in years_need_data_for} - {pd.Period(year) for year in data_year}
    # if there is a year we dont have data for
    if len(no_data_year) > 0:
        for yr in no_data_year:
            source_year = pd.Period(max(data_year))  # which year to to apply growth rate to (is this the logic we want??)
            source_data = df[df.index.year == source_year.year]  # use source year data

            # create new dataframe for missing year
            new_data = pd.DataFrame()
            for col in df.columns:
                new_data = apply_growth(source_data[col], growth_rate, source_year, yr, frequency)  # apply growth rate to column

            # add new year to original data frame
            df = pd.concat([df, new_data], sort=True)


def drop_extra_data(df, years_need_data_for):
    """ Remove any data that is not specified by YEARS_NEED_DATA_FOR

    Args:
        df:
        years_need_data_for:

    Returns:

    """
    data_year = df.index.year.unique()  # which years was data given for
    # which years is data given for that is not needed
    dont_need_year = {pd.Period(year) for year in data_year} - {pd.Period(year) for year in years_need_data_for}
    if len(dont_need_year) > 0:
        for yr in dont_need_year:
            df_sub = df[df.index.year != yr.year]  # choose all data that is not in the unneeded year
            df = df_sub


def is_leap_yr(year):
    """ Determines whether given year is leap year or not.

    Args:
        year (int): The year in question.

    Returns:
        bool: True for it being a leap year, False if not leap year.
    """
    return year % 4 == 0 and year % 100 != 0 or year % 400 == 0
