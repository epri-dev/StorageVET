"""
Copyright (c) 2023, Electric Power Research Institute

 All rights reserved.

 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

     * Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.
     * Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.
     * Neither the name of DER-VET nor the names of its contributors
       may be used to endorse or promote products derived from this software
       without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
"""
Library.py

Library of helper functions used in StorageVET.
"""
import numpy as np
import pandas as pd

BOUND_NAMES = ['ch_max', 'ch_min', 'dis_max', 'dis_min', 'ene_max', 'ene_min']


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
    new = source*(1+rate)**years  # apply growth rate to source data
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
        new_index = pd.date_range(start=f"1/1/{int(year)}", end=f"1/1/{int(year + 1)}", freq=frequency, closed='left')
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
    data_year = df.iloc[1:].index.year.unique()  # grab all but the first index
    # which years do we not have data for
    no_data_year = {pd.Period(year) for year in years_need_data_for} - {pd.Period(year) for year in data_year}
    # if there is a year we dont have data for
    if len(no_data_year) > 0:
        for yr in no_data_year:
            source_year = pd.Period(max(data_year))  # which year to to apply growth rate to (is this the logic we want??)
            source_data = df.loc[df.index.year == source_year.year]  # use source year data

            # create new dataframe for missing year
            try:
                new_data_df = pd.DataFrame()
                for col in df.columns:
                    new_data = apply_growth(source_data[col], growth_rate, source_year, yr, frequency)  # apply growth rate to column
                    new_data_df = pd.concat([new_data_df, new_data], axis=1, sort=True)
                    # add new year to original data frame
                df = pd.concat([df, new_data_df], sort=True)
            except AttributeError:
                new_data = apply_growth(source_data, growth_rate, source_year, yr, frequency)  # apply growth rate to column
                # add new year to original data frame
                df = pd.concat([df, new_data], sort=True)
    return df


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
            df_sub = df.loc[df.index.year != yr.year]  # choose all data that is not in the unneeded year
            df = df_sub
    return df


def is_leap_yr(year):
    """ Determines whether given year is leap year or not.

    Args:
        year (int): The year in question.

    Returns:
        bool: True for it being a leap year, False if not leap year.
    """
    return year % 4 == 0 and year % 100 != 0 or year % 400 == 0


def truncate_float(number, decimals=3):
    """

    Args:
        number: float with lots of decimal values
        decimals: number of decimals to keep

    Returns: a truncated version of the NUMBER

    """
    return round(number * 10**decimals) / 10**decimals
