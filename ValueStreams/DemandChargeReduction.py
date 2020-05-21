"""
DemandChargeReduction.py

This Python class contains methods and attributes specific for service analysis within StorageVet.
"""

__author__ = 'Miles Evans, Evan Giarta and Halley Nathwani'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Micah Botkin-Levy']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Evan Giarta', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'egiarta@epri.com', 'mevans@epri.com']
__version__ = '2.1.1.1'

from .ValueStream import ValueStream
import numpy as np
import cvxpy as cvx
import pandas as pd
import sys
import logging
import time

SATURDAY = 5

u_logger = logging.getLogger('User')
e_logger = logging.getLogger('Error')


class DemandChargeReduction(ValueStream):
    """ Retail demand charge reduction. A behind the meter service.

    """

    def __init__(self, params, tech, dt):
        """ Generates the objective function, finds and creates constraints.

        Args:
            params (Dict): input parameters
            tech (Technology): Storage technology object
            dt (float): optimization timestep (hours)
        """
        ValueStream.__init__(self, tech, 'DCM', dt)
        # self.demand_rate = params['rate']
        self.tariff = params['tariff']
        self.billing_period = params['billing_period']
        self.growth = params['growth']

        self.billing_period_bill = pd.DataFrame()
        self.monthly_bill = pd.DataFrame()

    def objective_function(self, variables, mask, load, generation, annuity_scalar=1):
        """ Generates the full objective function, including the optimization variables.

        Args:
            variables (Dict): dictionary of variables being optimized
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set
            load (list, Expression): the sum of load within the system
            generation (list, Expression): the sum of generation within the system
            annuity_scalar (float): a scalar value to be multiplied by any yearly cost or benefit that helps capture the cost/benefit over
                        the entire project lifetime (only to be set iff sizing)

        Returns:
            The expression of the objective function that it affects. This can be passed into the cvxpy solver.

        """
        start = time.time()
        total_demand_charges = 0
        net_load = load - variables['dis'] + variables['ch'] - generation  # this is the demand that still needs to be met

        sub_billing_period = self.billing_period.loc[mask]
        # determine and add demand charges monthly
        months = sub_billing_period.index.to_period('M')
        for mo in months.unique():
            # array of booleans; true for the selected month
            monthly_mask = (sub_billing_period.index.month == mo.month)

            # select the month's billing period data
            month_sub_billing_period = sub_billing_period.loc[monthly_mask]

            # set of unique billing periods in the selected month
            pset = {int(item) for sublist in month_sub_billing_period for item in sublist}

            # determine the index what has the first True value in the array of booleans
            # (the index of the timestep that corresponds to day 1 and hour 0 of the month)
            first_true = np.nonzero(monthly_mask)[0][0]

            for per in pset:
                # Add demand charge calculation for each applicable billing period (PER) within the selected month

                # get an array that is True only for the selected month
                billing_per_mask = monthly_mask.copy()

                for i in range(first_true, first_true+len(month_sub_billing_period)):
                    # loop through only the values that are 'True' (which should all be in order because
                    # the array should be sorted by datetime index) as they represent a month

                    # reassign the value at I to be whether PER applies to the time corresponding to I
                    # (reassign each 'True' value to 'False' if the billing period does not apply to that timestep)
                    billing_per_mask[i] = per in sub_billing_period.iloc[i]

                # add a demand charge for each billing period in a month (for every month being optimized)
                if np.all(billing_per_mask):
                    total_demand_charges += self.tariff.loc[per, 'Value'] * annuity_scalar * cvx.max(net_load)
                else:
                    total_demand_charges += self.tariff.loc[per, 'Value'] * annuity_scalar * cvx.max(net_load[billing_per_mask])
        # print(f'time took to make demand charge term: {time.time() - start}')
        return {self.name: total_demand_charges}

    def estimate_year_data(self, years, frequency):
        """ Update variable that hold timeseries data after adding growth data. These method should be called after
        add_growth_data and before the optimization is run.

        Args:
            years (List): list of years for which analysis will occur on
            frequency (str): period frequency of the timeseries data

        This function adds billing periods to the tariff that match the given year's structure, but the values have
        a growth rate applied to them. Then it lists them within self.billing_period.

        """
        data_year = self.billing_period.index.year.unique()
        no_data_year = {pd.Period(year) for year in years} - {pd.Period(year) for year in data_year}  # which years do we not have data for

        if len(no_data_year) > 0:
            for yr in no_data_year:
                source_year = pd.Period(max(data_year))

                years = yr.year - source_year.year

                first_day = '1/1/' + str(yr.year)
                last_day = '1/1/' + str(yr.year + 1)

                new_index = pd.date_range(start=first_day, end=last_day, freq=frequency, closed='left')
                size = new_index.size

                # make new tariff with charges that have increase with user-defined growth rate
                add_tariff = self.tariff.reset_index()
                add_tariff.loc[:, 'Value'] = self.tariff['Value'].values*(1+self.growth/100)**years
                add_tariff.loc[:, 'Billing Period'] = self.tariff.index + self.tariff.index.max()
                add_tariff = add_tariff.set_index('Billing Period', drop=True)
                # Build Energy Price Vector based on the new year
                temp = pd.DataFrame(index=new_index)
                temp['weekday'] = (new_index.weekday < SATURDAY).astype('int64')
                temp['he'] = (new_index + pd.Timedelta('1s')).hour + 1

                billing_period = [[] for _ in range(size)]

                for p in add_tariff.index:
                    # edit the pricedf energy price and period values for all of the periods defined
                    # in the tariff input file
                    bill = add_tariff.loc[p, :]
                    month_mask = (bill["Start Month"] <= temp.index.month) & (temp.index.month <= bill["End Month"])
                    time_mask = (bill['Start Time'] <= temp['he']) & (temp['he'] <= bill['End Time'])
                    weekday_mask = True
                    exclud_mask = False
                    if not bill['Weekday?'] == 2:  # if not (apply to weekends and weekdays)
                        weekday_mask = bill['Weekday?'] == temp['weekday']
                    if not np.isnan(bill['Excluding Start Time']) and not np.isnan(bill['Excluding End Time']):
                        exclud_mask = (bill['Excluding Start Time'] <= temp['he']) & (temp['he'] <= bill['Excluding End Time'])
                    mask = np.array(month_mask & time_mask & np.logical_not(exclud_mask) & weekday_mask)
                    if bill['Charge'].lower() == 'demand':
                        for i, true_false in enumerate(mask):
                            if true_false:
                                billing_period[i].append(p)
                billing_period = pd.Series(billing_period, dtype='object', index=temp.index)
                # temp.loc[:, 'billing_period'] = billing_period.values

                # ADD CHECK TO MAKE SURE ENERGY PRICES ARE THE SAME FOR EACH OVERLAPPING BILLING PERIOD
                # Check to see that each timestep has a period assigned to it
                if not billing_period.apply(len).all():
                    u_logger.error('The billing periods in the input file do not partition the year')
                    u_logger.error('please check the tariff input file')
                    e_logger.error('The billing periods in the input file do not partition the year')
                    e_logger.error('please check the tariff input file')
                    u_logger.error('Error with tariff input file!')
                    e_logger.error('Error with tariff input file!')
                    sys.exit()
                self.tariff = pd.concat([self.tariff, add_tariff], sort=True)
                self.billing_period = pd.concat([self.billing_period, billing_period], sort=True)

    # def monthly_bill_reports(self, results):
    #     """ Calculates demand charges per month and per billing period. The month-year is the index, while the billing
    #     period is an additional column.
    #
    #     Args:
    #         results (DataFrame): the concatenated timeseries reports from the active DERs
    #
    #     Returns: A dataframe with the demand charges for the simple monthly bill and a dataframe for the advanced monthly bill.
    #         Costs are positive
    #
    #     """
    #     net_loads = results.loc[:, ['Net Load (kW)', 'Original Net Load (kW)']]
    #     net_loads.loc[:, 'yr_mo'] = net_loads.index.to_period('M')
    #
    #     # if another year of data is added, there isnt a year column to denote the year it applies to, so we use the
    #     # self.billing_period values to determine which billing period applies
    #
    #     monthly_bill = pd.DataFrame()
    #
    #     # if mask contains more than a month, then determine dcterm monthly
    #     for year_month in net_loads['yr_mo'].unique():
    #         # array of booleans; true for the selected month (size = subs.size)
    #         monthly_mask = (self.billing_period.index.month == year_month.month) & (
    #                 self.billing_period.index.year == year_month.year)
    #
    #         # select the month's billing period data (size = days in month * 24 / dt)
    #         month_billing_period = self.billing_period.loc[monthly_mask]
    #
    #         # set of unique billing periods in the selected month
    #         pset = {int(item) for sublist in month_billing_period for item in sublist}
    #
    #         # determine the index what has the first True value in the array of booleans
    #         first_true = np.nonzero(monthly_mask)[0][0]
    #
    #         for per in pset:
    #             # Add demand charge calculation for each applicable billing period within the selected month
    #             billing_per_mask = monthly_mask.copy()
    #
    #             for i in range(first_true, first_true + len(month_billing_period)):
    #                 billing_per_mask[i] = per in self.billing_period.iloc[i]
    #
    #             billing_per_net = net_loads.loc[billing_per_mask, :]
    #
    #             # group demand charges by month
    #             demand = billing_per_net.groupby(by=['yr_mo'])[['Net Load (kW)', 'Original Net Load (kW)']].max() * self.tariff.loc[per, 'Value']
    #             demand.columns = pd.Index(['Demand Charge ($)', 'Original Demand Charge ($)'])
    #
    #             # create and append a column to denote the billing period
    #             billing_pd = pd.Series(np.repeat(per, len(demand)), name='Billing Period', index=demand.index)
    #
    #             temp_bill = pd.concat([demand, billing_pd], sort=False, axis=1)
    #             monthly_bill = monthly_bill.append(temp_bill)
    #     monthly_bill = monthly_bill.sort_index(axis=0)
    #     adv_monthly_bill = monthly_bill
    #     adv_monthly_bill.index.name = 'Month-Year'
    #     self.billing_period_bill = adv_monthly_bill
    #
    #     # add all demand charges that apply to the same month
    #     sim_monthly_bill = monthly_bill.groupby(monthly_bill.index.name)[['Demand Charge ($)', 'Original Demand Charge ($)']].sum()
    #     for month_yr_index in monthly_bill.index.unique():
    #         mo_yr_data = monthly_bill.loc[month_yr_index, :]
    #         if mo_yr_data.ndim > 1:
    #             billing_periods = ', '.join(str(int(period)) for period in mo_yr_data['Billing Period'].values)
    #         else:
    #             billing_periods = str(int(mo_yr_data['Billing Period']))
    #         sim_monthly_bill.loc[month_yr_index, 'Billing Period'] = '[' + billing_periods + ']'
    #     sim_monthly_bill.index.name = 'Month-Year'
    #     self.monthly_bill = sim_monthly_bill
    #
    #     return adv_monthly_bill, sim_monthly_bill

    def timeseries_report(self):
        """ Summaries the optimization results for this Value Stream.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        report = pd.DataFrame(index=self.billing_period.index)
        report.loc[:, 'Demand Charge Billing Periods'] = self.billing_period
        return report

    # def proforma_report(self, opt_years, results):
    #     """ Calculates the proforma that corresponds to participation in this value stream
    #
    #     Args:
    #         opt_years (list): list of years the optimization problem ran for
    #         results (DataFrame): DataFrame with all the optimization variable solutions
    #
    #     Returns: A tuple of a DateFrame (of with each year in opt_year as the index and the corresponding
    #     value this stream provided), a list (of columns that remain zero), and a list (of columns that
    #     retain a constant value over the entire project horizon).
    #
    #         Creates a dataframe with only the years that we have data for. Since we do not label the column,
    #         it defaults to number the columns with a RangeIndex (starting at 0) therefore, the following
    #         DataFrame has only one column, labeled by the int 0
    #
    #     """
    #     proforma, _, _ = ValueStream.proforma_report(self, opt_years, results)
    #     proforma.columns = ['Avoided Demand Charge']
    #
    #     # TODO: there should be a check to make sure the commitments where actually met before including it --HN
    #     for year in opt_years:
    #         billing_pd_charges = self.billing_period_bill
    #         year_monthly = billing_pd_charges[billing_pd_charges.index.year == year]
    #
    #         new_demand_charge = year_monthly['Demand Charge ($)'].values.sum()
    #         orig_demand_charge = year_monthly['Original Demand Charge ($)'].values.sum()
    #         avoided_cost = orig_demand_charge - new_demand_charge
    #         proforma.loc[pd.Period(year=year, freq='y'), 'Avoided Demand Charge'] = avoided_cost
    #
    #     return proforma, None, None

    # def update_tariff_rate(self, tariff, tariff_price_signal):
    #     """ Updates attributes related to the tariff rate with the provided tariff rate. If there is no
    #     tariff provided, then nothing is updated.
    #
    #     Args:
    #         tariff (DataFrame, None): raw tariff file (as read directly from the user given CSV)
    #         tariff_price_signal (DataFrame, None): time series form of the tariff -- contains the price for energy
    #             and the billing period(s) that time-step applies to
    #
    #     """
    #     self.billing_period = tariff_price_signal.loc[:, 'billing_period']
    #     self.tariff = tariff.loc[tariff.Charge.apply((lambda x: x.lower())) == 'demand', :]
