"""
EnergyTimeShift.py

This Python class contains methods and attributes specific for service analysis within StorageVet.
"""

__author__ = 'Miles Evans, Evan Giarta, and Halley Nathwani'
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
import logging

u_logger = logging.getLogger('User')
e_logger = logging.getLogger('Error')

SATURDAY = 5


class EnergyTimeShift(ValueStream):
    """ Retail energy time shift. A behind the meter service.

    """

    def __init__(self, params, tech, dt):
        """ Generates the objective function, finds and creates constraints.

        Args:
            params (Dict): input parameters
            tech (Technology): Storage technology object
            dt (float): optimization timestep (hours)
        """
        ValueStream.__init__(self, tech, 'retailETS', dt)
        self.p_energy = params['price']  # TODO: rename to self.price (matches other value streams)
        self.tariff = params['tariff']
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
        size = sum(mask)
        p_energy = cvx.Parameter(size, value=self.p_energy.loc[mask].values, name='energy_price')

        load_p = p_energy * load  # TODO: make sure element-wise multiplication
        discharge_p = p_energy * variables['dis']
        charge_p = p_energy * variables['ch']
        generation_p = p_energy * generation

        # self.costs.append(cvx.sum(load_penergy - discharge_penergy + charge_penergy - generation_penergy)*annuity_scalar)
        return {self.name: cvx.sum(load_p - discharge_p + charge_p - generation_p) * self.dt * annuity_scalar}

    def estimate_year_data(self, years, frequency):
        """ Update variable that hold timeseries data after adding growth data. These method should be called after
        add_growth_data and before the optimization is run.

        Args:
            years (List): list of years for which analysis will occur on
            frequency (str): period frequency of the timeseries data

        """
        data_year = self.p_energy.index.year.unique()
        no_data_year = {pd.Period(year) for year in years} - {pd.Period(year) for year in data_year}  # which years do we not have data for

        if len(no_data_year) > 0:
            for yr in no_data_year:
                source_year = pd.Period(max(data_year))

                years = yr.year - source_year.year

                first_day = '1/1/' + str(yr.year)
                last_day = '1/1/' + str(yr.year+1)

                new_index = pd.date_range(start=first_day, end=last_day, freq=frequency, closed='left')
                size = new_index.size

                # Build Energy Price Vector based on the new year
                temp = pd.DataFrame(index=new_index)
                temp['weekday'] = (new_index.weekday < SATURDAY).astype('int64')
                temp['he'] = (new_index + pd.Timedelta('1s')).hour + 1
                temp['hour fraction'] = temp.index.minute / 60
                temp['he fraction'] = temp['he'] + temp['hour fraction']
                temp['p_energy'] = np.zeros(size)

                for p in range(len(self.tariff)):
                    # edit the pricedf energy price and period values for all of the periods defined
                    # in the tariff input file
                    bill = self.tariff.iloc[p, :]

                    month_mask = (bill["Start Month"] <= temp.index.month) & (temp.index.month <= bill["End Month"])
                    time_mask = (bill['Start Time'] <= temp['he']) & (temp['he'] <= bill['End Time'])
                    weekday_mask = True
                    exclud_mask = False

                    if not bill['Weekday?'] == 2:  # if not (apply to weekends and weekdays)
                        weekday_mask = bill['Weekday?'] == temp['weekday']
                    if not np.isnan(bill['Excluding Start Time']) and not np.isnan(bill['Excluding End Time']):
                        exclud_mask = (bill['Excluding Start Time'] <= temp['he']) & (temp['he'] <= bill['Excluding End Time'])
                    mask = np.array(month_mask & time_mask & np.logical_not(exclud_mask) & weekday_mask)

                    current_energy_prices = temp.loc[mask, 'p_energy'].values
                    if np.any(np.greater(current_energy_prices, 0)):
                        # More than one energy price applies to the same time step
                        u_logger.warning('More than one energy price applies to the same time step.')
                    # Add energy prices
                    temp.loc[mask, 'p_energy'] += bill['Value']
                # apply growth to new energy rate
                new_p_energy = temp['p_energy']*(1+self.growth/100)**years
                self.p_energy = pd.concat([self.p_energy, new_p_energy], sort=True)  # add to existing

    # def monthly_bill_reports(self, results):
    #     """ Calculates energy charges per month and adds them to the simple monthly bill dataframe
    #
    #     Args:
    #         results (DataFrame): the concatenated timeseries reports from the active DERs
    #
    #     Returns: A dataframe with the demand charges for the simple monthly bill and a dataframe for the advanced monthly bill.
    #
    #     """
    #     net_loads = results.loc[:, ['Net Load (kW)', 'Original Net Load (kW)']]
    #     net_loads.loc[:, 'he'] = net_loads.index.hour + 1  # TODO: double check this -HN
    #     net_loads.loc[:, 'yr_mo'] = net_loads.index.to_period('M')
    #     net_loads.loc[:, 'hour fraction'] = net_loads.index.minute / 60
    #     net_loads.loc[:, 'he fraction'] = net_loads['he'] + net_loads['hour fraction']
    #
    #     # calculate energy cost by month
    #     net_loads.loc[:, 'Energy Charge ($)'] = self.dt * np.multiply(net_loads.loc[:, 'Net Load (kW)'], self.p_energy)
    #     net_loads.loc[:, 'Original Energy Charge ($)'] = self.dt * np.multiply(net_loads.loc[:, 'Original Net Load (kW)'], self.p_energy)
    #
    #     sim_monthly_bill = net_loads.groupby(by=['yr_mo'])[['Energy Charge ($)', 'Original Energy Charge ($)']].sum()
    #     sim_monthly_bill.index.name = 'Month-Year'
    #     self.monthly_bill = sim_monthly_bill
    #
    #     # Calculate energy charge per billing period: since overlapping energy charges are added,
    #     # we must calculate energy charges the "long way"
    #     monthly_bill = pd.DataFrame()
    #     for item in self.tariff.index:
    #         # determine subset of data that
    #         bill = self.tariff.loc[item, :]
    #         month_mask = (bill["Start Month"] <= net_loads.index.month) & (net_loads.index.month <= bill["End Month"])
    #         time_mask = (bill['Start Time'] <= net_loads['he fraction']) & (net_loads['he fraction'] <= bill['End Time'])
    #         weekday_mask = True
    #         exclud_mask = False
    #         if not bill['Weekday?'] == 2:  # if not (apply to weekends and weekdays)
    #             weekday_mask = bill['Weekday?'] == (net_loads.index.weekday < SATURDAY).astype('int64')
    #         if not np.isnan(bill['Excluding Start Time']) and not np.isnan(bill['Excluding End Time']):
    #             exclud_mask = (bill['Excluding Start Time'] <= net_loads['he fraction']) & (net_loads['he fraction'] <= bill['Excluding End Time'])
    #         mask = np.array(month_mask & time_mask & np.logical_not(exclud_mask) & weekday_mask)
    #         temp_df = net_loads.loc[mask, :].copy()
    #
    #         # Add energy prices
    #         energy_price = bill['Value']
    #
    #         # calculate energy cost by month (within billing period)
    #         temp_df.loc[:, 'Energy Charge ($)'] = self.dt * energy_price * temp_df.loc[:, 'Net Load (kW)']
    #         temp_df.loc[:, 'Original Energy Charge ($)'] = self.dt * energy_price * temp_df.loc[:, 'Original Net Load (kW)']
    #         retail_period = temp_df.groupby(by=['yr_mo'])[['Energy Charge ($)', 'Original Energy Charge ($)']].sum()
    #
    #         # add billing period column to df
    #         retail_period.loc[:, 'Billing Period'] = item
    #         monthly_bill = monthly_bill.append(retail_period)
    #
    #     adv_monthly_bill = monthly_bill.sort_index(axis=0)
    #     adv_monthly_bill.index.name = 'Month-Year'
    #     self.billing_period_bill = adv_monthly_bill
    #
    #     return adv_monthly_bill, sim_monthly_bill

    def timeseries_report(self):
        """ Summaries the optimization results for this Value Stream.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        report = pd.DataFrame(index=self.p_energy.index)
        report.loc[:, 'Energy Price ($/kWh)'] = self.p_energy
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
    #     proforma.columns = ['Avoided Energy Charge']
    #
    #     for year in opt_years:
    #         monthly_energy_bill = self.monthly_bill
    #         year_monthly = monthly_energy_bill[monthly_energy_bill.index.year == year]
    #
    #         new_energy_charge = year_monthly['Energy Charge ($)'].values.sum()
    #         orig_energy_charge = year_monthly['Original Energy Charge ($)'].values.sum()
    #         avoided_cost = orig_energy_charge - new_energy_charge
    #         proforma.loc[pd.Period(year=year, freq='y'), 'Avoided Energy Charge'] = avoided_cost
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
    #     self.p_energy = tariff_price_signal.loc[:, 'p_energy']
    #     self.tariff = tariff.loc[tariff.Charge.apply((lambda x: x.lower())) == 'energy', :]
