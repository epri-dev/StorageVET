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
Finances.py

This Python class contains methods and attributes vital for completing financial analysis given optimal dispathc.
"""

import pandas as pd
import numpy as np
import numpy_financial as npf
import storagevet.Library as Lib
from storagevet.ErrorHandling import *


SATURDAY = 5


class Financial:

    def __init__(self, params, start_year, end_year):
        """ Initialized Financial object for case

         Args:
            params (Dict): input parameters
        """

        # assign important financial attributes
        self.tariff = params['customer_tariff']
        self.customer_sided = params['customer_sided']

        self.mpc = params['mpc']
        self.dt = params['dt']
        self.n = params['n']
        self.inflation_rate = params['inflation_rate']/100
        self.npv_discount_rate = params['npv_discount_rate']/100
        self.growth_rates = {'default': params['def_growth']}
        self.frequency = params['frequency']  # we assume that this is the same as the load data because loaded from the same time_series
        self.verbose = params['verbose']
        self.external_incentives = params['external_incentives']
        self.yearly_data = params['yearly_data']

        # fuel cost attributes
        self.fuel_price_liquid = params['fuel_price_liquid']  # $/MMBtu
        self.fuel_price_gas = params['fuel_price_gas']  # $/MMBtu
        self.fuel_price_other = params['fuel_price_other']  # $/MMBtu

        # attributes shared with scenario and results
        self.start_year = start_year
        self.end_year = end_year

        # prep outputs
        self.fin_summary = pd.DataFrame()  # this is just the objective values evaluated at their minimum DO NOT REPORT TO USER -- FOR DEV
        self.pro_forma = pd.DataFrame()
        self.npv = pd.DataFrame()
        self.cost_benefit = pd.DataFrame()
        self.payback = pd.DataFrame()
        self.monthly_financials_data = pd.DataFrame()
        self.billing_period_bill = pd.DataFrame()
        self.monthly_bill = pd.DataFrame()

    def calculate(self, technologies, value_streams, results, opt_years):
        """ this function calculates the proforma, cost-benefit, npv, and payback using the optimization variable results
        saved in results and the set of technology and service instances that have (if any) values that the user indicated
        they wanted to use when evaluating the CBA.

        Instead of using the technologies and services as they are passed in from the call in the Results class, we will pass
        the technologies and services with the values the user denoted to be used for evaluating the CBA.

        Args:
            technologies (list): Dict of technologies (needed to get capital and om costs)
            value_streams (Dict): Dict of all services to calculate cost avoided or profit
            results (DataFrame): DataFrame of all the concatenated timseries_report() method results from each DER
                and ValueStream
            opt_years (list)

        """
        if self.customer_sided:
            original_load = results.get('Total Original Load (kW)', results.get('Total Load (kW)'))
            self.customer_bill(self.tariff, original_load, results.loc[:, 'Net Load (kW)'],
                               value_streams)
        # # get list of all object values
        # tech_list = [item for sublist in [list(instance_dict.values()) for instance_dict in technologies.values()] for item in sublist]

        proforma = self.proforma_report(technologies, value_streams, results, opt_years)
        self.cost_benefit_report(proforma)
        self.net_present_value_report(proforma)
        self.payback_report(technologies, proforma, opt_years)
        proforma.index.name = 'Year'
        self.pro_forma = proforma

    def customer_bill(self, tariff, base_load, net_load, valuestreams):
        """ Calculates the demand and energy charges based on the provided tariff.

        Args:
            tariff (pd.DataFrame): The full tariff provided, with energy and demand
            base_load (pd.DataFrame): The original load BTM, without any DERs connected on the customer's side
            net_load (pd.DataFrame):  The net load that the meter sees as a result of adding DERs behind the meter
            valuestreams (dict)

        Returns: 2 forms of the bill-- 1) a monthly index dataframe with a column for demand
            charges and a column for energy charges; 2) a billing period indexed dataframe with
            a column for demand charges and a column for energy charges

        """
        # 1) GROW TARIFF
        # collect growth rates to apply to charges
        charge_growth_rates = {
            'demand': valuestreams['DCM'].growth if 'DCM' in valuestreams.keys() else 0,
            'energy': valuestreams['retailTimeShift'].growth if 'retailTimeShift'
                                                                in valuestreams.keys() else 0
        }

        # get optimization years
        analysis_years = base_load.index.year.unique()
        base_year = min(analysis_years)
        # add rows for the tariff, one set of new rows per optimized year (make sure to record
        # the year)
        tariff_year = tariff.copy(deep=True)
        # add "Year" column
        tariff_year["Year"] = base_year
        for yr in analysis_years:
            if yr == base_year:
                continue
            change_in_yrs = yr - base_year
            # move index values into dataframe value
            add_tariff = tariff.reset_index()
            # add "Year" column
            add_tariff["Year"] = yr
            # iterate through each row
            for i, row in add_tariff.iterrows():
                # grow the value based on the corresponding growth rate
                charge = row.Charge.lower()
                growth = charge_growth_rates[charge]
                add_tariff.loc[i, 'Value'] = row['Value'] * (1 + growth / 100) ** change_in_yrs
            # give the new row new billing period labels
            add_tariff.loc[:, 'Billing Period'] = tariff.index + tariff_year.index.max()
            add_tariff = add_tariff.set_index('Billing Period', drop=True)
            # add the new tariff to the rest
            tariff_year = pd.concat([tariff_year, add_tariff], axis=0)

        he = base_load.index.hour + 1
        month = base_load.index.month
        year = base_load.index.year

        # 2) Calculate energy charge per billing period: since overlapping energy charges are
        # added, we must calculate energy charges the "long way"
        monthly_bill = pd.DataFrame()
        for item in tariff_year.index:
            # determine mask to select subset of data that applies to the selected billing period
            bill = tariff_year.loc[item, :]
            billing_pd_mask = self.create_bill_period_mask(bill, month, he,
                                                           base_load.index.weekday, year)

            temp_df = pd.DataFrame()
            # determine if energy charge or demand charge
            if bill['Charge'].lower() == 'energy':
                # Add energy prices
                energy_price = bill['Value']
                # calculate energy cost by month (within billing period)
                temp_df['Energy Charge ($)'] = self.dt * energy_price * net_load.loc[billing_pd_mask]
                temp_df['Original Energy Charge ($)'] = self.dt * energy_price * base_load.loc[billing_pd_mask]
                retail_period = temp_df.groupby(by=lambda x: x.to_period('M'))[['Energy Charge ($)', 'Original Energy Charge ($)']].sum()
                # add billing period column to df
                retail_period.loc[:, 'Billing Period'] = item
                monthly_bill = monthly_bill.append(retail_period, sort=False)
            elif bill['Charge'].lower() == 'demand':
                # Add demand prices
                demand_price = bill['Value']
                # calculate energy cost by month (within billing period)
                temp_df['Demand Charge ($)'] = net_load.loc[billing_pd_mask]
                temp_df['Original Demand Charge ($)'] = base_load.loc[billing_pd_mask]
                retail_period = temp_df.groupby(by=lambda x: x.to_period('M'))[['Demand Charge ($)', 'Original Demand Charge ($)']].max() * demand_price
                # add billing period column to df
                retail_period.loc[:, 'Billing Period'] = item
                monthly_bill = monthly_bill.append(retail_period, sort=False)
        adv_monthly_bill = monthly_bill.sort_index(axis=0)
        adv_monthly_bill.index.name = 'Month-Year'
        adv_monthly_bill.fillna(0)
        self.billing_period_bill = adv_monthly_bill

        # 3) sum each billing period that applies to the same month
        # add all demand charges that apply to the same month
        sim_monthly_bill = monthly_bill.groupby(level=0).sum()
        for month_yr_index in monthly_bill.index.unique():
            mo_yr_data = monthly_bill.loc[month_yr_index, :]

            try:
                sim_monthly_bill.loc[month_yr_index, 'Billing Period'] = f"{mo_yr_data['Billing Period'].values}"
            except AttributeError:
                sim_monthly_bill.loc[month_yr_index, 'Billing Period'] = "['{}']".format(mo_yr_data['Billing Period'])


        sim_monthly_bill.index.name = 'Month-Year'
        self.monthly_bill = sim_monthly_bill

        return adv_monthly_bill, sim_monthly_bill

    @staticmethod
    def create_bill_period_mask(tariff_row, month, he_minute, weekday, year=None):
        """ Given a row and information about the index of that data frame, create a pd.Series of
        booleans that indicate where the billing period (described by the TARIFF_ROW) applies

        Args:
            tariff_row:
            month:
            he_minute:
            weekday:
            year (pd.Series): option input, tariff_row should have a "Year" columns that the
            tariff applies to

        Returns: pd.Series of booleans

        """
        if tariff_row.ndim != 1:
            TellUser.error('Billing Periods must be unique, '
                           + 'please check the tariff input file')
            raise TariffError('Please check the retail tariff')
        month_mask = (tariff_row["Start Month"] <= month) & (month <= tariff_row["End Month"])
        time_mask = (tariff_row['Start Time'] <= he_minute) & (he_minute <= tariff_row['End Time'])
        weekday_mask = True
        exclud_mask = False
        if not tariff_row['Weekday?'] == 2:  # if not (apply to weekends and weekdays)
            weekday_mask = tariff_row['Weekday?'] == (weekday < SATURDAY).astype('int64')
        if not np.isnan(tariff_row['Excluding Start Time']) and not np.isnan(tariff_row['Excluding End Time']):
            exclud_mask = (tariff_row['Excluding Start Time'] <= he_minute) & (he_minute <= tariff_row['Excluding End Time'])
        billing_pd_mask = np.array(month_mask & time_mask & np.logical_not(exclud_mask) & weekday_mask)
        if year is not None:
            billing_pd_mask = (tariff_row['Year'] == year) & billing_pd_mask
        return billing_pd_mask

    @staticmethod
    def calc_retail_energy_price(tariff, freq, analysis_yr, non_zero=True):
        """ transforms tariff data file into time series dataFrame

        Args:
            tariff (DataFrame): raw tariff dataframe.
            freq (str): the frequency of the timeseries data we are working with
            analysis_yr (float): Year for which to build the tariff for

        Returns: a DataFrame with the index beginning at hour 0

        """
        temp = pd.DataFrame(index=Lib.create_timeseries_index([analysis_yr], freq))
        size = len(temp)

        # Build Energy Price Vector
        temp['he'] = (temp.index + pd.Timedelta('1s')).hour + 1
        temp.loc[:, 'p_energy'] = np.zeros(shape=size)

        billing_period = [[] for _ in range(size)]

        for p in tariff.index:
            # edit the pricedf energy price and period values for all of the periods defined
            # in the tariff input file
            bill = tariff.loc[p, :]
            mask = Financial.create_bill_period_mask(bill, temp.index.month, temp['he'], temp.index.weekday)
            if bill['Charge'].lower() == 'energy':
                current_energy_prices = temp.loc[mask, 'p_energy'].values
                if np.any(np.greater(current_energy_prices, 0)):
                    # More than one energy price applies to the same time step
                    TellUser.warning('More than one energy price applies to the same time step.')
                # Add energy prices
                temp.loc[mask, 'p_energy'] += bill['Value']
            elif bill['Charge'].lower() == 'demand':
                # record billing period
                for i, true_false in enumerate(mask):
                    if true_false:
                        billing_period[i].append(p)
        billing_period = pd.DataFrame({'billing_period': billing_period}, dtype='object')
        temp.loc[:, 'billing_period'] = billing_period.values

        # ADD CHECK TO MAKE SURE ENERGY PRICES ARE THE SAME FOR EACH OVERLAPPING BILLING PERIOD
        # Check to see that each timestep has a period assigned to it

        if (not billing_period.apply(len).all() or np.any(np.equal(temp.loc[:, 'p_energy'].values, 0))) and non_zero:
            TellUser.error('The billing periods in the input file do not partition the year, '
                           + 'please check the tariff input file')
            raise TariffError('Please check the retail tariff')
        return temp

    def get_fuel_cost(self, fuel_type):
        """ This function looks up and returns the fuel_price from
                Financial attributes, based on a fuel_type
        Args:
            fuel_type (String): it must also have an associated price attribute
                     valid types: liquid, gas, other
        Returns:
            a fuel_cost (Float)
                units are $/MMBtu
        """
        fuel_cost = {
            'liquid': self.fuel_price_liquid,
            'gas': self.fuel_price_gas,
            'other': self.fuel_price_other
        }
        return fuel_cost[fuel_type]

    def proforma_report(self, technologies, valuestreams, results, opt_years):
        """ Calculates and returns the proforma

        Args:
            technologies (list): list of technologies (needed to get capital and om costs)
            valuestreams (Dict): Dict of all services to calculate cost avoided or profit
            results (DataFrame): DataFrame of all the concatenated timseries_report() method
                results from each DER and ValueStream
            opt_years (list)

        Returns: dataframe proforma
        """
        pro_forma = pd.DataFrame(index=pd.period_range(self.start_year, self.end_year, freq='y'))
        # add VS proforma report
        for service in valuestreams.values():
            df = service.proforma_report(opt_years, self.apply_rate,
                                         self.fill_non_optimization_years, results)
            pro_forma = pd.concat([pro_forma, df], axis=1)

        # add avoided costs from tariff bill reduction: Demand Charges
        if 'Demand Charge ($)' in self.monthly_bill.columns:
            # get growth rate (or assume 0)
            growth_rate = 0
            if 'DCM' in valuestreams.keys():
                growth_rate = valuestreams['DCM'].growth
            demand_charge_df = self.calculate_yearly_avoided_cost('Demand', growth_rate)
            pro_forma = pd.concat([pro_forma, demand_charge_df], axis=1)

        # add avoided costs from tariff bill reduction: Energy Charges
        if 'Energy Charge ($)' in self.monthly_bill.columns:
            # get growth rate (or assume 0)
            growth_rate = 0
            if 'retailTimeShift' in valuestreams.keys():
                growth_rate = valuestreams['retailTimeShift'].growth
            energy_charge_df = self.calculate_yearly_avoided_cost('Energy', growth_rate)
            pro_forma = pd.concat([pro_forma, energy_charge_df], axis=1)

        # add technology's proforma report
        for tech in technologies:
            df = tech.proforma_report(self.apply_rate, self.fill_non_optimization_years, results)
            if df is not None:
                pro_forma = pd.concat([pro_forma, df], axis=1)

        # add tax incentives if user wants to consider them
        if self.external_incentives:
            for year in self.yearly_data.index:
                if self.start_year.year <= year <= self.end_year.year:
                    pro_forma.loc[pd.Period(year=year, freq='y'), 'Tax Credit'] = \
                        self.yearly_data.loc[year, 'Tax Credit (nominal $)']
                    pro_forma.loc[pd.Period(year=year, freq='y'), 'Other Incentives'] = \
                        self.yearly_data.loc[year, 'Other Incentive (nominal $)']

        # fill in zero columns
        pro_forma = pro_forma.fillna(value=0)

        # filter out the rows that are not years (type = string) and put them at the top of the
        # table, sort the rest
        str_indexed_rows = pro_forma[[isinstance(i, str) for i in pro_forma.index]]
        year_indexed_rows = pro_forma[[isinstance(i, pd.Period) for i in pro_forma.index]]
        pro_forma = pd.concat([str_indexed_rows, year_indexed_rows])

        # calculate the net (sum of the row's columns)
        pro_forma['Yearly Net Value'] = pro_forma.sum(axis=1)
        return pro_forma

    def calculate_yearly_avoided_cost(self, charge_type, growth_rate):
        """Calculated the yearly avoided cost in the monthly bill, given the original cost and
        the new cost column names in the monthly bill

        Args:
            charge_type (string): 'Demand' or 'Energy'
            growth_rate (float): the rate at which the prices will grow (user defined)

        Returns: a df that can be concatinated with the profroma
        """
        avoided_cost_df = pd.DataFrame()
        # splice labels with CHARGE_TYPE string
        original_cost = f'Original {charge_type} Charge ($)'
        new_cost = f'{charge_type} Charge ($)'
        avoided_cost_name = f'Avoided {charge_type} Charge'

        yr_costs = self.monthly_bill.groupby(by=lambda x: x.year)[[new_cost, original_cost]].sum()
        avoided_cost = yr_costs.loc[:, original_cost] - yr_costs.loc[:, new_cost]
        for year, value in avoided_cost.iteritems():
            avoided_cost_df.loc[pd.Period(year, freq='y'), avoided_cost_name] = value
        avoided_cost_df = self.fill_non_optimization_years(avoided_cost_df, growth_rate)
        return avoided_cost_df

    def apply_rate(self, df, escalation_rate, base_year):
        """

        Args:
            df (pd.Dataframe): profroma type df
            escalation_rate (float):
            base_year (int)

        Returns: a df where the escalation_rate is applied and t=0 is the base year

        """
        # if escalation_rate is not given, use user given inflation rate
        if escalation_rate is None:
            escalation_rate = self.inflation_rate
        # find max year in index (there might be string in index so we find it by hand)
        for yr in df.index:
            if isinstance(yr, str):
                continue
            t = yr.year - base_year
            df.loc[yr, :] = df.loc[yr, :] * ((1 + escalation_rate) ** t)
        return df

    def fill_non_optimization_years(self, df, escalation_rate, is_om_cost = False):
        """

        Args:
           df (pd.Dataframe): profroma type df with only years in the index, and those year correspond
            to analysis years
           escalation_rate (float):
           is_om_cost (boolean): False by default
               when is_om_cost is True, the value occurring at
               the minimum opt year will not be changed

        Returns: a df where the data is filled by escalating values before and after the
            optimization years

        """
        # if escalation_rate is not given, use user given inflation rate
        if escalation_rate is None:
            escalation_rate = self.inflation_rate
        filled_df = pd.DataFrame(index=pd.period_range(self.start_year, self.end_year, freq='y'))
        filled_df = pd.concat([filled_df, df], axis=1)
        first_optimization_year = min(df.index.values)
        last_optimization_year = max(df.index.values)
        if not is_om_cost:
            if first_optimization_year != self.start_year:
                fill_back_years = pd.period_range(self.start_year, first_optimization_year, freq='y')
                # back fill until you hit start year
                i = len(fill_back_years) - 1
                while i > 0:
                    year_with_data = fill_back_years[i]
                    fill_year = fill_back_years[i - 1]
                    prev_years = filled_df.loc[year_with_data, :].copy(deep=True)
                    filled_df.loc[fill_year, :] = prev_years / (1 + escalation_rate)
                    i -= 1

        # use linear interpolation for growth in between optimization years
        filled_df = \
            filled_df.apply(lambda x: x.interpolate(method='linear', limit_area='inside'), axis=0)

        if not is_om_cost:
            # forward fill growth columns with inflation
            for yr in pd.period_range(last_optimization_year + 1, self.end_year, freq='y'):
                filled_df.loc[yr, :] = filled_df.loc[yr - 1, :] * (1 + escalation_rate)
        else:
            # special case for O&M rates
            # backward fill
            filled_df.fillna(method='bfill', inplace=True)
            # forward fill
            filled_df.fillna(method='ffill', inplace=True)
            # apply escalation rate to all values
            filled_df = self.apply_rate(filled_df, escalation_rate, first_optimization_year.year)

        return filled_df

    def cost_benefit_report(self, pro_forma):
        """ Calculates and returns a cost-benefit data frame

        Args:
            pro_forma (DataFrame): Pro-forma DataFrame that was created from each ValueStream or DER active

        """
        # remove 'Yearly Net Value' from dataframe before preforming the rest (we dont want to include net values, so we do this first)
        pro_forma = pro_forma.drop('Yearly Net Value', axis=1)

        # prepare for cost benefit
        cost_df = pd.DataFrame(pro_forma.values.clip(max=0))
        cost_df.columns = pro_forma.columns
        benefit_df = pd.DataFrame(pro_forma.values.clip(min=0))
        benefit_df.columns = pro_forma.columns

        cost_pv = 0  # cost present value (discounted cost)
        benefit_pv = 0  # benefit present value (discounted benefit)
        self.cost_benefit = pd.DataFrame({'Lifetime Present Value': [0, 0]}, index=pd.Index(['Cost ($)', 'Benefit ($)']))
        for col in cost_df.columns:
            present_cost = npf.npv(self.npv_discount_rate, cost_df[col].values)
            present_benefit = npf.npv(self.npv_discount_rate, benefit_df[col].values)

            self.cost_benefit[col] = [np.abs(present_cost), present_benefit]

            cost_pv += present_cost
            benefit_pv += present_benefit
        self.cost_benefit['Lifetime Present Value'] = [np.abs(cost_pv), benefit_pv]

        # Transforming cost_benefit df bc XENDEE asked us to.
        self.cost_benefit = self.cost_benefit.T

    def net_present_value_report(self, pro_forma):
        """ Uses the discount rate to calculate the present value for each column within the proforma

        Args:
            pro_forma (DataFrame): Pro-forma DataFrame that was created from each ValueStream or DER active

        """
        # use discount rate to calculate NPV for net
        npv_dict = {}
        # NPV for growth_cols
        for col in pro_forma.columns:
            if col == 'Yearly Net Value':
                npv_dict.update({'Lifetime Present Value': [npf.npv(self.npv_discount_rate, pro_forma[col].values)]})
            else:
                npv_dict.update({col: [npf.npv(self.npv_discount_rate, pro_forma[col].values)]})
        self.npv = pd.DataFrame(npv_dict, index=pd.Index(['NPV']))

    def payback_report(self, technologies, proforma, opt_years):
        """ calculates and saves the payback period and discounted payback period in a dataframe

        Args:
            technologies
            proforma (pd.DataFrame): Pro-forma DataFrame that was created from each ValueStream or DER active
            opt_years (list):

        Returns:

        """
        self.payback = pd.DataFrame({'Payback Period': self.payback_period(technologies, proforma, opt_years),
                                     'Discounted Payback Period': self.discounted_payback_period(technologies, proforma, opt_years)},
                                    index=pd.Index(['Years'], name='Unit'))

    @staticmethod
    def payback_period(techologies, proforma, opt_years):
        """The payback period is the number of years it takes the project to break even. In other words, if you
        ended the analysis at year x, the sum of all costs and benefits up to that point would be zero.

        These outputs are independent from the analysis horizon.

        Args:
            proforma (pd.DataFrame): Pro-forma DataFrame that was created from each ValueStream or DER active
            techologies
            opt_years (list)

        Returns: capex/yearlynetbenefit where capex is the year 0 capital costs and yearlynetbenefit is the (benefits - costs) in the first opt_year

        Notes:
            In the case of multiple opt_years, do not report a payback period for now.

        """
        capex = 0
        for tech in techologies:
            capex += tech.get_capex(solution=True)

        first_opt_year = min(opt_years)
        yearlynetbenefit = proforma.iloc[:, :-1].loc[pd.Period(year=first_opt_year, freq='y'), :].sum()

        if yearlynetbenefit == 0:
            # with yearlynetbenefit = 0, the equation becomes undefined, so we return nan
            return np.nan

        return capex/yearlynetbenefit

    def discounted_payback_period(self, technologies, proforma, opt_years):
        """ This is the number of years it takes for the NPV of the project to be zero. This number should be higher than the payback period
        when the discount rate is > 0. This number should be the same as the payback period if the discount rate is zero.

        The exact discounted payback period can be calculated with the following formula: log(1/(1-(capex*dr/yearlynetbenefit)))/log(1+dr) where
        capex is the year 0 capital costs, yearlynetbenefit is the (benefits - costs) in the first opt_year converted into year 0 dollars,
        dr is the discount rate [0,1], and the log function is base e.

        These outputs are independent from the analysis horizon.

        Args:
            technologies (list):
            proforma (DataFrame):
            opt_years (list):

        Returns:

        Notes:
            In the case of multiple opt_years, do not report a payback period for now.

        """
        payback_period = self.payback_period(technologies, proforma, opt_years)  # This is simply (capex/yearlynetbenefit)
        dr = self.npv_discount_rate
        if ((dr * payback_period) >= 1) or (dr == 0):
            # with (dr * payback_period) >= 1, or dr=0, the equation becomes undefined, so we return nan
            return np.nan
        discounted_pp = np.log(1/(1-(dr*payback_period)))/np.log(1+dr)

        return discounted_pp

    def report_dictionary(self):
        """

        Returns: dictionary of DataFrames of any reports to be given to the user
            keys are the file name that the df will be saved with

        """
        df_dict = dict()
        df_dict['pro_forma'] = self.pro_forma
        df_dict['npv'] = self.npv
        df_dict['cost_benefit'] = self.cost_benefit
        df_dict['payback'] = self.payback
        if self.customer_sided:
            df_dict['adv_monthly_bill'] = self.billing_period_bill
            df_dict['simple_monthly_bill'] = self.monthly_bill

        return df_dict
