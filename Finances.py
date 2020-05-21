"""
Finances.py

This Python class contains methods and attributes vital for completing financial analysis given optimal dispathc.
"""

__author__ = 'Halley Nathwani, Miles Evans and Evan Giarta'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Micah Botkin-Levy']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Evan Giarta', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'egiarta@epri.com', 'mevans@epri.com']
__version__ = '2.1.1.1'


import pandas as pd
import numpy as np
import logging
import Library as Lib


SATURDAY = 5

e_logger = logging.getLogger('Error')
u_logger = logging.getLogger('User')


class Financial:

    def __init__(self, params):
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
        self.start_year = params['start_year']
        self.end_year = params['end_year']
        self.opt_years = params['opt_years']
        self.inflation_rate = params['inflation_rate']
        self.npv_discount_rate = params['npv_discount_rate']
        self.growth_rates = {'default': params['def_growth']}
        self.frequency = params['frequency']  # we assume that this is the same as the load data because loaded from the same time_series
        self.verbose = params['verbose']
        self.external_incentives = params['external_incentives']
        self.yearly_data = params['yearly_data']

        # create financial inputs data table

        # prep outputs
        self.fin_summary = pd.DataFrame()  # this is just the objective values evaluated at their minimum DO NOT REPORT TO USER -- FOR DEV
        self.pro_forma = pd.DataFrame()
        self.npv = pd.DataFrame()
        self.cost_benefit = pd.DataFrame()
        self.payback = pd.DataFrame()
        self.monthly_financials_data = pd.DataFrame()
        self.billing_period_bill = pd.DataFrame()
        self.monthly_bill = pd.DataFrame()

        # self.federal_tax_rate = params['federal_tax_rate']
        # self.state_tax_rate = params['state_tax_rate']
        # self.property_tax_rate = params['property_tax_rate']

    def preform_cost_benefit_analysis(self, technologies, value_streams, results):
        """ this function calculates the proforma, cost-benefit, npv, and payback using the optimization variable results
        saved in results and the set of technology and service instances that have (if any) values that the user indicated
        they wanted to use when evaluating the CBA.

        Instead of using the technologies and services as they are passed in from the call in the Results class, we will pass
        the technologies and services with the values the user denoted to be used for evaluating the CBA.

        Args:
            technologies (Dict): Dict of technologies (needed to get capital and om costs)
            value_streams (Dict): Dict of all services to calculate cost avoided or profit
            results (DataFrame): DataFrame of all the concatenated timseries_report() method results from each DER
                and ValueStream

        """
        if self.customer_sided:
            self.customer_bill(self.tariff, results.loc[:, 'Total Load (kW)'], results.loc[:, 'Net Load (kW)'])
        proforma = self.proforma_report(technologies, value_streams, results)
        self.cost_benefit_report(proforma)
        self.net_present_value_report(proforma)
        self.payback_report(proforma)

    def customer_bill(self, tariff, base_load, net_load):
        """ Calculates the demand and energy charges based on the provided tariff.

        Args:
            tariff (pd.DataFrame): The full tariff provided, with energy and demand
            base_load (pd.DataFrame): The original load BTM, without any DERs connected on the customer's side
            net_load (pd.DataFrame):  The net load that the meter sees as a result of adding DERs behind the meter

        Returns: 2 forms of the bill-- 1) a monthly index dataframe with a column for demand charges and a column
            for energy charges; 2) a billing period indexed dataframe with a column for demand charges and a column for
            energy charges

        Note: place holder -- might remove later

        """
        he = base_load.index.hour + 1  # TODO: double check this -HN
        month = base_load.index.month
        minute = base_load.index.minute
        he_minute = he + minute/60

        # if another year of data is added, there isnt a year column to denote the year it applies to, so we use the
        # self.billing_period values to determine which billing period applies

        # Calculate energy charge per billing period: since overlapping energy charges are added,
        # we must calculate energy charges the "long way"
        monthly_bill = pd.DataFrame()
        for item in tariff.index:
            # determine subset of data that
            bill = tariff.loc[item, :]
            month_mask = (bill["Start Month"] <= month) & (month <= bill["End Month"])
            time_mask = (bill['Start Time'] <= he_minute) & (he_minute <= bill['End Time'])
            weekday_mask = True
            exclud_mask = False
            if not bill['Weekday?'] == 2:  # if not (apply to weekends and weekdays)
                weekday_mask = bill['Weekday?'] == (base_load.index.weekday < SATURDAY).astype('int64')
            if not np.isnan(bill['Excluding Start Time']) and not np.isnan(bill['Excluding End Time']):
                exclud_mask = (bill['Excluding Start Time'] <= he_minute) & (he_minute <= bill['Excluding End Time'])
            billing_pd_mask = np.array(month_mask & time_mask & np.logical_not(exclud_mask) & weekday_mask)
            # billing_pd_net_load = pd.concat([net_load.loc[billing_pd_mask, :], base_load.loc[billing_pd_mask, :]], axis=1)
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

        # sum each billing period that applies to the same month
        # add all demand charges that apply to the same month
        sim_monthly_bill = monthly_bill.groupby(level=0).sum()
        for month_yr_index in monthly_bill.index.unique():
            mo_yr_data = monthly_bill.loc[month_yr_index, :]
            sim_monthly_bill.loc[month_yr_index, 'Billing Period'] = f"{mo_yr_data['Billing Period'].values}"
        sim_monthly_bill.index.name = 'Month-Year'
        self.monthly_bill = sim_monthly_bill

        return adv_monthly_bill, sim_monthly_bill

    @staticmethod
    def calc_retail_energy_price(tariff, freq, analysis_yrs, non_zero=True):
        """ transforms tariff data file into time series dataFrame

        Args:
            tariff (DataFrame): raw tariff dataframe.
            freq (str): the frequency of the timeseries data we are working with
            analysis_yrs (list): List of years that the analysis should be run on.

        Returns: a DataFrame with the index beginning at hour 0

        """
        temp = pd.DataFrame(index=Lib.create_timeseries_index(analysis_yrs, freq))
        size = len(temp)

        # Build Energy Price Vector
        temp['weekday'] = (temp.index.weekday < SATURDAY).astype('int64')
        temp['he'] = (temp.index + pd.Timedelta('1s')).hour + 1
        temp.loc[:, 'p_energy'] = np.zeros(shape=size)

        billing_period = [[] for _ in range(size)]

        for p in tariff.index:
            # edit the pricedf energy price and period values for all of the periods defined
            # in the tariff input file
            bill = tariff.loc[p, :]
            month_mask = (bill["Start Month"] <= temp.index.month) & (temp.index.month <= bill["End Month"])
            time_mask = (bill['Start Time'] <= temp['he']) & (temp['he'] <= bill['End Time'])
            weekday_mask = True
            exclud_mask = False
            if not bill['Weekday?'] == 2:  # if not (apply to weekends and weekdays)
                weekday_mask = bill['Weekday?'] == temp['weekday']
            if not np.isnan(bill['Excluding Start Time']) and not np.isnan(bill['Excluding End Time']):
                exclud_mask = (bill['Excluding Start Time'] <= temp['he']) & (temp['he'] <= bill['Excluding End Time'])
            mask = np.array(month_mask & time_mask & np.logical_not(exclud_mask) & weekday_mask)
            if bill['Charge'].lower() == 'energy':
                current_energy_prices = temp.loc[mask, 'p_energy'].values
                if np.any(np.greater(current_energy_prices, 0)):
                    # More than one energy price applies to the same time step
                    u_logger.debug('More than one energy price applies to the same time step.')
                # Add energy prices
                temp.loc[mask, 'p_energy'] += bill['Value']
            elif bill['Charge'].lower() == 'demand':
                for i, true_false in enumerate(mask):
                    if true_false:
                        billing_period[i].append(p)
        billing_period = pd.DataFrame({'billing_period': billing_period}, dtype='object')
        temp.loc[:, 'billing_period'] = billing_period.values

        # ADD CHECK TO MAKE SURE ENERGY PRICES ARE THE SAME FOR EACH OVERLAPPING BILLING PERIOD
        # Check to see that each timestep has a period assigned to it
        if (not billing_period.apply(len).all() or np.any(np.equal(temp.loc[:, 'p_energy'].values, 0))) and non_zero:
            e_logger.error('The billing periods in the input file do not partition the year')
            e_logger.error('please check the tariff input file')
            e_logger.error('Error with tariff input file when calculating retail energy prices.')
            u_logger.error('The billing periods in the input file do not partition the year')
            u_logger.error('please check the tariff input file')
            u_logger.error('Error with tariff input file!')
            raise ValueError('Please check the retail tariff')
        return temp

    def proforma_report(self, technologies, valuestreams, results):
        """ Calculates and returns the proforma

        Args:
            technologies (Dict): Dict of technologies (needed to get capital and om costs)
            valuestreams (Dict): Dict of all services to calculate cost avoided or profit
            results (DataFrame): DataFrame of all the concatenated timseries_report() method results from each DER
                and ValueStream

        Returns: dataframe proforma
        """
        # create yearly data table
        yr_index = pd.period_range(start=self.start_year, end=self.end_year, freq='y')
        project_index = np.insert(yr_index.values, 0, 'CAPEX Year')
        pro_forma = pd.DataFrame(index=project_index)

        # list of all the financials that are constant
        const_col = []
        # list of financials that are zero unless already specified
        zero_col = []

        # add VS proforma report
        for service in valuestreams.values():
            df, zero_names, const_names = service.proforma_report(self.opt_years, results)
            pro_forma = pro_forma.join(df)
            if zero_names:
                zero_col += zero_names
            if const_names:
                const_col += const_names

        # add technology's proforma report
        for tech in technologies.values():
            df = tech.proforma_report(self.opt_years, results)
            pro_forma = pro_forma.join(df)
            zero_col += [tech.zero_column_name]
            const_col += [tech.fixed_column_name]

        # add avoided costs from tariff bill reduction: Demand Charges
        if 'Demand Charge ($)' in self.monthly_bill.columns:
            yearly_demand_charges = self.monthly_bill.groupby(by=lambda x: x.year)[['Demand Charge ($)', 'Original Demand Charge ($)']].sum()
            avoided_cost = yearly_demand_charges.loc[:, 'Original Demand Charge ($)'] - yearly_demand_charges.loc[:, 'Demand Charge ($)']

            pro_forma['Avoided Demand Charge'] = 0
            for year, value in avoided_cost.iteritems():
                pro_forma.loc[pd.Period(year, freq='y'), 'Avoided Demand Charge'] = value

        # add avoided costs from tariff bill reduction: Energy Charges
        if 'Energy Charge ($)' in self.monthly_bill.columns:
            yearly_demand_charges = self.monthly_bill.groupby(by=lambda x: x.year)[['Energy Charge ($)', 'Original Energy Charge ($)']].sum()
            avoided_cost = yearly_demand_charges.loc[:, 'Original Energy Charge ($)'] - yearly_demand_charges.loc[:, 'Energy Charge ($)']

            pro_forma['Avoided Energy Charge'] = 0
            for year, value in avoided_cost.iteritems():
                pro_forma.loc[pd.Period(year, freq='y'), 'Avoided Energy Charge'] = value

        # add tax incentives if user wants to consider them
        if self.external_incentives:
            for year in self.yearly_data.index:
                if self.start_year.year <= year <= self.end_year.year:
                    pro_forma.loc[pd.Period(year=year, freq='y'), 'Tax Credit'] = self.yearly_data.loc[year, 'Tax Credit (nominal $)']
                    pro_forma.loc[pd.Period(year=year, freq='y'), 'Other Incentives'] = self.yearly_data.loc[year, 'Other Incentive (nominal $)']
                    zero_col += ['Tax Credit', 'Other Incentives']

        # the rest of columns should grow year over year
        growth_col = list(set(list(pro_forma)) - set(const_col) - set(zero_col))

        # set the 'CAPEX Year' row to all zeros
        pro_forma.loc['CAPEX Year', growth_col + const_col] = np.zeros(len(growth_col + const_col))
        # use linear interpolation for growth in between optimization years
        pro_forma[growth_col] = pro_forma[growth_col].apply(lambda x: x.interpolate(method='linear', limit_area='inside'), axis=0)

        # forward fill growth columns with inflation
        last_sim = max(self.opt_years)
        for yr in pd.period_range(start=last_sim + 1, end=self.end_year, freq='y'):
            pro_forma.loc[yr, growth_col] = pro_forma.loc[yr - 1, growth_col] * (1 + self.inflation_rate / 100)
        # backfill growth columns (needed for year 0)
        pro_forma[growth_col] = pro_forma[growth_col].fillna(value='bfill')
        # fill in constant columns
        pro_forma[const_col] = pro_forma[const_col].fillna(method='ffill')
        # fill in zero columns
        pro_forma[zero_col] = pro_forma[zero_col].fillna(value=0)

        # calculate the net (sum of the row's columns)
        pro_forma['Yearly Net Value'] = pro_forma.sum(axis=1)
        self.pro_forma = pro_forma
        return pro_forma

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
            present_cost = np.npv(self.npv_discount_rate / 100, cost_df[col].values)
            present_benefit = np.npv(self.npv_discount_rate / 100, benefit_df[col].values)

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
                npv_dict.update({'Lifetime Present Value': [np.npv(self.npv_discount_rate / 100, pro_forma[col].values)]})
            else:
                npv_dict.update({col: [np.npv(self.npv_discount_rate / 100, pro_forma[col].values)]})
        self.npv = pd.DataFrame(npv_dict, index=pd.Index(['NPV']))

    def payback_report(self, proforma):
        """ calculates and saves the payback period and discounted payback period in a dataframe

        Args:
            proforma (DataFrame): Pro-forma DataFrame that was created from each ValueStream or DER active

        Returns:

        """
        self.payback = pd.DataFrame({'Payback Period': self.payback_period(proforma),
                                     'Discounted Payback Period': self.discounted_payback_period(proforma)},
                                    index=pd.Index(['Years']))

    def payback_period(self, proforma):
        """The payback period is the number of years it takes the project to break even. In other words, if you
        ended the analysis at year x, the sum of all costs and benefits up to that point would be zero.

        These outputs are independent from the analysis horizon.

        Args:
            proforma (DataFrame): Pro-forma DataFrame that was created from each ValueStream or DER active

        Returns: capex/yearlynetbenefit where capex is the year 0 capital costs and yearlynetbenefit is the (benefits - costs) in the first opt_year

        Notes:
            In the case of multiple opt_years, do not report a payback period for now.

        """
        capex = abs(proforma.loc['CAPEX Year', :].sum())

        first_opt_year = min(self.opt_years)
        yearlynetbenefit = proforma.loc[pd.Period(year=first_opt_year, freq='y'), :].sum()

        return capex/yearlynetbenefit

    def discounted_payback_period(self, proforma):
        """ This is the number of years it takes for the NPV of the project to be zero. This number should be higher than the payback period
        when the discount rate is > 0. This number should be the same as the payback period if the discount rate is zero.

        The exact discounted payback period can be calculated with the following formula: log(1/(1-(capex*dr/yearlynetbenefit)))/log(1+dr) where
        capex is the year 0 capital costs, yearlynetbenefit is the (benefits - costs) in the first opt_year converted into year 0 dollars,
        dr is the discount rate [0,1], and the log function is base e.

        These outputs are independent from the analysis horizon.

        Args:
            proforma(DataFrame):

        Returns:

        Notes:
            In the case of multiple opt_years, do not report a payback period for now.

        """
        payback_period = self.payback_period(proforma)  # This is simply (capex/yearlynetbenefit)

        if not self.npv_discount_rate:
            dr = self.npv_discount_rate*.01
            discounted_pp = np.log(1/(1-(dr*payback_period)))/np.log(1+dr)
        else:
            # if discount_rate = 0, then the equation simplifies to np.log(1)/np.log(1)-- which is undefined, so we return nan
            discounted_pp = 'nan'

        return discounted_pp
