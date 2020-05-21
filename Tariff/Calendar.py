import requests
import pprint
import Period as period
import EnergyTier as et
import xlsxwriter
import os
import pandas as pd


class API:
    def __init__(self):
        self.URL = "https://api.openei.org/utility_rates"
        self.PARAMS = {'version': 5, 'api_key': 'AHNGCVlcEKTgH6fgr1bXiSnneVb00masZcjSgp3I', 'format': 'json',
                       'getpage': '5b78ba715457a3bf45af0aea', 'limit': 20}  # 'address': '302 College Ave, Palo Alto'
        self.r = requests.get(url=self.URL, params=self.PARAMS)
        self.data = self.r.json()
        self.tariff = None
        self.energyratestructure = []
        self.energyweekdayschedule = []
        self.energyweekendschedule = []
        self.energy_period_list = []

        self.max = None
        self.rate = None
        self.unit = None
        self.adj = None
        self.sell = None

        self.weekday_date_list = []
        self.weekend_date_list = []
        self.date_list = []

    def print_all(self):
        """
        Prints necessary identifying information of all tariffs that show from result page on OpenEI

        """
        count = 1
        for item in self.data["items"]:
            print("---------------------------------------------------", count)
            print("Utility.......", item["utility"])
            print("Name..........", item["name"])
            if "enddate" in item:
                print("End Date......", item["enddate"])
            if "startdate" in item:
                print("Start Date....", item["startdate"])
            print("EIA ID........", item["eiaid"])
            print("URL...........", item["uri"])
            if "description" in item:
                print("Description...", item["description"])
            print(" ")
            count += 1

    def reset(self):
        """
        Resets tariff's tier values to None; necessary for print_index

        """
        self.max = None
        self.rate = None
        self.unit = None
        self.adj = None
        self.sell = None

    def print_index(self, index):
        """
        Establishes all periods and tiers of the tariff using period and tier objects

        Args:
            index (Int): user input for which tariff they choose

        """
        i = index
        label = self.data["items"][i - 1]["label"]
        params = {'version': 5, 'api_key': 'AHNGCVlcEKTgH6fgr1bXiSnneVb00masZcjSgp3I', 'format': 'json', 'getpage': label, 'detail': 'full'}
        r = requests.get(url=self.URL, params=params)
        self.tariff = r.json()

        if "energyratestructure" in self.tariff["items"][0]:
            # print(self.tariff["items"][0]["energyratestructure"])
            self.energyratestructure = self.tariff["items"][0]["energyratestructure"]
            pcount = 1  # period count
            tcount = 1  # tier count
            for p in self.energyratestructure:
                self.energy_period_list.append(period.Period(pcount))
                for i in p:
                    if "max" in i:
                        self.max = i["max"]

                    if "rate" in i:
                        self.rate = i["rate"]

                    if "unit" in i:
                        self.unit = i["unit"]

                    if "adjustment" in i:
                        self.adj = i["adjustment"]

                    if "sell" in i:
                        self.sell = i["sell"]

                    self.energy_period_list[pcount - 1].add(et.Tier(tcount, self.max, self.rate, self.unit, self.adj, self.sell))
                    tcount += 1
                    self.reset()
                tcount = 1
                pcount += 1

    def print_energy_structure(self):
        """
        Prints energy structure, month and hour schedule of when every period is active, to terminal

        """
        pprint.pprint(self.tariff)
        if not self.energy_period_list:  # if list is empty it is not printed
            pass
        else:
            print(" ")
            print("Tiered Energy Usage Charge Structure")
            for period in self.energy_period_list:
                print(" ")
                period.tostring()
                for tier in period.tier_list:
                    tier.tostring()
            print(" ")

        self.energyweekdayschedule = self.tariff["items"][0]["energyweekdayschedule"]
        self.energyweekendschedule = self.tariff["items"][0]["energyweekendschedule"]
        for year in self.energyweekdayschedule:
            count = 0
            for month in year:
                year[count] = month + 1
                count += 1
            print(year)
        print('=----------------------------------------------------------------------=')
        for year in self.energyweekendschedule:
            count = 0
            for month in year:
                year[count] = month + 1
                count += 1
            print(year)

    def calendar(self):
        """
        Makes an excel file with three spreadsheets: weekday schedule, weekend schedule, and the rates of each period

        """
        # create three workbook with three worksheets
        workbook = xlsxwriter.Workbook('calendar.xlsx')
        wksht_weekday = workbook.add_worksheet(name="Weekday")
        wksht_weekend = workbook.add_worksheet(name="Weekend")
        wksht_rates = workbook.add_worksheet(name="Rates")

        hours = [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        # conditional formatting that changes the format of a cell based on number value
        # yellow
        yellow = workbook.add_format()
        yellow.set_align('center')
        yellow.set_bg_color('yellow')
        yellow.set_bold()
        yellow.set_font_color('black')
        yellow.set_border(1)
        yellow.set_border_color('white')
        cond_yellow = wksht_weekday.conditional_format(1, 1, 12, 24, {'type': 'cell', 'criteria': '=', 'value': 1, 'format': yellow})
        cond_yellow = wksht_weekend.conditional_format(1, 1, 12, 24, {'type': 'cell', 'criteria': '=', 'value': 1, 'format': yellow})

        # blue
        blue = workbook.add_format()
        blue.set_align('center')
        blue.set_bg_color('blue')
        blue.set_bold()
        blue.set_font_color('white')
        blue.set_border(1)
        blue.set_border_color('white')
        cond_blue = wksht_weekday.conditional_format(1, 1, 12, 24, {'type': 'cell', 'criteria': '=', 'value': 2, 'format': blue})
        cond_blue = wksht_weekend.conditional_format(1, 1, 12, 24, {'type': 'cell', 'criteria': '=', 'value': 2, 'format': blue})

        # green
        green = workbook.add_format()
        green.set_align('center')
        green.set_bg_color('green')
        green.set_bold()
        green.set_font_color('white')
        green.set_border(1)
        green.set_border_color('white')
        cond_green = wksht_weekday.conditional_format(1, 1, 12, 24, {'type': 'cell', 'criteria': '=', 'value': 3, 'format': green})
        cond_green = wksht_weekend.conditional_format(1, 1, 12, 24, {'type': 'cell', 'criteria': '=', 'value': 3, 'format': green})

        # red
        red = workbook.add_format()
        red.set_align('center')
        red.set_bg_color('red')
        red.set_bold()
        red.set_font_color('black')
        red.set_border(1)
        red.set_border_color('white')
        cond_red = wksht_weekday.conditional_format(1, 1, 12, 24, {'type': 'cell', 'criteria': '=', 'value': 4, 'format': red})
        cond_red = wksht_weekend.conditional_format(1, 1, 12, 24, {'type': 'cell', 'criteria': '=', 'value': 4, 'format': red})

        # purple
        purple = workbook.add_format()
        purple.set_align('center')
        purple.set_bg_color('purple')
        purple.set_bold()
        purple.set_font_color('white')
        purple.set_border(1)
        purple.set_border_color('white')
        cond_purple = wksht_weekday.conditional_format(1, 1, 12, 24, {'type': 'cell', 'criteria': '=', 'value': 5, 'format': purple})
        cond_purple = wksht_weekend.conditional_format(1, 1, 12, 24, {'type': 'cell', 'criteria': '=', 'value': 5, 'format': purple})

        # lime
        lime = workbook.add_format()
        lime.set_align('center')
        lime.set_bg_color('lime')
        lime.set_bold()
        lime.set_font_color('black')
        lime.set_border(1)
        lime.set_border_color('white')
        cond_lime = wksht_weekday.conditional_format(1, 1, 12, 24, {'type': 'cell', 'criteria': '=', 'value': 6, 'format': lime})
        cond_lime = wksht_weekend.conditional_format(1, 1, 12, 24, {'type': 'cell', 'criteria': '=', 'value': 6, 'format': lime})

        # else
        center = workbook.add_format()
        center.set_align('center')
        cond_else = wksht_weekday.conditional_format(1, 1, 12, 24, {'type': 'cell', 'criteria': '>', 'value': 6, 'format': center})
        cond_else = wksht_weekend.conditional_format(1, 1, 12, 24, {'type': 'cell', 'criteria': '>', 'value': 6, 'format': center})

        # -------------------- weekday --------------------
        # write hours in header
        for i in range(len(hours)):
            wksht_weekday.write(0, i+1, hours[i])
        wksht_weekday.set_column(1, 24, 3.14, center)

        # write months in first column
        for i in range(len(months)):
            wksht_weekday.write(i+1, 0, months[i])
        wksht_weekday.set_column(0, 0, 4, center)

        # write all periods conditional formatting in weekday schedule
        x = 0
        y = 0
        for month in self.energyweekdayschedule:
            for hour in month:
                if hour == 1:
                    wksht_weekday.write(1 + y, 1 + x, hour, cond_yellow)
                elif hour == 2:
                    wksht_weekday.write(1 + y, 1 + x, hour, cond_blue)
                elif hour == 3:
                    wksht_weekday.write(1 + y, 1 + x, hour, cond_green)
                elif hour == 4:
                    wksht_weekday.write(1 + y, 1 + x, hour, cond_red)
                elif hour == 5:
                    wksht_weekday.write(1 + y, 1 + x, hour, cond_purple)
                elif hour == 6:
                    wksht_weekday.write(1 + y, 1 + x, hour, cond_lime)
                else:
                    wksht_weekday.write(1 + y, 1 + x, hour, cond_else)
                x += 1
            x = 0
            y += 1

        # -------------------- weekend --------------------
        # write hours in header
        for i in range(len(hours)):
            wksht_weekend.write(0, i+1, hours[i])
        wksht_weekend.set_column(1, 24, 3.14, center)

        # write months in first column
        for i in range(len(months)):
            wksht_weekend.write(i+1, 0, months[i])
        wksht_weekend.set_column(0, 0, 4, center)

        # write all periods with conditional formatting in weekend schedule
        x = 0
        y = 0
        for month in self.energyweekendschedule:
            for hour in month:
                if hour == 1:
                    wksht_weekend.write(1 + y, 1 + x, hour, cond_yellow)
                elif hour == 2:
                    wksht_weekend.write(1 + y, 1 + x, hour, cond_blue)
                elif hour == 3:
                    wksht_weekend.write(1 + y, 1 + x, hour, cond_green)
                elif hour == 4:
                    wksht_weekend.write(1 + y, 1 + x, hour, cond_red)
                elif hour == 5:
                    wksht_weekend.write(1 + y, 1 + x, hour, cond_purple)
                elif hour == 6:
                    wksht_weekend.write(1 + y, 1 + x, hour, cond_lime)
                else:
                    wksht_weekend.write(1 + y, 1 + x, hour, cond_else)
                x += 1
            x = 0
            y += 1

        # -------------------- rates --------------------
        # write period and tiers in header
        header = ['Period', 'Tier 1', 'Tier 2', 'Tier 3', 'Tier 4', 'Tier 5', 'Tier 6', 'Tier 7', 'Tier 8']
        for i in range(len(header)):
            wksht_rates.write(0, i, header[i])
        wksht_rates.set_column(0, 0, 6.14, center)
        wksht_rates.set_column(1, 8, 8.3, center)

        # write period number and subsequent tier rates
        period_number = 1
        count = 0
        for period in self.energy_period_list:
            wksht_rates.write(period_number, 0, period_number)
            for tier in period.tier_list:
                wksht_rates.write(period_number, 1 + count, tier.get_rate())
                count += 1
            count = 0
            period_number += 1
        workbook.close()

    def read_calendar(self):
        """
        After user confirms their excel workbook is complete, each sheet is turned into a data frame

        """
        print(" ")
        file = "calendar.xlsx"
        print("DF_WEEKDAY")
        df_weekday = pd.read_excel(file, sheet_name="Weekday")
        print(df_weekday)
        print(" ")
        print("DF_WEEKEND")
        df_weekend = pd.read_excel(file, sheet_name="Weekend")
        print(df_weekend)
        print(" ")
        print("DF_RATES")
        df_rates = pd.read_excel(file, sheet_name="Rates")
        print(df_rates)
        print(" ")

    def run(self):
        """
        Runs the program utilizing the functions

        """
        self.print_all()
        i = int(input("Which tariff would you like to use?..."))
        self.print_index(i)
        self.print_energy_structure()
        self.calendar()
        file = "calendar.xlsx"
        os.startfile(file)
        response = input("Type 'ready' when you are done editing the excel file...")
        while response != "ready":
            response = input("Type 'ready' when you are done editing the excel file...")
        self.read_calendar()


def main():
    api = API()
    api.run()

if __name__ == "__main__": main()