import requests
import pprint
import Period as period
import EnergyTier as et
import csv
import os
import pandas as pd


class API:
    def __init__(self):
        self.URL = "https://api.openei.org/utility_rates"
        self.PARAMS = {'version': 5, 'api_key': 'AHNGCVlcEKTgH6fgr1bXiSnneVb00masZcjSgp3I', 'format': 'json',
                       'getpage': '58ff9b425457a34848cd3b46', 'limit': 20}  # 'address': '302 College Ave, Palo Alto'
        self.r = requests.get(url=self.URL, params=self.PARAMS)
        self.data = self.r.json()
        self.temp_file = "tariff_temp.csv"
        self.new_file = "tariff.csv"

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

        self.header = ['Period', 'Tier 1 Max', 'Tier 1 Rate',
                                 'Tier 2 Max', 'Tier 2 Rate',
                                 'Tier 3 Max', 'Tier 3 Rate',
                                 'Tier 4 Max', 'Tier 4 Rate',
                                 'Tier 5 Max', 'Tier 5 Rate',
                                 'Tier 6 Max', 'Tier 6 Rate',
                                 'Tier 7 Max', 'Tier 7 Rate',
                                 'Tier 8 Max', 'Tier 8 Rate']

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

    def energy_structure(self):
        """
        Prints energy structure, month and hour schedule of when every period is active, to terminal

        """
        self.energyweekdayschedule = self.tariff["items"][0]["energyweekdayschedule"]
        self.energyweekendschedule = self.tariff["items"][0]["energyweekendschedule"]
        for year in self.energyweekdayschedule:
            count = 0
            for month in year:
                year[count] = month + 1
                count += 1
        for year in self.energyweekendschedule:
            count = 0
            for month in year:
                year[count] = month + 1
                count += 1

    def calendar(self):
        """
        Makes a csv file with weekday schedule, weekend schedule, and the rates of each period

        """
        with open(self.temp_file, "w", newline='') as csvfile:
            tariff_writer = csv.writer(csvfile)
            count = 0
            hours = [" ", 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

            tariff_writer.writerow(hours)
            for i in self.energyweekdayschedule:
                i.insert(0, months[count])
                tariff_writer.writerow(i)
                count += 1

            tariff_writer.writerow(" ")
            tariff_writer.writerow(" ")
            tariff_writer.writerow(" ")

            count = 0
            tariff_writer.writerow(hours)
            for i in self.energyweekendschedule:
                i.insert(0, months[count])
                tariff_writer.writerow(i)
                count += 1

            tariff_writer.writerow(" ")
            tariff_writer.writerow(" ")
            tariff_writer.writerow(" ")

            tariff_writer.writerow(self.header)
            for period in self.energy_period_list:
                row = [period.number]
                for tier in period.tier_list:
                    row.append(tier.max)
                    row.append(tier.rate)
                tariff_writer.writerow(row)

    def read_csv(self):
        """
        Reads the csv file back and creates three data frames based on weekday schedule, weekend schedule, and periods

        """
        with open(self.temp_file, 'r') as inp, open(self.new_file, "w") as out:
            writer = csv.writer(out)
            for row in csv.reader(inp):
                if ''.join(row).strip():  # https://stackoverflow.com/questions/18890688/how-to-skip-blank-line-while-reading-csv-file-using-python/54381516
                    writer.writerow(row)
        os.remove(self.temp_file)
        text = pd.read_csv(self.new_file)

        # weekday schedule
        print("============================")
        print("DF_WEEKDAY")
        weekday_df = text[:12]
        print(weekday_df)
        print("\n")

        # weekend schedule
        print("DF_WEEKEND")
        weekend_df = text[13:25]
        weekend_df.reset_index(drop=True, inplace=True)
        print(weekend_df)
        print("\n")

        # periods and tiers
        print("DF_PERIODS")
        periods_df = text[25:]

        # rename header to period header
        header = periods_df.iloc[0]
        periods_df = periods_df[1:]
        periods_df = periods_df.rename(columns=header)

        # reset index to start at 0
        periods_df.reset_index(drop=True, inplace=True)

        # remove all columns that are nan
        periods_df = periods_df.loc[:, periods_df.columns.notnull()]
        print(periods_df)
        print("\n")

    def run(self):
        """
        Runs the program utilizing the functions

        """
        self.print_all()
        i = int(input("Which tariff would you like to use?..."))
        self.print_index(i)
        self.energy_structure()
        self.calendar()
        os.startfile(self.temp_file)
        response = input("Type 'ready' when you are done editing the excel file...")
        while response != "ready":
            response = input("Type 'ready' when you are done editing the excel file...")
        self.read_csv()


def main():
    api = API()
    api.run()


if __name__ == "__main__": main()
