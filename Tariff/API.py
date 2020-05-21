import requests
import pprint
import Period as period
import EnergyTier as et
import csv

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
        self.schedule = []
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

    def dates(self, dates, weekday):
        """
        Looks at energy weekday schedule and establishes a list of periods to describe the schedule using start and end months

        """
        if weekday is True:
            schedule = self.energyweekdayschedule
        else:
            schedule = self.energyweekendschedule
        dates = dates
        switch = False  # true if period in row for the first time (start is set)
        period = 1      # period we are looking for in schedule
        month = 1       # current month of for loop
        start = 0       # start month
        end = 0         # end month
        index = 0       # index to compare current row to next row

        # continues to loop unless period is not found in row
        while True:
            for row in schedule:
                # don't need to check past row 11
                if index != 11:
                    index += 1

                # switch is False if start has not been set
                if switch is False:
                    if period in row:
                        start = month
                        end = month
                        switch = True
                        month += 1
                    else:
                        month += 1

                # switch is True if start has been set
                elif switch is True:
                    if period in row:
                        end = month
                        month += 1
                    else:
                        dates.append([period, start, end, period])
                        switch = False
                        month += 1

                # if for loop is on last loop, append what it has
                if month >= 13:
                    # if period was never given months it is not appended to list
                    if start == 0:
                        period += 1
                        month = 1
                        index = 0
                        break
                    dates.append([period, start, end, period])
                    period += 1
                    month = 1
                    index = 0

                # if the next row is different from the current row it will append the current period and start a new one
                if schedule[index] != row:
                    if start == 0:
                        continue
                    else:
                        dates.append([period, start, end, period])
                        switch = False

            # if start is 0 then the period does not exist in the schedule and we are done
            if period > len(self.energy_period_list):
                self.rates(dates)
                break
            else:
                start = 0
                switch = False

    def hours(self, dates, weekday):
        """
        Looks at energy weekday schedule and establishes range of hours that a period is active for

        """
        if weekday is True:
            schedule = self.energyweekdayschedule
        else:
            schedule = self.energyweekendschedule
        dates = dates
        switch = 0
        start = 0     # start month
        end = 0       # end month
        ex_start = 0  # excluding start month
        ex_end = 0    # excluding end month
        time = 0

        for p in dates:
            period = p[3]   # period that we are finding active times for
            index = p[1]-1  # look at starting month to find times that period is active
            month = schedule[index]
            for hour in month:
                time += 1
                # case 0: start month has not yet been found, once found goes to case 1
                if switch == 0:
                    if hour == period:
                        start = time
                        end = time
                        switch = 1
                    else:
                        continue

                # case 1: start month is set, if hour is equal to period there is possible gap which goes to case 2
                elif switch == 1:
                    if hour == period:
                        end = time
                    else:
                        if start == 1:
                            continue
                        else:
                            ex_start = end
                            switch = 2

                # case 2: if there is a gap between a period, sets ex_end goes to case 3
                elif switch == 2:
                    if hour == period:
                        end = time
                        ex_end = time
                        switch = 3
                    else:
                        continue

                # case 3: sets end of gap
                elif switch == 3:
                    if hour == period:
                        end = time
                    else:
                        continue

            if ex_end == 0:
                ex_start = None
                ex_end = None
            p.append(start-1)
            p.append(end-1)
            p.append(ex_start)
            p.append(ex_end)
            del p[3]
            ex_start = 0
            ex_end = 0
            switch = False
            time = 0

    def rates(self, dates):
        """
        Assigns rates to each energy period in date_list before list is formatted

        """
        temp_list = []
        for p in self.energy_period_list:
            p.get_highest_rate()
            temp_list.append(p.highest_rate)

        for period in dates:
            period.append(temp_list[period[3]-1])

    def clean_list(self, dates):
        """
        Removes duplicates from date_list and orders it according to starting month of every period

        """
        self.remove_duplicates(dates)
        dates.sort(key=self.take_second)  # sorts list based on second element (starting month)
        count = 1
        for p in dates:
            p[0] = count
            count += 1
            p.append(p.pop(3))  # moves rate to end of list

    def remove_duplicates(self, dates):
        """
        Removes all duplicates from a list leaving only one of an element

        """
        for p in dates:
            while dates.count(p) >= 2:
                dates.remove(p)

    def take_second(self, elem):
        """
        Args:
            elem: list

        Returns:
            elem[1] (Element): second element of list

        """
        return elem[1]

    def run(self):
        self.print_all()
        i = int(input("Which tariff would you like to use?..."))
        self.print_index(i)
        self.print_energy_structure()

        print("WEEKDAY")
        weekday = []
        self.dates(weekday, True)
        self.hours(weekday, True)
        self.clean_list(weekday)
        for p in weekday:
            print(p)

        print("WEEKEND")
        weekend = []
        self.dates(weekend, False)
        self.hours(weekend, False)
        self.clean_list(weekend)
        for p in weekend:
            print(p)

        """
            with open('tariff.csv', mode='w', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',')
            filewriter.writerow(['Billing Period', 'Start Month', 'End Month', 'Start Time', 'End Time', 'Excluding Start Time', 'Excluding End Time', 'Weekday?', 'Value', 'Charge', 'Name_optional'])
            for p in self.date_list:
                filewriter.writerow([p[0], p[1], p[2], p[3], p[4], p[5], p[6], None, p[7]])
        """


def main():
    api = API()
    api.run()

if __name__ == "__main__": main()