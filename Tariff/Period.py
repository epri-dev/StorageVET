class Period:
    def __init__(self, number):
        self.number = number
        self.tier_list = []
        self.highest_rate = 0

    def get_tier(self, number):
        """"
        Args:
            number (Int): index for which tier to return

        Returns:
            self.tier_list (EnergyTier): tier based on argument index

        """
        return self.tier_list[number]

    def add(self, tier):
        """"
        Args:
            tier (EnergyTier): new tier to be appended to tier_list

        """
        self.tier_list.append(tier)

    def tostring(self):
        """
        Pretty print

        """
        print("Period " + str(self.number) + "-------------------------=")

    def get_highest_rate(self):
        """
        Sets the highest rate out of the tier_list

        """
        for tier in self.tier_list:
            if tier.get_rate() is None:
                continue
            elif tier.get_rate() > self.highest_rate:
                self.highest_rate = tier.get_rate()
