class Tier:
    def __init__(self, number, max, rate, unit, adj, sell):
        self.number = number
        self.max = max
        self.rate = rate
        self.adj = adj
        self.unit = unit
        self.sell = sell

    def tostring(self):
        """
        Pretty print all necessary tier information

        """
        if self.max is not None:
            if self.unit is not None:
                print("Tier " + str(self.number) + ": Max Usage: " + str(self.max) + " " + self.unit)
            else:
                print("Tier " + str(self.number) + ": Max Usage: " + str(self.max))

        if self.rate is not None:
            if self.unit is not None:
                print("Tier " + str(self.number) + ": Rate: $" + str(self.rate) + " /" + self.unit)
            else:
                print("Tier " + str(self.number) + ": Rate: $" + str(self.rate))

        if self.adj is not None:
            if self.unit is not None:
                print("Tier " + str(self.number) + ": Adjustments: $" + str(self.adj) + " /" + self.unit)
            else:
                print("Tier " + str(self.number) + ": Adjustments: $" + str(self.adj))

        if self.sell is not None:
            if self.unit is not None:
                print("Tier " + str(self.number) + ": Sell: $" + str(self.sell) + " /" + self.unit)
            else:
                print("Tier " + str(self.number) + ": Sell: $" + str(self.sell))

    def get_rate(self):
        """
        Returns:
            self.rate (Int): this tier's rate

        """
        return self.rate

