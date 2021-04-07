"""
Copyright (c) 2021, Electric Power Research Institute

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

