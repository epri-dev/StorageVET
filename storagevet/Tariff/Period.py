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
