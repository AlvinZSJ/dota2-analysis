##########
# Author: ZHANG Shenjia
#
##########


import pandas as pd


class RuleMetrics:

    def __init__(self, lhs, rhs, rule_type):
        """

        :param lhs:         (list) heros in the left hand side of association rule
        :param rhs:         (list) heros in the right hand side of association rule
        :param rule_type:   (str)  type of rule: allies / enemies
        """
        self._lhs = lhs
        self._rhs = rhs

        if rule_type not in ['allies', 'enemies']:
            raise Exception("rule type should be allies or enemies!")
        else:
            self._rule_type = rule_type

    def get_allies_support(self, df_win):
        """
        Get support of lhs + rhs, who are allies

        :param df_win:  (DataFrame) dataframe containing radiant and dire heros in the win side
                         df should contain table:  hero_1 ... hero_5

        :return: allies support support
        """

        total_count = df_win.shape[0]
        df = df_win.loc[:, 'hero_1':'hero_5']
        allies = self._lhs + self._rhs

        row_count = df.isin(allies).sum(axis=1)
        support_count = (row_count >= len(self._lhs)).sum()

        return support_count * 1.0 / total_count

    def get_allies_win_rate(self, df_win, df_lose):
        """

        :param df_win:  (DataFrame) dataframe containing radiant and dire heros in the win side
                         df should contain table:  hero_1 ... hero_5

        :param df_lose: (DataFrame) dataframe containing radiant and dire heros in the lose side
                         df should contain table:  hero_1 ... hero_5

        :return: allies win rate
        """

        support_win = self.get_allies_support(df_win)
        support_lose = self.get_allies_support(df_lose)

        return support_win * 1.0 / (support_win + support_lose)

    def get_win_support(self, df_match, radiant, dire, winner):
        """
        Get support of rule: lhs ==> rhs
        if winner is 1, the win side is radiant, else win side is dire

        :param df_match:    (DataFrame) df of game matches
                            df table should contain: winner | radiant_hero_1 ... radiant_hero_5 | dire_hero_1 ... dire_hero_5

        :param radiant:     (list) radiant heros
        :param dire:        (list) dire heros
        :param winner:      winner of one match: 1 radiant win, -1 dire win
        :return: win support
        """
        if radiant is not None and dire is not None:
            len_rule = len(radiant + dire)
            df = pd.concat([df_match.loc[:, "winner"] == winner,
                            df_match.loc[:, "radiant_hero_1":"radiant_hero_5"].isin(radiant),
                            df_match.loc[:, "dire_hero_1":"dire_hero_5"].isin(dire)],
                           axis=1)

        elif radiant is not None and dire is None:
            len_rule = len(radiant)
            df = pd.concat([df_match.loc[:, "winner"] == winner,
                            df_match.loc[:, "radiant_hero_1":"radiant_hero_5"].isin(radiant)],
                           axis=1)

        elif radiant is None and dire is not None:
            len_rule = len(dire)
            df = pd.concat([df_match.loc[:, "winner"] == winner,
                            df_match.loc[:, "dire_hero_1":"dire_hero_5"].isin(dire)],
                           axis=1)

        else:
            raise Exception('heros in radiant and dire are all None!')

        win_support = (df.sum(axis=1) >= len_rule + 1).sum()
        return win_support

    def get_enemies_confidence(self, df_match):
        """
        Get confidence of enemies based association rule -e ==> r

        :param df_match:    (DataFrame) df of game matches
                            df table should contain: winner | radiant_hero_1 ... radiant_hero_5 | dire_hero_1 ... dire_hero_5

        :return: confidence of rule -e ==> r
        """
        rhs_win_support = 0
        lhs_lose_support = 0

        # support of rule: -e ==> r
        rhs_win_support += self.get_win_support(df_match,
                                                radiant=self._lhs,
                                                dire=self._rhs,
                                                winner=-1)

        rhs_win_support += self.get_win_support(df_match,
                                                radiant=self._rhs,
                                                dire=self._lhs,
                                                winner=1)

        # lhs are enemies, who should lose
        # support of -e
        lhs_lose_support += self.get_win_support(df_match,
                                                 radiant=self._lhs,
                                                 dire=None,
                                                 winner=-1)

        lhs_lose_support += self.get_win_support(df_match,
                                                 radiant=None,
                                                 dire=self._lhs,
                                                 winner=1)

        confidence = rhs_win_support * 1.0 / lhs_lose_support
        # print("rhs_win_support: {}, lhs_lose_support: {}, confidence: {}"
        #       .format(rhs_win_support, lhs_lose_support, confidence))
        return confidence

    def get_counter_coefficient(self, df_match):
        """
        Get counter coefficient of association rule: -e ==> r

        :param df_match:    (DataFrame) df of game matches
                            df table should contain: winner | radiant_hero_1 ... radiant_hero_5 | dire_hero_1 ... dire_hero_5

        :return: counter coefficient
        """
        # lhs are enemies, who should lose
        rhs_win_support = 0

        lhs = self._lhs
        rhs = self._rhs

        # support of rule: -e ==> r
        rhs_win_support += self.get_win_support(df_match,
                                                radiant=lhs,
                                                dire=rhs,
                                                winner=-1)

        rhs_win_support += self.get_win_support(df_match,
                                                radiant=rhs,
                                                dire=lhs,
                                                winner=1)

        # rhs are enemies, who should lose
        lhs_win_support = 0

        lhs = self._rhs
        rhs = self._lhs

        # support of rule: -r ==> e
        lhs_win_support += self.get_win_support(df_match,
                                                radiant=lhs,
                                                dire=rhs,
                                                winner=-1)

        lhs_win_support += self.get_win_support(df_match,
                                                radiant=rhs,
                                                dire=lhs,
                                                winner=1)

        # print("lhs: {}, rhs: {}".format(self._lhs, self._rhs))

        counter_coefficient = \
            rhs_win_support * 1.0 / (rhs_win_support + lhs_win_support)
        # print("rhs_win_support: {}, lhs_win_support: {}, counter_coefficient: {}"
        #       .format(rhs_win_support, lhs_win_support, counter_coefficient))

        return counter_coefficient


class AssociationRule(RuleMetrics):

    def __init__(self, lhs, rhs, rule_type):

        RuleMetrics.__init__(self, lhs, rhs, rule_type)

        # metrics of rule
        # If it is based on allies, the metrics are allies_support and allies_win_rate
        self.allies_support = None
        self.allies_win_rate = None

        # If it is based on enemies, the metrics are enemies_confidence and counter_coefficient
        self.enemies_confidence = None
        self.counter_coefficient = None

    def get_lhs(self):
        return self._lhs

    def get_rhs(self):
        return self._rhs

    def get_rule_type(self):
        return self._rule_type

    def compute_metrics(self, df_win, df_lose, df_match):
        """
        Compute and set rule metrics according to the type of rule
        If it is based on allies, the metrics are allies_support and allies_win_rate
        If it is based on enemies, the metrics are enemies_confidence and counter_coefficient

        :param df_win:  (DataFrame) dataframe containing radiant and dire heros in the win side
                         df should contain table:  hero_1 ... hero_5

        :param df_lose: (DataFrame) dataframe containing radiant and dire heros in the lose side
                         df should contain table:  hero_1 ... hero_5

        :param df_match:    (DataFrame) df of game matches
                            df table should contain: winner | radiant_hero_1 ... radiant_hero_5 | dire_hero_1 ... dire_hero_5

        """
        if self._rule_type == "allies":
            self.allies_support = self.get_allies_support(df_win)
            self.allies_win_rate = self.get_allies_win_rate(df_win, df_lose)

        elif self._rule_type == "enemies":
            self.enemies_confidence = self.get_enemies_confidence(df_match)
            self.counter_coefficient = self.get_counter_coefficient(df_match)

        else:
            raise Exception("Check rule type! It should be allies or enemies!")

    def set_metrics(self,
                    allies_support=None,
                    allies_win_rate=None,
                    enemies_confidence=None,
                    counter_coefficient=None):
        """
        Set metrics of rule
        If it is based on allies, the metrics are allies_support and allies_win_rate
        If it is based on enemies, the metrics are enemies_confidence and counter_coefficient

        :param allies_support:      (float) support of allies based rule
        :param win_rate:            (float) win rate of allies based rule
        :param enemies_confidence:  (float) confidence of enemies based rule
        :param counter_coefficient: (float) counter coefficient of enemies based rule
        """
        if self._rule_type == "allies":
            if enemies_confidence is not None or counter_coefficient is not None:
                raise Exception("Can not set enemies based metrics for allies based rule")

            self.allies_support = allies_support
            self.allies_win_rate = allies_win_rate

        elif self._rule_type == "enemies":
            if allies_support is not None or allies_win_rate is not None:
                raise Exception("Can not set enemies based metrics for allies based rule")

            self.enemies_confidence = enemies_confidence
            self.counter_coefficient = counter_coefficient

        else:
            raise Exception("Check rule type! It should be allies or enemies!")


if __name__ == "__main__":
    df_radiant_win_radiant_heros = pd.read_csv('radiant_win_radiant_heros.csv')
    df_dire_win_radient_heros = pd.read_csv('dire_win_radiant_heros.csv')
    df_radiant_win_match = pd.read_csv('radiant_win_match.csv')
    lhs = ['pudge']
    rhs = ['lycan']

    rule = AssociationRule(lhs, rhs, "enemies")

    print("TEST===============================TEST\n"
          "lhs: {}      rhs: {}\n".format(lhs, rhs))

    rule.compute_metrics(df_radiant_win_radiant_heros,
                         df_dire_win_radient_heros,
                         df_radiant_win_match)

    print(rule.allies_support)
    print(rule.allies_win_rate)
    print(rule.enemies_confidence)
    print(rule.counter_coefficient)

    print('allies support: {}'
          .format(rule.get_allies_support(df_radiant_win_radiant_heros)))

    print('win_rate: {}'
          .format(rule.get_allies_win_rate(df_radiant_win_radiant_heros,
                                           df_dire_win_radient_heros)))

    print('confidence: {}'
          .format(rule.get_enemies_confidence(df_radiant_win_match)))

    print('counter coefficient: {}'
          .format(rule.get_counter_coefficient(df_radiant_win_match)))









