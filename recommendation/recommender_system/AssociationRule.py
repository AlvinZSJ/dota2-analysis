import pandas as pd


class AssociationRule:

    def __init__(self, lhs, rhs, rule_type):
        """

        :param lhs:         (list) heros in the left hand side of association rule
        :param rhs:         (list) heros in the right hand side of association rule
        :param rule_type:   (str)  type of rule: allies / enemies
        """
        self.lhs = lhs
        self.rhs = rhs
        self.rule_type = rule_type

    def get_allies_support(self, df_win):
        """
        Get support of lhs + rhs, who are allies

        :param df:  (DataFrame) dataframe containing radiant and dire heros
                    df should contain table:  hero_1 ... hero_5

        :return: allies support support
        """

        total_count = df_win.shape[0]
        df = df_win.loc[:, 'hero_1':'hero_5']
        allies = self.lhs + self.rhs

        row_count = df.isin(allies).sum(axis=1)
        support_count = (row_count >= len(self.lhs)).sum()

        return support_count * 1.0 / total_count

    def get_allies_win_rate(self, df_win, df_lose):
        """

        :param df_win: members of win side (radiant_win_radiant_hero, dire_win_dire_hero)
        :param df_lose: members of lose side (radiant_win_dire_hero, dire_win_radiant_hero)
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
        rhs_win_support += self.get_win_support(df_match, radiant=self.lhs, dire=self.rhs, winner=-1)
        rhs_win_support += self.get_win_support(df_match, radiant=self.rhs, dire=self.lhs, winner=1)

        # lhs are enemies, who should lose
        # support of -e
        lhs_lose_support += self.get_win_support(df_match, radiant=self.lhs, dire=None, winner=-1)
        lhs_lose_support += self.get_win_support(df_match, radiant=None, dire=self.lhs, winner=1)

        return rhs_win_support * 1.0 /lhs_lose_support

    def get_counter_coefficient(self, df_match):
        """
        Get counter coefficient of association rule: -e ==> r

        :param df_match:    (DataFrame) df of game matches
                            df table should contain: winner | radiant_hero_1 ... radiant_hero_5 | dire_hero_1 ... dire_hero_5

        :return: counter coefficient
        """
        # lhs are enemies, who should lose
        rhs_win_support = 0

        lhs = self.lhs
        rhs = self.rhs

        # support of rule: -e ==> r
        rhs_win_support += self.get_win_support(df_match, radiant=lhs, dire=rhs, winner=-1)
        rhs_win_support += self.get_win_support(df_match, radiant=rhs, dire=lhs, winner=1)

        # rhs are enemies, who should lose
        lhs_win_support = 0

        lhs = self.rhs
        rhs = self.lhs

        # support of rule: -r ==> e
        lhs_win_support += self.get_win_support(df_match, radiant=lhs, dire=rhs, winner=1)
        lhs_win_support += self.get_win_support(df_match, radiant=rhs, dire=lhs, winner=-1)

        return rhs_win_support * 1.0 / (rhs_win_support + lhs_win_support)


if __name__ == "__main__":
    df_radiant_win_radiant_heros = pd.read_csv('radiant_win_radiant_heros.csv')
    df_dire_win_radient_heros = pd.read_csv('dire_win_radiant_heros.csv')
    df_radiant_win_match = pd.read_csv('radiant_win_match.csv')
    lhs = ['bloodseeker']
    rhs = ['pudge']

    rule = AssociationRule(lhs, rhs, "allies")

    print("TEST===============================TEST\n"
          "lhs: {}      rhs: {}\n".format(lhs, rhs))

    print('allies support: {}'.format(rule.get_allies_support(df_radiant_win_radiant_heros)))
    print('win_rate: {}'.format(rule.get_allies_win_rate(df_radiant_win_radiant_heros, df_dire_win_radient_heros)))
    print('confidence: {}'.format(rule.get_enemies_confidence(df_radiant_win_match)))
    print('counter coefficient: {}'.format(rule.get_counter_coefficient(df_radiant_win_match)))





