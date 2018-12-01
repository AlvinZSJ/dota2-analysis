###############
# Author ZHANG Shenjia
#
###############

import numpy as np
import time
import random
import pandas as pd
from AssociationRule import *
from FPTree import *
from itertools import chain, combinations
import copy


class EnemyGenerator:
    """
    Generate one or five enemies randomly

    Generate one: randomly pick one from the Top k (10 recommended) most pciked heros
    Generate five: randomly pick 5 heroes as enemies from all available heroes

    """

    def __init__(self, df_all_heroes, k):
        """

        :param df_heroes: (DataFrame) df of all heroes (win & lose)
                          df should contain table:  hero_1 ... hero_5
        """
        self.k = k
        self.all_heroes = self.count_all_heroes(df_all_heroes)

    def gen_one_enemy(self, allies, curr_enemies):
        random_index = \
            int(np.round(np.random.uniform(0, self.k - 1 if self.k - 1 > 0 else 0, 1)))

        print("allies: {}, enemies: {}".format(allies, curr_enemies))

        exist_heroes = \
            allies + curr_enemies if curr_enemies is not None else allies

        top_k_enemies = list(set(self.all_heroes) - set(exist_heroes))

        print("random index {}, len enemies {}".format(random_index, len(top_k_enemies)))
        enemy = top_k_enemies[random_index]

        return [enemy]

    def gen_five_enemies(self, allies):

        non_chosen_heroes = list(set(self.all_heroes) - set(allies))

        shuffled_heroes = random.shuffle(non_chosen_heroes)

        return shuffled_heroes[:5]

    def count_all_heroes(self, df_heroes):
        """

        :param df_heroes: (DataFrame) df of all heroes (win & lose)
                          df should contain table:  hero_1 ... hero_5

        :return:          (list) heroes sorted in desc order according to the count
        """
        columns = ['hero_1', 'hero_2', 'hero_3', 'hero_4', 'hero_5']
        df_heroes = df_heroes[columns]

        dict_hero_count = dict()
        for column in columns:

            curr_dict_hero_count = \
                dict(df_heroes[column].value_counts())

            dict_hero_count = \
                self.merge_hero_count(dict_hero_count, curr_dict_hero_count)

        return list(dict_hero_count)

    @staticmethod
    def merge_hero_count(dict_hero_count_1, dict_hero_count_2):
        heroes_in_dict_1 = list(dict_hero_count_1)

        for hero in list(dict_hero_count_2):

            if hero in heroes_in_dict_1:
                dict_hero_count_1[hero] += dict_hero_count_2[hero]

            else:
                dict_hero_count_1[hero] = dict_hero_count_2[hero]

        return dict_hero_count_1


class Recommender:

    def __init__(self, freq_hero_sets, df_win, df_lose, df_match):
        # np.random.random(int(round(time.time())))
        self.freq_hero_sets = freq_hero_sets
        self.df_win = df_win
        self.df_lose = df_lose
        self.df_match = df_match

        self.rules_recomd_by_allies = list()
        self.rules_recomd_by_enemies = list()
        self.allies = list()
        self.enemies = list()

    def reset_recommender(self):
        self.allies = list()
        self.enemies = list()
        self.rules_recomd_by_allies = list()
        self.rules_recomd_by_enemies = list()

    def recommend(self,
                  k,
                  allies,
                  allies_metric_type,
                  enemies,
                  enemies_metric_type,
                  min_support_enemies):
        """

        :param k:                       (int) number of top heroes ranked by metrics who should be selected
        :param allies:
        :param allies_metric_type:
        :param enemies:
        :param enemies_metric_type:
        :return:
        """
        allies_powerset = self.get_power_set(allies)
        exist_allies = self.get_power_set(self.allies)

        # print("allies power set: {} \n"
              # "exist pwoer set: {}".format(allies_powerset, exist_allies))

        # remove power set of allies set which have been used for recommendation
        new_allies_powerset = \
            list(set(allies_powerset) - set(exist_allies))

        # print("new allies power set: {}".format(new_allies_powerset))

        self.allies = copy.deepcopy(allies)

        freq_hero_sets = \
            copy.deepcopy(self.freq_hero_sets)

        for subset in new_allies_powerset:
            allies_subset = list(subset)

            if not allies_subset:
                continue

            # duplicated rules may exist, drop them at ranking stage
            self.rules_recomd_by_allies += \
                self.get_recommend_based_on_allies(allies_subset)

        # since freq_hero_sets are pruned during
        # the process of finding rules recommended by allies,
        # so it need be reassigned
        self.freq_hero_sets = freq_hero_sets

        # For enemies, just find candidate rules for each newly added enemy
        new_enemies = list(set(enemies) - set(self.enemies))
        self.enemies = enemies

        for enemy in new_enemies:
            # duplicated rules may exist, drop them at ranking stage
            self.rules_recomd_by_enemies += \
                self.get_recommend_based_on_enemies(enemy, min_support_enemies)

        self.freq_hero_sets = freq_hero_sets

        heroes_recomd_by_allies, heroes_recomd_by_enemies = \
            self.rank_heroes(allies_metric_type=allies_metric_type,
                             enemies_metric_type=enemies_metric_type)

        print("len_self rule allies {}".format(len(self.rules_recomd_by_allies)))
        print("recommended ranked allies: {}, enemies: {}"
              .format(heroes_recomd_by_allies[:k], heroes_recomd_by_enemies[:k]))
        print("This time allies {}".format(allies))

        heroes_recomd = \
            list(set(heroes_recomd_by_allies[:k] + heroes_recomd_by_enemies[:k]))

        heroes_recomd = list(set(heroes_recomd) - set(self.allies + self.enemies))

        random_index = \
            int(np.round(np.random.uniform(0, len(heroes_recomd) - 1 if len(heroes_recomd) - 1 > 0 else 0, 1)))

        return heroes_recomd[random_index]

    # Different combination of metrics can be tested
    def rank_heroes(self,
                    allies_metric_type,
                    enemies_metric_type):

        assert allies_metric_type in ["support", "win_rate"], \
            "allies_metrics_type should be in [support, win_rate]"

        assert enemies_metric_type in ["confidence", "coefficient"], \
            "enemies_metric_type should be in [confidence, coefficient]"

        print("len self rule allies before sort {}"
              .format(len(self.rules_recomd_by_allies)))

        if allies_metric_type == "support":
            self.rules_recomd_by_allies = sorted(self.rules_recomd_by_allies,
                                        key=lambda rule: rule.allies_support,
                                        reverse=True)

        elif allies_metric_type == "win_rate":
            self.rules_recomd_by_allies = sorted(self.rules_recomd_by_allies,
                                        key=lambda rule: rule.allies_win_rate,
                                        reverse=True)

        else:
            raise ValueError("allies_metrics_type should be in [support, win_rate]")

        print("len self rule allies after sort {}"
              .format(len(self.rules_recomd_by_allies)))

        ranked_hero_allies = \
            map(lambda rule: rule.get_rhs()[0], self.rules_recomd_by_allies)

        if enemies_metric_type == "confidence":
            self.rules_recomd_by_enemies = sorted(self.rules_recomd_by_enemies,
                                                  key=lambda rule: rule.enemies_confidence,
                                                  reverse=True)

        elif enemies_metric_type == "coefficient":
            self.rules_recomd_by_enemies = sorted(self.rules_recomd_by_enemies,
                                                  key=lambda rule: rule.counter_coefficient,
                                                  reverse=True)

        else:
            raise ValueError("enemies_metric_type should be in [confidence, coefficient]")

        ranked_hero_enemies = \
            map(lambda rule: rule.get_rhs()[0], self.rules_recomd_by_enemies)

        return list(set(ranked_hero_allies)), list(set(ranked_hero_enemies))

    def get_recommend_based_on_allies(self, allies):
        num_allies = len(allies)

        all_candidate_rules = list()
        pruned_hero_sets = list()

        allies_not_recomd = copy.deepcopy(allies)

        for hero_set in self.freq_hero_sets:

            if len(allies_not_recomd) == 112:
                print("all allies have been used for recommendation")
                break

            if not all(hero in hero_set for hero in allies) or \
                                len(hero_set) <= num_allies:
                continue

            pruned_hero_sets.append(hero_set)
            recomd_candidates = \
                list(set(hero_set) - set(allies_not_recomd))

            for candidate in recomd_candidates:

                candidate = candidate.strip(" ")
                if candidate in allies_not_recomd:
                    print("candidate: {}, allies_not_recommend {}".format(candidate, allies_not_recomd))
                    continue

                allies_not_recomd.append(candidate)

                rule = AssociationRule(lhs=allies,
                                       rhs=[candidate],
                                       rule_type="allies")

                rule.compute_metrics(df_win=self.df_win,
                                     df_lose=self.df_lose,
                                     df_match=self.df_match)

                all_candidate_rules.append(rule)

        return all_candidate_rules

    def get_recommend_based_on_enemies(self, enemy, min_support):
        """

        :param enemy:           (str) name of one enemy in current enemies
        :param min_support:     (int) min support count of heroes who win the enemy
        :return:
        """
        df_lose = self.df_lose
        df_win = self.df_win
        df_match = self.df_match

        indexes = \
            df_lose.index[df_lose.isin([enemy]).any(axis=1)].tolist()

        # print("indexes {}".format(indexes))

        df_lose = df_lose.iloc[indexes, :]
        df_win = df_win.iloc[indexes, :]

        heroes_win = df_win.reset_index(drop=True).values.tolist()

        print("current heroes_win len {}".format(len(heroes_win)))

        min_support = int(round(len(heroes_win) * min_support))
        fp_tree = FPTree(heroes_win, min_support)
        freq_heroes = fp_tree.gen_freq_itemsets()

        print("win_heroes len: {}".format(len(freq_heroes)))

        candidates = \
            list(filter(lambda hero_set: len(hero_set) == 1, freq_heroes))

        print("enemies freq hero num: {}".format(len(candidates)))

        all_candidate_rules = list()
        for hero in candidates:
            # print("hero: {}, enemy: {}".format(hero, enemy))
            # radiant_win_indexes = \
            #     (df_match["winner"] == 1) & \
            #     (df_match.loc[:, "radiant_hero_1":"radiant_hero_5"].isin(hero).any(axis=1)) & \
            #     (df_match.loc[:, "dire_hero_1":"dire_hero_5"].isin([enemy]).any(axis=1))
            #
            # dire_win_indexes = \
            #     (df_match["winner"] == -1) & \
            #     (df_match.loc[:, "radiant_hero_1":"radiant_hero_5"].isin([enemy]).any(axis=1)) & \
            #     (df_match.loc[:, "dire_hero_1":"dire_hero_5"].isin(hero).any(axis=1))
            #
            # print("radiant_win_indexes sum : {}".format(radiant_win_indexes.sum()))
            # print("dire_win_indexes sum {}".format(dire_win_indexes.sum()))
            # df_match = df_match.loc[(radiant_win_indexes | dire_win_indexes)]

            # print("df_match==== {}".format(df_match))
            hero = hero[0].strip(" ")
            if hero in (self.allies + self.enemies):
                continue

            rule = AssociationRule(lhs=[enemy], rhs=[hero], rule_type="enemies")

            rule.compute_metrics(df_win=df_win, df_lose=df_lose, df_match=df_match)

            all_candidate_rules.append(rule)

        return all_candidate_rules

    def recommend_with_inc_enemies(self,
                                   allies,
                                   enemies,
                                   df_heroes,
                                   freq_enemies_select_k,
                                   recomd_select_k,
                                   allies_metric_type,
                                   enemies_metric_type,
                                   min_support_enemies):
        """

        :param init_ally:       (list) list of initialized allies
        :param df_heroes:
        :param k:
        :return:
        """
        # print("begin allies: {}".format(allies))
        # print("Initialize enemies generator")
        enemies_generator = EnemyGenerator(df_heroes,
                                           freq_enemies_select_k)

        while len(allies) < 5:
            print("Start recommendation. current allies: {}, current enemies: {}"
                  .format(allies, enemies))

            if not enemies:
                enemies += \
                    enemies_generator.gen_one_enemy(allies=allies,
                                                    curr_enemies=None)

            recomd_hero = self.recommend(k=recomd_select_k,
                                         allies=copy.deepcopy(allies),
                                         allies_metric_type=allies_metric_type,
                                         enemies=copy.deepcopy(enemies),
                                         enemies_metric_type=enemies_metric_type,
                                         min_support_enemies=min_support_enemies)

            # print("recomd hero: {}".format(recomd_hero))
            allies.append(recomd_hero)
            # print("allies_after_append: {}".format(allies))
            if len(enemies) == 5:
                continue

            enemies += \
                enemies_generator.gen_one_enemy(allies=allies,
                                                curr_enemies=enemies)

            # print("recommended hero: {}\n"
            #       "current allies: {}, current enemies {}".format(recomd_hero, allies, enemies))

        return allies, enemies


    def recommend_with_given_enemies(self, init_ally, df_heroes, k):
        """

        :param init_ally:      (list) one ally initialized randomly
        :param df_heroes:
        :param k:              (int) If generate only one enemy,
                                     it should be chosen from top k most frequent selected heroes
        :return:
        """

        enemies_generator = EnemyGenerator(df_heroes, k)
        enemies = enemies_generator.gen_five_enemies(init_ally)

        pass


    @staticmethod
    def get_power_set(origin_set):
        if not origin_set:
            return list()

        iter_power_set = chain.from_iterable(combinations(origin_set, r)
                                             for r in range(len(origin_set) + 1))
        return list(iter_power_set)


if __name__ == "__main__":

    min_support_allies = 0.0001
    min_support_enemies = 0.05
    ###########
    # Load and prepare data
    ###########
    print("Loading and preparing data...")
    sTime = time.time()

    df_dire_win_dire_heroes = pd.read_csv("../data/processed_data/dire_win_dire_heros.csv")
    df_radiant_win_radiant_heroes = pd.read_csv("../data/processed_data/radiant_win_radiant_heros.csv")

    df_win_heroes = pd.concat([df_dire_win_dire_heroes,
                               df_radiant_win_radiant_heroes],
                              axis=0,
                              sort=False)

    df_win_heroes = df_win_heroes.reset_index(drop=True)
    # df_win_heroes.to_csv("./inter_data/df_win_heroes.csv")

    df_dire_win_radiant_heroes = pd.read_csv("../data/processed_data/dire_win_radiant_heros.csv")
    df_radiant_win_dire_heroes = pd.read_csv("../data/processed_data/radiant_win_dire_heros.csv")

    df_lose_heroes = pd.concat([df_dire_win_radiant_heroes,
                                df_radiant_win_dire_heroes],
                               axis=0,
                               sort=False)

    df_lose_heroes = df_lose_heroes.reset_index(drop=True)
    # df_lose_heroes.to_csv("./inter_data/df_lose_heroes.csv")

    df_all_heroes = pd.concat([df_win_heroes, df_lose_heroes], axis=0)
    # df_all_heroes.to_csv("./inter_data/df_all_heroes.csv")

    all_heroes = df_all_heroes.reset_index(drop=True).values.tolist()

    fp_tree = FPTree(transactions=all_heroes,
                     min_support_count=int(round(min_support_allies * len(all_heroes))))

    freq_allies = fp_tree.gen_freq_itemsets()

    df_radiant_win_match = pd.read_csv("../data/processed_data/radiant_win_match.csv")
    df_dire_win_match = pd.read_csv("../data/processed_data/dire_win_match.csv")

    df_match = pd.concat([df_radiant_win_match,
                          df_dire_win_match],
                         axis=0,
                         sort=False)

    df_match = df_match.reset_index(drop=True)
    # df_match.to_csv("./inter_data/df_match.csv")
    print("df_match : {}".format(df_match))

    print("Data preparation complete. Time: {}".format(time.time() - sTime))

    ###########
    # Recommendation
    ###########

    print("Initialize Recommender...")
    sTime = time.time()
    recommender = Recommender(freq_hero_sets=freq_allies,
                              df_win=df_win_heroes,
                              df_lose=df_lose_heroes,
                              df_match=df_match)

    print("Recommender initialized. Time: {}"
          .format(time.time() - sTime))


    # for ally in
    with open("../data/processed_data/all_heroes.txt", "r") as f:
        line = f.read()
        all_heroes = line.split(",")

    allies = list()
    enemies = list()
    print("all_heroes: {}".format(all_heroes))
    allies_metric_type = "win_rate"
    enemies_metric_type = "coefficient"

    for i, hero in enumerate(all_heroes):
        if i == 100:
            break

        sTime = time.time()
        init_allies = [hero.strip(" ")]
        init_enemies = list()

        recomd_allies, recomd_enemies = \
            recommender.recommend_with_inc_enemies(allies=init_allies,
                                                   enemies=init_enemies,
                                                   df_heroes=df_all_heroes,
                                                   freq_enemies_select_k=20,
                                                   recomd_select_k=7,
                                                   allies_metric_type=allies_metric_type,
                                                   enemies_metric_type=enemies_metric_type,
                                                   min_support_enemies=min_support_enemies)

        allies.append(recomd_allies)
        enemies.append(recomd_enemies)

        recommender.reset_recommender()

        print(">>>>>>>>>>round<<<<<<<<<<<<< {}, Time {}".format(i + 1, time.time() - sTime))

        print("=" * 15 + "\nCurrent allies: {} \nCurrent enemies: {}"
              .format(init_allies, init_enemies))

        print("-" * 15 + "\nAllies after recommendation: {} \nEnemies after recommendation: {}\n"
              .format(recomd_allies, recomd_enemies) + "=" * 15)

    allies_columns = ["allies_hero_{}".format(i) for i in range(1, 6)]
    enemies_columns = ["enemies_hero_{}".format(i) for i in range(1, 6)]

    df_allies = pd.DataFrame(allies, columns=allies_columns)
    df_enemies = pd.DataFrame(enemies, columns=enemies_columns)
    df_recomd = pd.concat([df_allies, df_enemies], axis=1, sort=False)
    df_recomd.to_csv("./test_data/recomd_{}.csv".format(allies_metric_type + "_" + enemies_metric_type))




