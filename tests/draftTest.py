import unittest
import random
from src.hero import Hero
from src.draft import Draft


class DraftTestCase(unittest.TestCase):

    def setUp(self):
        self.heroes_json_file_path = '../data/heroes.json'
        self.all_heroes = Hero.load_heroes_data(self.heroes_json_file_path)

    def test_init(self):
        heroes = [Hero.get_hero_by_id(self.all_heroes, i) for i in range(1, 4)]
        draft1 = Draft(heroes)
        self.assertEqual(3, draft1.get_heroes_num())

        heroes = [Hero.get_hero_by_id(self.all_heroes, i) for i in range(1, 7)]
        with self.assertRaises(OverflowError):
            draft2 = Draft(heroes)

    def test_add_hero(self):
        draft = Draft()
        hero_ids = list()
        i = 0
        while not draft.is_full():
            self.assertEqual(i, draft.get_heroes_num())
            hero_id = random.randint(1, 113)
            if hero_id != 24:
                draft.add_hero(Hero.get_hero_by_id(self.all_heroes, hero_id))
                hero_ids.append(hero_id)
                i += 1

        self.assertEqual(i, draft.get_heroes_num())

        with self.assertRaises(TypeError):
            draft.add_hero(1)

        with self.assertRaises(OverflowError):
            draft.add_hero(Hero(1))

        # TEST CASE NOT ALL COVERED!!! MORE TO BE DONE!!!
