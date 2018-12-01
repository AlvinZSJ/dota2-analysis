import unittest
from src.hero import Hero


class HeroTestCase(unittest.TestCase):

    def setUp(self):
        self.heroes_json_file_path = '../data/heroes.json'
        self.heroes = Hero.load_heroes_data(self.heroes_json_file_path)

    def test_load_heroes_data(self):
        self.assertEqual(len(self.heroes), self.heroes['zuus'].id - 1)  # id '24' is missing

    def test_get_hero_by_name(self):
        # Test to get hero 'antimage' (id 1)
        hero1 = Hero.get_hero_by_name(self.heroes, 'antimage')
        self.assertEqual('antimage', hero1.name)
        self.assertEqual(1, hero1.id)
        self.assertEqual('Anti-Mage', hero1.localized_name)
        # Test to get a hero whose name is not in heores
        hero2 = Hero.get_hero_by_name(self.heroes, 'Not a hero name')
        self.assertIsNone(hero2)
        # Test to get hero 'arc_warden' (id 113)
        hero3 = Hero.get_hero_by_name(self.heroes, 'arc_warden')
        self.assertEqual('arc_warden', hero3.name)
        self.assertEqual(113, hero3.id)
        self.assertEqual('Arc Warden', hero3.localized_name)

    def test_get_hero_by_id(self):
        # Test to get hero 'antimage' (id 1)
        hero1 = Hero.get_hero_by_id(self.heroes, 1)
        self.assertEqual('antimage', hero1.name)
        self.assertEqual(1, hero1.id)
        self.assertEqual('Anti-Mage', hero1.localized_name)
        # Test to get hero 'arc_warden' (id 113)
        hero2 = Hero.get_hero_by_id(self.heroes, 113)
        self.assertEqual('arc_warden', hero2.name)
        self.assertEqual(113, hero2.id)
        self.assertEqual('Arc Warden', hero2.localized_name)
        # Test to get a hero whose id is 24 (which is missing!)
        with self.assertRaises(AssertionError):
            Hero.get_hero_by_id(self.heroes, 24)
        # Test to get heroes whose id are out of range
        with self.assertRaises(AssertionError):
            Hero.get_hero_by_id(self.heroes, 114)

        with self.assertRaises(AssertionError):
            Hero.get_hero_by_id(self.heroes, 0)

    def test_is_match(self):
        self.assertTrue(Hero.is_match(self.heroes, 1, 'antimage'))
        self.assertFalse(Hero.is_match(self.heroes, 2, 'antimage'))
        with self.assertRaises(AssertionError):
            Hero.is_match(self.heroes, 0, 'Non-existed hero')
