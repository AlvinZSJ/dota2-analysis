import json


class Hero(object):

    def __init__(self, hero_id, hero_name=None, localized_name=None):
        self._id = hero_id
        self._name = hero_name
        self._localized_name = localized_name
        self._frequency = 0
        self._frequent_rate = 0.0
        self._win_rate = 0.0

    #################### Getter and Setter ####################
    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, hero_id):
        self._id = hero_id

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, hero_name):
        self._name = hero_name

    @property
    def localized_name(self):
        return self._localized_name

    @localized_name.setter
    def localized_name(self, localized_name):
        self._localized_name = localized_name

    @property
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, frequency):
        self._frequency = frequency

    @property
    def frequent_rate(self):
        return self._frequency / 'TODO'

    @frequent_rate.setter
    def frequent_rate(self, frequent_rate):
        self._frequent_rate = frequent_rate

    @property
    def win_rate(self):
        return self._win_rate

    @win_rate.setter
    def win_rate(self, win_rate):
        self._win_rate = win_rate

    #################### static methods ####################

    @staticmethod
    def get_hero_by_id(heroes, hero_id):
        assert hero_id != 24, "Input hero id '24' is not in range!"
        assert 0 < hero_id <= len(heroes)+1, "Input hero id is not in range!"
        for hero in heroes.values():
            if hero.id == hero_id:
                return hero
        return None

    @staticmethod
    def get_hero_by_name(heroes, hero_name):
        try:
            hero = heroes[hero_name]
        except KeyError:
            hero = None
            print('Can not found hero, maybe hero_name is wrong.')
        return hero

    @staticmethod
    def is_match(heroes, hero_id, hero_name):
        hero = Hero.get_hero_by_name(heroes, hero_name)
        assert hero is not None, 'Can not found hero by name'
        return hero_id == hero.id

    @staticmethod
    def load_heroes_data(heroes_json_file):
        with open(heroes_json_file) as f:
            json_data = f.read()
        assert json_data is not None, 'Load heroes json file FAILED.'
        heroes_data = json.loads(json_data)['heroes']

        # sort on hero id
        sorted_heroes = sorted(heroes_data, key=lambda x: x['id'])
        heroes = dict()
        for hero in sorted_heroes:
            hero_id = hero['id']
            hero_name = hero['name']
            hero_localized_name = hero['localized_name']
            heroes[hero_name] = Hero(hero_id, hero_name, hero_localized_name)
        return heroes




