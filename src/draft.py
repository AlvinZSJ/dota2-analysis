from src.hero import Hero


class Draft(object):

    def __init__(self, heroes=None):
        self._heroes = list()
        if heroes is None:
            return

        for hero in heroes:
            if self.is_full():
                raise OverflowError("Draft is already full!")
            self._heroes.append(hero)

    @property
    def heroes(self):
        return self._heroes

    def get_heroes_num(self):
        return len(self.heroes)

    def is_empty(self):
        return len(self.heroes) == 0

    def is_full(self):
        return len(self.heroes) == 5

    def add_hero(self, hero):
        if not isinstance(hero, Hero):
            raise TypeError('You should add a Hero!')
        if self.is_full():
            raise OverflowError("Draft is already full!")
        self.heroes.append(hero)

    def remove_hero(self, hero):
        if not isinstance(hero, Hero):
            raise TypeError('You should specify a Hero!')
        self.heroes.remove(hero)

    def get_heroes(self):
        return self.heroes

    def get_hero_by_index(self, index):
        return self.heroes[index]
