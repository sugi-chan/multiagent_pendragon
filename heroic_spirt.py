

class HeroicSpirit():
    def __init__(self,
                 health=1000,
                 NP=None,
                 spot=1
                 ):
        super().__init__()
        self.health = health
        self.NP = NP
        self.spot = spot

    def skill_1(self):
        print('skill_1')
        output_dict = {'target': [self.spot],
                       'hp_boost': 1,
                       'np_boost': .01,
                       'critical_boost': .01}
        return output_dict

    def skill_2(self):
        print('skill_2')
        output_dict = {'target': [self.spot],
                       'hp_boost': 1,
                       'np_boost': .01,
                       'critical_boost': .01}
        return output_dict

    def skill_3(self):
        print('skill_3')
        output_dict = {'target': [self.spot],
                       'hp_boost': 1,
                       'np_boost': .01,
                       'critical_boost': .01}
        return output_dict
