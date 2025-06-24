class Woopsies:
    # logging woopsies, errors, and info (progress and some outputs)
    def __init__(self):
        self.woopsies = []
        self.infos = []
        self.woopsie_calls = []
        self.info_calls = []
        self.woopsie_count = 0
        self.info_count = 0
        self.current_subID = ''

    def add_info(self, caller = 'not_specified', info = 'not_specified'):
        self.infos.append(info)
        self.info_calls.append(caller)
        self.info_count += 1

    def add_woopsie(self, caller = 'not_specified', woopsie = 'not_specified'):
        self.woopsies.append(woopsie)
        self.woopsie_calls.append(caller)
        self.woopsie_count += 1

    def all_woopsies(self):
        woopsies_list = [[caller, woopsie] for caller, woopsie in zip(self.woopsie_calls, self.woopsies)]
        return woopsies_list

    def all_info(self):
        infos_list = [[caller, info] for caller, info in zip(self.info_calls, self.infos)]
        return infos_list

    def get_subID(self):
        return self.current_subID
    
    def set_subID(self, subID):
        self.current_subID = subID

if __name__ == "__main__":
    print('not called')