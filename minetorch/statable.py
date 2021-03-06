class Statable:
    def load_state_dict(self):
        raise NotImplementedError

    def state_dict(self):
        raise NotImplementedError
