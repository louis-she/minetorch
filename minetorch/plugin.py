class Plugin():

    def __init__(self):
        self.name = self.__class__.__name__
        self.miner = None

    def before_hook(self, hook_name, payload):
        return True

    def set_miner(self, miner):
        self.miner = miner

    def notify(self, message, _type='info'):
        message = f"[{self.name}] {message}"
        self.miner.notify(message, _type)

    def __getattr__(self, key):
        if self.miner is None or key not in self.miner.__dict__:
            raise AttributeError(key)
        return getattr(self.miner, key)

