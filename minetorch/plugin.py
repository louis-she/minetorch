class Plugin():

    def __init__(self):
        self._name = self.__class__.__name__
        self.trainer = None

    def before_hook(self, hook_name, payload):
        return True

    def set_trainer(self, trainer):
        self.trainer = trainer

    def notify(self, message, _type='info'):
        message = f"[{self._name}] {message}"
        self.trainer(message, _type)

    def __getattr__(self, key):
        if self.trainer is None or key not in self.trainer.__dict__:
            raise AttributeError
        return self.trainer[key]
