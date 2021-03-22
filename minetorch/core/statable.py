from typing import Mapping, OrderedDict
from ignite.base import Serializable


class Statable(Serializable):

    def __init__(self):
        self._scope = f"{__package__}.{self.__class__.__name__}"
        self.state = OrderedDict()
        self._state_dict_all_req_keys = tuple(map(lambda k: self._scopped_key(k), self.__class__._state_dict_all_req_keys))

    def state_dict(self) -> OrderedDict:
        state = OrderedDict()
        for key, value in self.state.items():
            state[self._scopped_key(key)] = value
        return state

    def load_state_dict(self, state_dict: Mapping) -> None:
        super().load_state_dict(state_dict)
        for key, val in state_dict.items():
            if not key.startswith(self._scope):
                continue
            self.state[self._unscopped_key(key)] = val

    def _unscopped_key(self, key: str) -> str:
        return key.replace(f"{self._scope}.", "")

    def _scopped_key(self, key: str) -> str:
        if key.find(self._scope) != -1:
            return key
        return f"{self._scope}.{key}"


