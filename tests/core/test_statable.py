from typing import OrderedDict
import pytest
from minetorch.core import Statable


class FooStatable(Statable):

    _state_dict_all_req_keys = ("key1", "key2")


def test_statable_get_state():
    foo_statable = FooStatable()
    foo_statable.state["key1"] = "value1"
    foo_statable.state["key2"] = "value2"

    state_dict = foo_statable.state_dict()
    assert state_dict["minetorch.core.FooStatable.key1"] == "value1"
    assert state_dict["minetorch.core.FooStatable.key2"] == "value2"


def test_statable_restore_state():
    test_statable = FooStatable()

    with pytest.raises(ValueError):
        state = {"key1": "value1", "key2": "value2"}
        test_statable.load_state_dict(state)

    with pytest.raises(ValueError):
        state = {"minetorch.core.FooStatable.key1": "value1"}
        test_statable.load_state_dict(state)

    state = {
        "minetorch.core.FooStatable.key1": "value1",
        "minetorch.core.FooStatable.key2": "value2",
        "minetorch.core.FooStatable.key3": "value3",
        "minetorch.Bar.key4": "value4"
    }
    test_statable.load_state_dict(state)
    assert test_statable.state == OrderedDict({
        "key1": "value1",
        "key2": "value2",
        "key3": "value3"
    })

