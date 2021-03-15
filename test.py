from functools import wraps


def decorator_create(event_name):
    def decorator(func):
        func.__minetorch_event_name__ = event_name
        print(func.__dict__)

        @wraps(func)
        def inner(self, *args, **kwargs):
            return func(self, *args, **kwargs)
        return inner
    return decorator


class Test():

    @decorator_create("drawer.before_epoch_end")
    def hola(self):
        pass


t = Test()
t.hola()

print([method_name for method_name in dir(t) if callable(getattr(t, method_name))])
