import append_sys_path  # noqa: F401


def web(environ, start_response):
    from minetorch.web import app
    return app(environ, start_response)


def pusher(environ, start_response):
    from minetorch.pusher import app
    return app(environ, start_response)


__all__ = ['web', 'pusher']
