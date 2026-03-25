_model_entrypoints = {}

def register_decoder(fn=None, *, name=None):
    """Register a decoder builder. Key defaults to the module basename (e.g. file name); pass ``name=`` when several builders live in one module."""

    def _register(f):
        model_name = name if name is not None else f.__module__.split('.')[-1]
        _model_entrypoints[model_name] = f
        return f

    if fn is not None:
        return _register(fn)
    return _register

def model_entrypoints(model_name):
    return _model_entrypoints[model_name]

def is_model(model_name):
    return model_name in _model_entrypoints