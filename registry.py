class RegisterMachine(object):
    def __init__(self, name):
        # name of register
        self._name = name
        self._name_method_map = dict()

    def register(self, obj=None):
        # obj == None for function call register
        # otherwise for decorator way
        if obj != None:
            name = obj.__name__
            self._name_method_map[name] = obj
        else:

            def wrapper(func):
                name = func.__name__
                self._name_method_map[name] = func
                return func

            return wrapper

    def get(self, name):
        return self._name_method_map[name]


TAG_FUNCTIONS = RegisterMachine("tag_functions")
