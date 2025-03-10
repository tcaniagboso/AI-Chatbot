class Singleton:
    ''' Represents Singleton pattern'''
    
    _instances = {}

    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__new__(cls)
            cls._instances[cls].__initialized = False  # Track init status
        return cls._instances[cls]

    def __init__(self):
        if self.__initialized:
            return
        self.__initialized = True
        self._init_singleton()

    def _init_singleton(self):
        pass  # Child classes put their init logic here