from pathlib import Path

###


class LocalStorageDirectoryManager():

    def __init__(self) -> None:
        self.root = None
        self.tmp = None
        self.data = None
        self.logs = None
        self.datasets = None
        self.models = None

        self.__initialize()
        self.__mkdirs()
        self.__validate()

    #

    def __initialize(self) -> None:
        self.root = Path(__file__).parent.parent.parent

        self.tmp = self.root.joinpath('tmp')
        self.data = self.root.joinpath('data')

        self.logs = self.tmp.joinpath('logs')

        self.datasets = self.data.joinpath('datasets')
        self.models = self.data.joinpath('models')

    def __mkdirs(self) -> None:
        self.tmp.mkdir(exist_ok=True)
        self.data.mkdir(exist_ok=True)

        self.logs.mkdir(exist_ok=True)

        self.datasets.mkdir(exist_ok=True)
        self.models.mkdir(exist_ok=True)

    def __validate(self) -> None:
        assert self.root.is_dir()

        assert self.tmp.is_dir()
        assert self.data.is_dir()

        assert self.logs.is_dir()

        assert self.datasets.is_dir()
        assert self.models.is_dir()


###


class LocalStorageManager():

    def __init__(self) -> None:
        self.dirs = LocalStorageDirectoryManager()
