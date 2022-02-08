from pathlib import Path

###


class LocalStorageDirectoryManager():

    def __init__(self) -> None:
        self.root = None
        self.tmp = None
        self.data = None
        self.logs = None
        self.data_raw = None
        self.data_processed = None
        self.data_checkpoints = None
        self.data_predictions = None

        self.__initialize()
        self.__mkdirs()
        self.__validate()

    #

    def __initialize(self) -> None:
        self.root = Path(__file__).parent.parent.parent

        self.tmp = self.root.joinpath('tmp')
        self.data = self.root.joinpath('data')

        self.logs = self.tmp.joinpath('logs')

        self.data_raw = self.data.joinpath('raw')
        self.data_processed = self.data.joinpath('processed')
        self.data_checkpoints = self.data.joinpath('checkpoints')
        self.data_predictions = self.data.joinpath('predictions')

    def __mkdirs(self) -> None:
        self.tmp.mkdir(exist_ok=True)
        self.data.mkdir(exist_ok=True)

        self.logs.mkdir(exist_ok=True)

        self.data_raw.mkdir(exist_ok=True)
        self.data_processed.mkdir(exist_ok=True)
        self.data_checkpoints.mkdir(exist_ok=True)
        self.data_predictions.mkdir(exist_ok=True)

    def __validate(self) -> None:
        assert self.root.is_dir()

        assert self.tmp.is_dir()
        assert self.data.is_dir()

        assert self.logs.is_dir()

        assert self.data_raw.is_dir()
        assert self.data_processed.is_dir()
        assert self.data_checkpoints.is_dir()
        assert self.data_predictions.is_dir()


###


class LocalStorageManager():

    def __init__(self) -> None:
        self.dirs = LocalStorageDirectoryManager()

    #

    @staticmethod
    def is_empty(directory: Path) -> bool:
        return any(Path(directory).iterdir())

    #

    def nn_checkpoint_url(self, model_name: str) -> Path:
        return self.dirs.data_checkpoints.joinpath(f"{model_name}.h5")

    def answers_predictions_url(self, dataset_name: str) -> Path:
        assert isinstance(dataset_name, str)
        return self.dirs.data_predictions.joinpath(f"answers.{dataset_name}.json")
