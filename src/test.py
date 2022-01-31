from utils.storage import LocalStorageManager

###

LocalStorage = LocalStorageManager()

###


def test():
    print()
    print("    root = ", str(LocalStorage.dirs.root))
    print("     tmp = ", str(LocalStorage.dirs.tmp))
    print("    logs = ", str(LocalStorage.dirs.logs))
    print("    data = ", str(LocalStorage.dirs.data))
    print("datasets = ", str(LocalStorage.dirs.datasets))
    print("  models = ", str(LocalStorage.dirs.models))
    print()


###

if __name__ == "__main__":
    test()
