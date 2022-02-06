from utils.data import get_default_raw_file_name


class Configuration:

    def __init__(self):
        self.argv_json_complete_name = None
        self.previous_debug_mode = False

    def get_argv_json_complete_name(self):
        return self.argv_json_complete_name

    def set_argv_json_complete_name(self, json_name, debug_mode=False):
        if json_name is None:
            self.argv_json_complete_name = get_default_raw_file_name()
        else:
            self.argv_json_complete_name = json_name
        self.previous_debug_mode = debug_mode

    def argv_changed(self, json_name, debug_mode):
        if debug_mode or self.previous_debug_mode:
            return True
        return self.argv_json_complete_name != json_name
