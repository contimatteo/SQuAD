DoD = {}
def insert_dict(dict_key):
    if dict_key not in DoD.keys():
        DoD[dict_key] = {}

def df_apply_function_with_dict(df, function, dict_key, key, **kwargs):
    insert_dict(dict_key)
    dictionary = get_dict()[dict_key]

    def __apply(df_row):
        if df_row[key] not in dictionary.keys():
            dictionary[df_row[key]] = function(df_row, **kwargs)
        return dictionary[df_row[key]]

    return df.apply(lambda x: __apply(x), axis = 1)

def df_apply_function_with_dict_2(df, function1, function2, dict_key, key, **kwargs):
    insert_dict(dict_key)
    dictionary = get_dict()[dict_key]

    def __apply(df_row):
        #key=function2(df_row[key1])
        if df_row[key] not in dictionary.keys():
            dictionary[df_row[key]] = function2(df_row[key], **kwargs)
        return dictionary[df_row[key]]

    return df.apply(lambda x: __apply(x), axis = 1)

def get_dict():
    return DoD