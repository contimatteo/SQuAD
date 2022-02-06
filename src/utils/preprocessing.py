###

dod = {}

###


# obsolete

# def insert_dict(dict_key):
#     if dict_key not in dod.keys():
#         dod[dict_key] = {}
#
#
# def df_apply_function_with_dict(df, function, dict_key, key, **kwargs):
#     insert_dict(dict_key)
#     dictionary = get_dict()[dict_key]
#
#     def __apply(df_row):
#         if df_row[key] not in dictionary.keys():
#             dictionary[df_row[key]] = function(df_row, **kwargs)
#         return dictionary[df_row[key]]
#
#     return df.apply(__apply, axis=1)
#
#
# def get_dict():
#     return dod
