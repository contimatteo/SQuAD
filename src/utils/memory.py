import os
import psutil
import numpy as np

###


class MemoryUtils:

    @staticmethod
    def __apply_casting(df, col, NAlist):
        ### make variables for Int, max and min
        IsInt = False
        mx = df[col].max()
        mn = df[col].min()

        ### test if column can be converted to an integer
        asint = df[col].fillna(0).astype(np.int64)
        result = (df[col] - asint)
        result = result.sum()
        if -0.01 < result < 0.01:
            IsInt = True

        ### Make Integer/unsigned Integer datatypes
        if IsInt:
            if mn >= 0:
                if mx < 255:
                    df[col] = df[col].astype(np.uint8)
                elif mx < 65535:
                    df[col] = df[col].astype(np.uint16)
                elif mx < 4294967295:
                    df[col] = df[col].astype(np.uint32)
                else:
                    df[col] = df[col].astype(np.uint64)
            else:
                if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)

                    ### Make float datatypes 32 bit
        else:
            df[col] = df[col].astype(np.float32)

    #

    @staticmethod
    def reduce_df_storage(df, debug=False):
        start_mem_usg = df.memory_usage().sum() / 1024**2

        if debug:
            print("Memory usage of properties dataframe is :", start_mem_usg, " MB")

        NAlist = []  ### Keeps track of columns that have missing values filled in.

        for col in df.columns:
            if df[col].dtype != object:  #### Exclude strings
                if debug:
                    print("******************************")
                    print("Column: ", col)
                    print("dtype before: ", df[col].dtype)

                MemoryUtils.__apply_casting(df, col, NAlist)

                if debug:
                    print("dtype after: ", df[col].dtype)
                    print("******************************")

        if debug:
            print("___MEMORY USAGE AFTER COMPLETION:___")
            mem_usg = df.memory_usage().sum() / 1024**2
            print("Memory usage is: ", mem_usg, " MB")
            print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")

        return df, NAlist

    @staticmethod
    def print_total_PID_usage():
        pid = os.getpid()
        python_process = psutil.Process(pid)
        memoryUse = python_process.memory_info()[0] / 2.**30  ### memory use in GB...I think
        print('memory use:', memoryUse, " gigabytes")
