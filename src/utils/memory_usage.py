import os
import psutil


def memory_usage():
    pid = os.getpid()
    python_process = psutil.Process(pid)
    memoryUse = python_process.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory use:', memoryUse, " gigabytes")
