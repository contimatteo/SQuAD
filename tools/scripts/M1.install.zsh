#

pip install --no-binary=h5py h5py


export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1   
pip install grpcio


pip install cython
pip install git+git://github.com/pandas-dev/pandas.git


pip install -r requirements.M1.txt 
