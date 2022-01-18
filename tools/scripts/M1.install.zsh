#

pip install --no-binary=h5py h5py


export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1   
pip install grpcio


pip install cython
pip install --no-binary :all: --no-use-pep517 pandas
# pip install --no-binary :all: pandas


pip install -r requirements.M1.txt 
# pip install --no-cache-dir -r requirements.M1.txt 
