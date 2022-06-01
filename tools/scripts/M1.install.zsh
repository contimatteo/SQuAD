#

pip install -U pip
pip install wheel

#

export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1   

pip install cython
pip install --no-binary=h5py h5py
pip install grpcio

#

is_pip_package_installed()
{
    package_name=$1
    echo $(python -c "import pkgutil; print(1 if pkgutil.find_loader('$package_name') else 0)")
}

if [[ $(is_pip_package_installed 'pandas') -eq '0' ]]
then
    # pip install git+git://github.com/pandas-dev/pandas.git
    git clone --depth 1 https://github.com/pandas-dev/pandas.git
    cd pandas
    python3 setup.py install
    cd ..
    rm -rf pandas
fi

# 

pip install -r ./tools/requirements/M1.txt 
