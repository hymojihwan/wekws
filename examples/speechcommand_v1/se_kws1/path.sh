MAIN_ROOT=$PWD/../../..
export PATH=$PWD:$PATH

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=../../../:$PYTHONPATH

if [ -f "${MAIN_ROOT}"/tools/activate_python.sh ]; then
    . "${MAIN_ROOT}"/tools/activate_python.sh
else
    echo "[INFO] "${MAIN_ROOT}"/tools/activate_python.sh is not present"
fi