PYTHON_INCS=-I/usr/local/Cellar/python/3.7.4_1/Frameworks/Python.framework/Versions/3.7/include/python3.7m/
PYTHON_LIBS = /usr/local/Cellar/python/3.7.4_1/Frameworks/Python.framework/Versions/3.7/lib/libpython3.7.dylib

HTOOL_INCS=-I../include
BLAS_LIB=-framework Accelerate

all: htool_python.dylib

htool_python.dylib: htool_python.cpp
	mpic++ -std=c++11 ${BLAS_LIB} ${HTOOL_INCS} ${PYTHON_INCS} -shared  htool_python.cpp -o htool_python.dylib ${PYTHON_LIBS}
