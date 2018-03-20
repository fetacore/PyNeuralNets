# Feedforward Neural Nets
To test the algo you need python3 (recommended in a virtualenv) and once you have done that just do
```
source <virtualenv Folder>/bin/activate
pip install -r requirements.txt
```

In order to run an example in cython you do
```
python setup.py build_ext --inplace && python test.py
```
This will create the .c files which we will use with emsdk to generate the WebAssembly executable.
