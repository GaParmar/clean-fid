
## Compiling locally
```
    python setup.py bdist_wheel
```

## Installing locally
```
    pip uninstall clean-fid
    pip install dist/clean_fid-0.1.35-py3-none-any.whl
```

## Run the tests locally
 - run `bash tests/tests_main.sh`
 - ensure that there are not Errors thrown in the logfile


## Push to pip
```
python -m twine upload dist/clean_fid-0.1.35-py3-none-any.whl
```