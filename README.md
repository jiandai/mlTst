quick proj notes:
* programs
	
	* iris-sk.py: test scikit-learn /w iris dataset

	* keras1.py: keras test /w uci housing dataset

* theano test:

	* theano.test() on server:
```
Ran 408 tests in 38.413s
FAILED (SKIP=26, errors=80)
```

	* theano.test() on laptop:

```
Ran 408 tests in 834.216s
FAILED (SKIP=26, errors=80)
```

* issue on windows:

	* To run directly from powershell /w anaconda on windows, errors can pop out, e.g.

``` 
c:/anaconda3/mingw/bin/../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/bin/ld.exe: cannot open output file C:\Users\daij12\AppData\Local\Theano\compiledir_Windows-7-6.1.7601-SP1-Intel64_Family_6_Model_61_Stepping_4_GenuineIntel-3.5.2-64\cutils_ext\cutils_ext.pyd: Permission denied

ImportError: DLL load failed: The specified procedure could not be found.
```

	* So to run on windows, please set the shel first


```
cmd /k C:\Anaconda3\Scripts\activate
```

