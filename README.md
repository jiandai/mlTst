quick proj notes:

* ML programs
	
	* iris-sk.py: test scikit-learn /w iris dataset

	* keras1.py: uci housing dataset /w scikit-learn and keras

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

	* tstTheano.py
http://deeplearning.net/tutorial/lenet.html
	* MNIST
http://deeplearning.net/tutorial/code/convolutional_mlp.py
http://deeplearning.net/tutorial/code/mlp.py
http://deeplearning.net/tutorial/code/logistic_sgd.py

* tensorflow test:
	* install ```
pip install --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-0.12.1-cp35-cp35m-win_amd64.whl ```
	* Issue about "Cannot remove entries from nonexistent file c:\anaconda3\lib\site-packages\easy-install.pth"
		* download https://bootstrap.pypa.io/ez_setup.py
		* run python ez_setup.py
	* Issue with "FileNotFoundError: [WinError 2] The system cannot find the file specified: 'c:\\anaconda3\\lib\\site-packages\\setuptools-33.1.1-py3.5.egg'" though installation passed
	* mnist.py
		* error on aws ec2: W tensorflow/core/framework/op_kernel.cc:975] Resource exhausted: OOM when allocating tensor with shape[10000,28,28,32]
