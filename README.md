some proj notes:

* To run directly from powershell, an error pops out:
```c:/anaconda3/mingw/bin/../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/bin/ld.exe: cannot open output file C:\Users\daij12\AppData\Local\Theano\compiledir_Windows-7-6.1.7601-SP1-Intel64_Family_6_Model_61_Stepping_4_GenuineIntel-3.5.2-64\cutils_ext\cutils_ext.pyd: Permission denied
```

* So to run in a windows shell, please use 
```
cmd /k C:\Anaconda3\Scripts\activate
```
