from windows import call_subprocess_Popen
x=call_subprocess_Popen(['g++', '-v'])
print(x.pid)
