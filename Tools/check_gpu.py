from pynvml import *
nvmlInit()
 
deviceCount = nvmlDeviceGetCount()#几块显卡
print(deviceCount)
 
for i in range(deviceCount):
    handle = nvmlDeviceGetHandleByIndex(i)
    print ("Device", i, ":", nvmlDeviceGetName(handle)) #具体是什么显卡