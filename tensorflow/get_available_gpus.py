# ref: https://gist.github.com/jovianlin/b5b2c4be45437992df58a7dd13dbafa7
# ref: http://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
# Ver 20170416 by Jian
# no longer available in tf 1.x

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

print get_available_gpus()
