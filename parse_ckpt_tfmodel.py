import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

#首先，使用tensorflow自带的python打包库读取模型
model_reader = pywrap_tensorflow.NewCheckpointReader(r"F:/myself/mtcnn/mtcnn_tf-master/mtcnn_tf-master/tmp/model/pnet/pnet-30")

#然后，使reader变换成类似于dict形式的数据
var_dict = model_reader.get_variable_to_shape_map()

for key in var_dict:
    print("variable name: ", key)
    print(model_reader.get_tensor(key))
