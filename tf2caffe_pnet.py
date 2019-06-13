import sys
import tensorflow as tf
import caffe
import numpy as np
import cv2
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

from tensorflow.python import pywrap_tensorflow
checkpoint_path = "./pnet/pnet-30"
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()


cf_prototxt = "./pnet.prototxt"
cf_model = "./pnet.caffemodel"

# cf_prototxt = "./rnet.prototxt"
# cf_model = "./rnet.caffemodel"


def tensor4d_transform(tensor):
    return tensor.transpose((3, 2, 0, 1))
def tensor2d_transform(tensor):
    return tensor.transpose((1, 0))

def tf2caffe(checkpoint_path,cf_prototxt,cf_model):
    net = caffe.Net(cf_prototxt, caffe.TRAIN)
    for key_i in var_to_shape_map:
        try:
            if 'data' in key_i:
                pass
            elif 'weights' in key_i:
                a = key_i.split('/')
                if (len(a) == 3):
                    key_caffe = a[0]
                    continue
                elif (len(a) == 2):
                    key_caffe = a[0]
                else:
                    key_caffe = a[1]
                if (reader.get_tensor(key_i).ndim == 2):
                    if key_caffe == 'conv4_1':
                    #if key_caffe == 'cls_fc':
                        weights = tensor2d_transform(reader.get_tensor(key_i))
                    elif key_caffe == 'conv4_2':
                    #elif key_caffe == 'bbox_fc':
                        weights = tensor2d_transform(reader.get_tensor(key_i))
                    elif key_caffe == 'conv4_3':
                    #elif key_caffe == 'landmark_fc':
                        weights = tensor2d_transform(reader.get_tensor(key_i))
                    else:
                        weights = tensor2d_transform(reader.get_tensor(key_i))
                else:
                    assert (reader.get_tensor(key_i).ndim == 4)
                    weights = tensor4d_transform(reader.get_tensor(key_i))

                net.params[key_caffe][0].data.flat = weights.flat
                print("convert key tf:{}".format(key_i))
            elif 'biases' in key_i:
                a = key_i.split('/')
                if (len(a) == 3):
                    key_caffe = a[0]
                    continue
                elif (len(a) == 2):
                    key_caffe = a[0]
                else:
                    key_caffe = a[0]
                net.params[key_caffe][1].data.flat = reader.get_tensor(key_i).flat
                print("convert key tf:{}".format(key_i))
            elif 'mean_rgb' in key_i:
                pass
            elif 'global' in key_i:
                pass
            else:
                #sys.exit("Warning!  Unknown tf:{}".format(key_i))
                print("----------Warning! Unknown tf:{}".format(key_i))
        except KeyError:
            #print("convert key tf:{}".format(key_i))
            pass
    net.save(cf_model)
    print("\n- Finished.\n")

if __name__ == "__main__":
    tf2caffe(checkpoint_path, cf_prototxt, cf_model)
