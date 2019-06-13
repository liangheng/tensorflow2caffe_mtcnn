import caffe
import numpy as np

# 使输出的参数完全显示
# 若没有这一句，因为参数太多，中间会以省略号“……”的形式代替
np.set_printoptions(threshold='nan')


MODEL_FILE = './det1.prototxt'
PRETRAIN_FILE = './det1.caffemodel'
params_txt = './det1_params.txt'
# MODEL_FILE = './pnet.prototxt'
# PRETRAIN_FILE = './pnet.caffemodel'
# params_txt = './pnet_params.txt'

# MODEL_FILE = './det3.prototxt'
# PRETRAIN_FILE = './det3.caffemodel'
# params_txt = './det3_params.txt'
# MODEL_FILE = './onet.prototxt'
# PRETRAIN_FILE = './onet.caffemodel'
# params_txt = './onet_params.txt'

# MODEL_FILE = './rnet.prototxt'
# PRETRAIN_FILE = './rnet.caffemodel'
# params_txt = './rnet_params.txt'

# MODEL_FILE = './det2.prototxt'
# PRETRAIN_FILE = './det2_half.caffemodel'
# params_txt = './det2_params.txt'
pf = open(params_txt, 'w')

# 让caffe以测试模式读取网络参数
net = caffe.Net(MODEL_FILE, PRETRAIN_FILE, caffe.TEST)

# 遍历每一层
for param_name in net.params.keys():
    try:
        if 'relu' in param_name:
            continue
        if 'PReLU' in param_name:
            continue
        #权重参数
        weight = net.params[param_name][0].data
        #偏置参数
        bias = net.params[param_name][1].data

        #该层在prototxt文件中对应“top”的名称
        pf.write(param_name)
        pf.write('\n')

        # 写权重参数
        pf.write(param_name + '_weight:')
        # 权重参数是多维数组，为了方便输出，转为单列数组
        pf.write(str(weight.shape))
        pf.write('\n')
        #pf.write(str(weight))
        #np.savetxt(params_txt, weight)
        weight.shape = (-1, 1)

        for w in weight:
            #pf.write('%ff, ' % w)
            pf.write('%.10f, ' % w)

        # 写偏置参数
        pf.write('\n' + param_name + '_bias:')
        pf.write(str(bias.shape))
        pf.write('\n')
        #pf.write(str(bias))
        # 偏置参数是多维数组，为了方便输出，转为单列数组
        bias.shape = (-1, 1)
        for b in bias:
            #pf.write('%ff, ' % b)
            pf.write('%.10f, ' % b)
        pf.write('\n\n')
    except KeyError:
        pf.close()
pf.close()