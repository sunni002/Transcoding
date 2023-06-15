#快速入门
import mindspore
from mindspore import nn
from mindspore.dataset import vision, transforms
from mindspore.dataset import MnistDataset
#1.处理数据集
# Download data from open datasets 下载mnist数据集
# from download import download
#
# url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
#       "notebook/datasets/MNIST_Data.zip"
# path = download(url, "./", kind="zip", replace=True)
#查看数据类型
train_dataset = MnistDataset("MNIST_Data/train", shuffle=False)
print(type(train_dataset))
#打印数据集中包含的数据列名，用于dataset的预处理。
train_dataset = MnistDataset('MNIST_Data/train')
test_dataset = MnistDataset('MNIST_Data/test')
print(train_dataset.get_col_names())#打印训练数据集列名
#MindSpore的dataset使用数据处理流水线（Data Processing Pipeline），
# 需指定map、batch、shuffle等操作。这里我们使用map对图像数据及标签进行变换处理，
# 然后将处理好的数据集打包为大小为64的batch。
def datapipe(dataset, batch_size):
    image_transforms = [
        vision.Rescale(1.0 / 255.0, 0),#基于给定的缩放和平移因子调整图像的像素大小。输出图像的像素大小为：output = image * rescale + shift。
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),#图像标准化（均值，标准差）
        vision.HWC2CHW()#图像通道转换
    ]
    label_transform = transforms.TypeCast(mindspore.int32)#将输入的Tensor转换为指定的数据类型。

    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    dataset = dataset.batch(batch_size)
    return dataset
# Map vision transforms and batch dataset
train_dataset = datapipe(train_dataset, 64)#函数执行
test_dataset = datapipe(test_dataset, 64)

#使用create_tuple_iterator或create_dict_iterator对数据集进行迭代
print("测试集预处理结果：（4DTensor格式）")
for image, label in test_dataset.create_tuple_iterator():
    print(f"Shape of image [N, C, H, W]: {image.shape} {image.dtype}")#batch,channel,height,width
    print(f"Shape of label: {label.shape} {label.dtype}")
    break
    #方法2：
for data in test_dataset.create_dict_iterator():
    print(f"Shape of image [N, C, H, W]: {data['image'].shape} {data['image'].dtype}")
    print(f"Shape of label: {data['label'].shape} {data['label'].dtype}")
    break




