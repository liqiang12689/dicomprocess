import SimpleITK as sitk
import skimage.io as io


def read_img(path):
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)
    return data


# 显示一个系列图
def show_img(data):
    for i in range(data.shape[0]):
        io.imshow(data[i, :, :], cmap='gray')
        print(i)
        # io.show()


# 单张显示
# def show_img(ori_img):
#     io.imshow(ori_img[110], cmap='gray')
#     io.show()


# path = r'D:\myPython\My2dUnet\unet\data1\train\label\xuwenhua\xuwenhua-1.nii'  # 数据所在路径
# data = read_img(path)
# show_img(data)
