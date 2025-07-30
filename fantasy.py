import torch
from torch import nn
from torch.nn import functional as F

def rotation():
    def _transform(images):
        # 获取图像的形状，忽略批次维度
        size = images.shape[1:]
        # 对图像进行 0 次和 2 次 90 度旋转，并将结果堆叠在一起
        # torch.rot90(images, k, (2, 3)) 对图像进行 k 次 90 度旋转，(2, 3) 表示旋转的维度。
        # torch.stack 将旋转后的图像堆叠在一起，view(-1, *size) 将结果展平
        return torch.stack([torch.rot90(images, k, (2, 3)) for k in range(4)], 1).view(-1, *size)

    return _transform, 4

#同样的旋转，但是只选择两次，一次0次选择，一次两次选择
def rotation2():
    def _transform(images):
        size = images.shape[1:]
        return torch.stack([torch.rot90(images, k, (2, 3)) for k in [0, 2]], 1).view(-1, *size)

    return _transform, 2

def color_perm():
    def _transform(images):
        size = images.shape[1:]
        # 对图像的颜色通道进行置换，并将结果堆叠在一起 五次变化，包括本身共六张
        images = torch.stack([images,
                              torch.stack([images[:, 0, :, :], images[:, 2, :, :], images[:, 1, :, :]], 1),
                              torch.stack([images[:, 1, :, :], images[:, 0, :, :], images[:, 2, :, :]], 1),
                              torch.stack([images[:, 1, :, :], images[:, 2, :, :], images[:, 0, :, :]], 1),
                              torch.stack([images[:, 2, :, :], images[:, 0, :, :], images[:, 1, :, :]], 1),
                              torch.stack([images[:, 2, :, :], images[:, 1, :, :], images[:, 0, :, :]], 1)], 1).view(-1, *size)
        #contiguous() 确保张量在内存中是连续的，方便后续操作
        return images.contiguous()

    return _transform, 6

#和上面一样，只变化了三个
def color_perm3():
    def _transform(images):
        size = images.shape[1:]
        images = torch.stack([images,
                              torch.stack([images[:, 1, :, :], images[:, 2, :, :], images[:, 0, :, :]], 1),
                              torch.stack([images[:, 2, :, :], images[:, 0, :, :], images[:, 1, :, :]], 1)], 1).view(-1, *size)
        return images.contiguous()

    return _transform, 3

#选择+改色，选择两次+三次变色
def rot_color_perm6():
    def _transform(images):
        size = images.shape[1:]
        out = []
        for x in [images, torch.rot90(images, 2, (2, 3))]:
            out.append(x)
            out.append(torch.stack([x[:, 1, :, :], x[:, 2, :, :], x[:, 0, :, :]], 1))
            out.append(torch.stack([x[:, 2, :, :], x[:, 0, :, :], x[:, 1, :, :]], 1))
        return torch.stack(out, 1).view(-1, *size).contiguous()

    return _transform, 6

#选择4次，变两色，共12张
def rot_color_perm12():
    def _transform(images):
        size = images.shape[1:]
        out = []
        for k in range(4):
            x = torch.rot90(images, k, (2, 3))
            out.append(x)
            out.append(torch.stack([x[:, 1, :, :], x[:, 2, :, :], x[:, 0, :, :]], 1))
            out.append(torch.stack([x[:, 2, :, :], x[:, 0, :, :], x[:, 1, :, :]], 1))
        return torch.stack(out, 1).view(-1, *size).contiguous()

    return _transform, 12

#选择四次变五色，共计24
def rot_color_perm24():
    def _transform(images):
         # 检查输入照片的通道数
        assert images.shape[1] == 3, f"Unexpected number of channels in input: {images.shape[1]}"
    
        size = images.shape[1:]
        out = []
        for k in range(4):
            x = torch.rot90(images, k, (2, 3))
            # print(f"After rotation: {x.size()}")  # 打印旋转后的张量尺寸

            # 检查旋转后的照片通道数
            assert x.shape[1] == 3, f"Unexpected number of channels after rotation: {x.shape[1]}"
            
            out.append(x)

            for perm in [(0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
                stacked = torch.stack([x[:, i, :, :] for i in perm], 1)
                # print(f"After stacking: {stacked.size()}")  # 打印堆叠后的张量尺寸
                # 检查堆叠后的照片通道数
                # assert stacked.shape[1] == 3, f"Unexpected number of channels after stacking: {stacked.shape[1]}"
                out.append(stacked)

        #检查所有张量尺寸是否一致
        for tensor in out:
            assert tensor.shape[1:] == size, f"Tensor shape {tensor.shape[1:]} does not match expected shape {size}"

        return torch.stack(out, 1).view(-1, *size).contiguous()

    return _transform, 24