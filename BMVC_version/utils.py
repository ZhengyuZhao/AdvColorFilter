import torch
import torch.nn as nn
import csv


#image quantization
def quantization(x):
   x_quan=torch.round(x*255)/255 
   return x_quan

#picecwise-linear color filter
def CF(img, param,pieces):
        
    param=param[:,:,None,None]
    color_curve_sum = torch.sum(param, 4) + 1e-30
    total_image = img * 0
    for i in range(pieces):
      total_image += torch.clamp(img - 1.0 * i /pieces, 0, 1.0 / pieces) * param[:, :, :, :, i]
    total_image *= pieces/ color_curve_sum 
    return total_image

#parsing the data annotation
def load_ground_truth(csv_filename):
    image_id_list = []
    label_ori_list = []
    label_tar_list = []

    with open(csv_filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            image_id_list.append( row['ImageId'] )
            label_ori_list.append( int(row['TrueLabel']) )
            label_tar_list.append( int(row['TargetClass']) )

    return image_id_list,label_ori_list,label_tar_list


# simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
    def forward(self, x):
        return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]

# values are standard normalization for ImageNet images, 
# from https://github.com/pytorch/examples/blob/master/imagenet/main.py
norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

 
