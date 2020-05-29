import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable

class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))
        else:
            self.skip = None    
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.InstanceNorm2d(out_channels))

    def forward(self, x):
        identity = x
        out = self.block(x)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        out = F.relu(out)

        return out

class ResBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResBlockDown, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.resblock = ResBlock(out_channels, out_channels)
        
    def forward(self, x):
        out = self.conv(x)
        return self.resblock(out)
    
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc
        downconv = ResBlockDown(input_nc, inner_nc, 4, 2, 1)
        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nn.ReLU(True)

        if outermost:
            upsample = nn.Upsample(scale_factor=2)
            upconv = ResBlockDown(inner_nc*2, outer_nc, 3, 1, 1)
            down = [downconv]
            up = [uprelu, upsample, upconv]
            model = down + [submodule] + up
        elif innermost:
            upsample = nn.Upsample(scale_factor=2)
            upconv = ResBlockDown(inner_nc, outer_nc, 3, 1, 1)
            down = [downrelu, downconv]
            up = [uprelu, upsample, upconv]
            model = down + up
        else:
            upsample = nn.Upsample(scale_factor=2)
            upconv = ResBlockDown(inner_nc*2, outer_nc, 3, 1, 1)
            down = [downrelu, downconv]
            up = [uprelu, upsample, upconv]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

imagetransform = transforms.Compose([
                transforms.Resize((256, 192)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
modelPath = "scrapseg_model.pth"

def segment(img, outputfile, threshold=0.5):
    model = UnetGenerator(3, 1, 6, ngf=64)
    model.load_state_dict(
        torch.load(modelPath)
    )
    model = model
#     inputOfmodel = torch.zeros(1,3,256,192)
#     print(inputOfmodel.shape)
#     print(inputOfmodel)
#     torch.onnx.export(model, inputOfmodel, 'uplaraNew_model2.onnx', verbose=True)
#     .cuda()
# Image.open(inputfile)
    image = imagetransform(img)
    image = Variable(image).unsqueeze(0)
#     .cuda()
    output = model(image)
    output = torch.sigmoid(output).cpu().detach().numpy()[0, 0, :, :]
    output = (output > threshold).astype(float)
    img = Image.fromarray((output*255).astype(np.uint8))
    img.save(outputfile)
    print("reaches here")
    return img

from flask import Flask, request, send_file, make_response
from flask_restful import Resource, Api
from flask_cors import CORS
import requests
from io import BytesIO

app = Flask(__name__)
api = Api(app)
# CORS(app, origins="*", allow_headers=[
#     "Content-Type", "Authorization", "Access-Control-Allow-Credentials"])
UPLOAD_FOLDER = ''
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def build_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response
def build_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

class SegmentationModel(Resource):
    def post(self):
        if request.method == 'OPTIONS':
            response = build_preflight_response()
            return response
        url="https://homepages.cae.wisc.edu/~ece533/images/airplane.png"
        response = requests.get(url)
        img_input = Image.open(BytesIO(response.content))
        print(img_input)
        img = segment('preview.jpg','output.jpg')
        return build_actual_response(send_file('output.jpg', attachment_filename='out.jpg'))
    def get(self):
        if request.method == 'OPTIONS':
            response = build_preflight_response()
            return response
        image_url = request.args.get('image_url')
        print(image_url)
        url="https://homepages.cae.wisc.edu/~ece533/images/airplane.png"
        responseOfImage = requests.get(image_url)
        img_input = Image.open(BytesIO(responseOfImage.content))
        print(img_input)
        img = segment(img_input,'output.jpg')
        return build_actual_response(send_file('output.jpg', attachment_filename='out.jpg'))
    def options(self):
        if request.method == 'OPTIONS':
            response = build_preflight_response()
            return response

api.add_resource(SegmentationModel, '/output', methods=['POST','GET'])

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)