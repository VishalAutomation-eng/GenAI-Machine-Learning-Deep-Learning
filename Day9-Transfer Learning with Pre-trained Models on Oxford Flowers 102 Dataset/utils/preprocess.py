from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

def get_preprocess_fn(model_name):
    if model_name == "resnet50":
        return resnet_preprocess
    if model_name == "vgg16":
        return vgg_preprocess
    if model_name == "mobilenetv2":
        return mobilenet_preprocess
    raise ValueError("Invalid model name")
