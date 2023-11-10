import ast
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
from torch import __version__

resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

models = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16}

# obtain ImageNet labels
with open('imagenet1000_clsid_to_human.txt') as imagenet_classes_file:
    imagenet_classes_dict = ast.literal_eval(imagenet_classes_file.read())

def classify_image(img_path, model_name):
    # Load the image
    img_pil = Image.open(img_path)

    # Define transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Preprocess the image
    img_tensor = preprocess(img_pil)

    # Resize the tensor (add dimension for batch)
    img_tensor.unsqueeze_(0)

    # Apply model to input
    model = models[model_name]

    # Put model in evaluation mode
    model = model.eval()

    # Apply data to model
    with torch.no_grad():  # Use torch.no_grad() to disable gradient computation during inference
        output = model(img_tensor)

    # Return index corresponding to predicted class
    pred_idx = output.data.numpy().argmax()

    # Check if the predicted class corresponds to a dog
    if 'dog' in imagenet_classes_dict[pred_idx].lower():
        print("Dog")
    else:
        print("Not a Dog")

# Example usage
img_path = "path/to/your/image.jpg"
model_name = 'resnet'  # You can change this to 'alexnet' or 'vgg'
classify_image(img_path, model_name)
