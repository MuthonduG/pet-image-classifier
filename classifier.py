import ast
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import torch

resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

models = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16}

# obtain ImageNet labels
with open('imagenet1000_clsid_to_human.txt') as imagenet_classes_file:
    imagenet_classes_dict = ast.literal_eval(imagenet_classes_file.read())

def read_dog_classes(file_path):
    with open(file_path, 'r') as file:
        print([line.strip().lower() for line in file])

def classify_image(img_path, model_name, dog_classes_file):
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

    # Check if the predicted class corresponds to a dog-related class
    dog_classes = read_dog_classes(dog_classes_file)
    predicted_label = imagenet_classes_dict[pred_idx].lower()

    if any(dog_class in predicted_label for dog_class in dog_classes):
        print("Dog")
    else:
        print("Not a Dog")

# Example usage
img_path = "path/to/your/image.jpg"
model_name = 'resnet'  # You can change this to 'alexnet' or 'vgg'
dog_classes_file = "dogs.txt"
classify_image(img_path, model_name, dog_classes_file)
