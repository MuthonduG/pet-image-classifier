import ast
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models

class DogClassifier:
    def __init__(self, model_name, dog_classes_file):
        self.model = self.load_pretrained_model(model_name)
        self.imagenet_classes_dict = self.load_imagenet_classes()
        self.dog_classes = self.read_dog_classes(dog_classes_file)
        self.transform = self.create_transform()

    def load_pretrained_model(self, model_name):
        model = getattr(models, model_name)(pretrained=True)
        model.eval()
        return model

    def load_imagenet_classes(self):
        with open('imagenet1000_clsid_to_human.txt') as imagenet_classes_file:
            return ast.literal_eval(imagenet_classes_file.read())

    def read_dog_classes(self, file_path):
        with open(file_path, 'r') as file:
            return [line.strip().lower() for line in file]

    def create_transform(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def classify_image(self, img_path):
        img_pil = Image.open(img_path)
        img_tensor = self.transform(img_pil)
        img_tensor.unsqueeze_(0)

        with torch.no_grad():
            output = self.model(img_tensor)

        pred_idx = output.data.numpy().argmax()
        predicted_label = self.imagenet_classes_dict[pred_idx].lower()

        if any(dog_class in predicted_label for dog_class in self.dog_classes):
            return "Dog"
        else:
            return "Not a Dog"

class CustomDogClassifier(DogClassifier):
    def __init__(self, model_name, dog_classes_file, additional_dog_classes=None):
        super().__init__(model_name, dog_classes_file)
        
        if additional_dog_classes is not None:
            self.dog_classes.extend(additional_dog_classes)

    def classify_image_from_path(self, img_path):
        result = self.classify_image(img_path)
        print(result)

# Example usage
model_name = 'resnet'  # You can change this to 'alexnet' or 'vgg'
dog_classes_file = "./dogs.txt"
additional_dog_classes = ["your_additional_dog_class"]

custom_dog_classifier = CustomDogClassifier(model_name, dog_classes_file, additional_dog_classes)
img_path = "./assets/dogs/d1.jpg"
custom_dog_classifier.classify_image_from_path(img_path)
