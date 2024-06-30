import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import AdeversarialDataset
import numpy as np
from torchvision.transforms import transforms
from torchvision.models import resnet50, ResNet50_Weights
from detector.utils.timing import timeit

model = resnet50()
resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

num_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=2)

print("Loading model...")
model.load_state_dict(torch.load("./detector/saved_models/ResNet18_attacker=L1ProjectedGradientDescent_epsilon=0.01_resnet50_2024_04_15_19_34_47_epochs=1000_batch=160_lr=0.001_diff=0_optimizer=Adam_scheduler=lr_scheduler.StepLR/epoch=115_trainacc=0.9993113782000191_testacc=0.99905636317266.pth"))
print("Model loaded!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scripted_model = torch.jit.script(model)
scripted_model.to(device)

scripted_model = torch.ao.quantization.quantize_dynamic(
    scripted_model,
    {torch.nn.Linear}, 
    dtype=torch.qint8)


transformations = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

adversarial = AdeversarialDataset("./data/german/detect_attack/ResNet18_attacker=L2DeepFool_epsilon=0.01/test", transform=transformations)
loader = DataLoader(adversarial, batch_size=1, shuffle=False)

accuracies: list[float] = []
@timeit
def accuracy():
    global scripted_model
    scripted_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            output = scripted_model(inputs)

            _, predicted = torch.max(output, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    # Calculate accuracy
    acc = correct / total
    accuracies.append(acc)


if __name__ == '__main__':
    times: list[float] = []
    print("Starting the accuracy test!")
    for i in range(10):
        print(f"Accuracy for iteration {i}")
        _, t = accuracy()
        times.append(t)
    print(f"Avg. accuracy on 10 runs is: {np.mean(accuracies):.4f}")
