import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import  torch.nn.functional as F
import torch.optim as optim


transforms=transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])
device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
#print(device)
#charger les données
train_dataset=datasets.MNIST(root='./data',train=True,download=True,transform=transforms)
test_dataset=datasets.MNIST(root='./data',train=False,download=True,transform=transforms)
print(f"la longeur du train est :\n",len(train_dataset))
print(f"la longeur du test est :\n",len(test_dataset))
#Créer un DataLoader
train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=64,shuffle=False)
image,label=train_dataset[0]
print("la taille de l'image est :\n",image.shape)
#affichage des images
"""fix,axes=plt.subplots(1,10,figsize=(10,4))
for i in range(10):
    axes[i].imshow(train_dataset[i][0].numpy().squeeze(),cmap="gray")
    axes[i].set_title(f"{train_dataset[i][1]}")
    axes[i].axis("off")
plt.show()
"""
#classe de neurones
class CNN_MNIST(nn.Module):
    def __init__(self):
        super(CNN_MNIST,self).__init__()
        self.conv1=nn.Conv2d(1,16,kernel_size=3,padding=1) #1x28x28 → 16x28x28
        self.bn1=nn.BatchNorm2d(16) #normalisation de la première couche de convolution
        self.conv2=nn.Conv2d(16,32,kernel_size=3,padding=1) #16x28x28 → 32x28x28
        self.bn2=nn.BatchNorm2d(32) #normalisation de la deuxième couche de convolution
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2) #32x28x28 → 32x14x14 2 fois donc je passe à 7x7
        self.dropout=nn.Dropout(0.25)
        self.fc1=nn.Linear(32*7*7,128) # 32x7x7 à  128
        self.fc2=nn.Linear(128,10)
    def forward(self,x):
        x=self.pool(F.relu(self.bn1(self.conv1(x)))) #1ère couche de convolution
        x=self.pool(F.relu(self.bn2(self.conv2(x)))) #2eme couche de convolution
        x=x.view(x.size(0),-1) # aplatissement
        x=self.dropout(F.relu(self.fc1(x))) #couche cachée
        x=self.fc2(x) #couche de sortie
        return x
    print("le modèle est prêt")
#initialisation modèle,fonction de perte et optimiseur
model=CNN_MNIST().to(device)
criterion=nn.CrossEntropyLoss()
optimizer=optim.AdamW(model.parameters(),lr=0.001)
#optimizer=optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.1) #reduction du taux d'apprentissage
# entraînement
num_epochs=10 #nombre d'époques
train_loss=[] #perte d'entraînement
for epoch in range(num_epochs):
    model.train() #mode training
    running_loss=0.0 #perte du training
    for images,labels in train_loader:
        images,labels=images.to(device),labels.to(device)
        optimizer.zero_grad() #init des gradients
        outputs=model(images) #propagation avant
        loss=criterion(outputs,labels) # perte
        loss.backward() #propagation arrière
        optimizer.step() #optimisation
        running_loss+=loss.item()#*image.size(0) #ajout de perte

    scheduler.step() #réduction  taux d'apprentissage
    epoch_loss=running_loss/len(train_loader) #perte moyenne
    train_loss.append(running_loss/len(train_loader)) #ajout  perte moyenne

#evaluation du test
    model.eval() #mode évaluation
    correct=0 #correct predictions
    total=0 #nombre total de prédictions
    with torch.no_grad():
        for images,labels in test_loader:
            images,labels=images.to(device),labels.to(device)
            outputs=model(images)
            _,predicted=torch.max(outputs.data,1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
    accuracy=correct/total*100

    print(f"Époque [{epoch+1}/{num_epochs}] | Perte moyenne : {epoch_loss:.4f}| Précision : {accuracy:.2f}%")
# affichage de la perte d'entrainement
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_loss, label='Perte d\'entraînement')
plt.xlabel('Époque')
plt.ylabel('Perte')
plt.legend()
plt.grid()
plt.show()

# eval final
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"\nPrécision finale sur le test : {100 * correct / total:.2f}%")

# affichage des prédictions
temp_loader = DataLoader(test_dataset, batch_size=6, shuffle=True)
images, labels = next(iter(temp_loader))
images, labels = images.to(device), labels.to(device) #charger les données syr le gpu
# prediction model
outputs = model(images)
_, predicted = torch.max(outputs, 1)
images, predicted, labels = images.cpu(), predicted.cpu(), labels.cpu() # revenir sur cpu
# prediction images
plt.figure(figsize=(10, 6))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i].numpy().squeeze(), cmap='gray')
    plt.title(f"Prédit : {predicted[i]}\nVrai : {labels[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()