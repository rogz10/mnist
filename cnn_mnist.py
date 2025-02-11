
import torch
from torchvision import datasets, transforms
import ssl
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



# Transformation des images en tenseurs et normalisation
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

#ssl._create_default_https_context = ssl._create_unverified_context

# Chargement des datasets d'entraînement et de test
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

print("taille d'entrainement:\n",len(train_dataset))
print("taille de test :\n",len(test_dataset))
print("premier entrainement",train_dataset[0])
print("premier test",test_dataset[0])
# Création d'un DataLoader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
print("la longeur du batch:\n",len(train_loader)) # afficher le nombre de batchs
print("la longeur du batch:\n",len(test_loader)) # afficher le nombre de batchs
data_iter = iter(train_loader) #création d'un itérateur

images, labels = next(data_iter) #récupération d'un batch
print("la taille des images est :\n",images.size()) # afficher la taille des images
print("la taille des labels est :\n",labels.size()) # afficher la taille des labels
print("la taille des images est :\n",images.shape) # afficher la taille des images

fig, axes = plt.subplots(1, 10, figsize=(10, 4))
for i in range(10):
    axes[i].imshow(images[i].numpy().squeeze(), cmap='gray')
    axes[i].set_title(f"Label : {labels[i].item()}")
    axes[i].axis('off')
plt.show()


#classe de neurones
class Perceptron(nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        #self.linear=nn.Linear(input_dim,10)
        self.fc1 = nn.Linear(28*28, 128)  # Couche d'entrée (784 → 128 neurones)
        self.fc2 = nn.Linear(128, 64)    # Couche cachée (128 → 64 neurones)
        self.fc3 = nn.Linear(64, 10)     # Couche de sortie (64 → 10 neurones) """
        #self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
       # x = x.view(x.size(0), -1)  # Aplatir l'image (28x28 → 784)
       x = x.view(-1, 28 * 28)  # platir l'image (28x28 → 784)
        #produit lineaire
       #linear_output = self.linear(x)
       #application de la fonction sigmoide
       #activated_output = torch.sigmoid(linear_output)
       #return activated_output

       x = F.relu(self.fc1(x))          # ReLU sur la première couche
       x = F.relu(self.fc2(x))          # ReLU sur la deuxième couche
       x = self.fc3(x)                  # Pas d'activation (logits pour CrossEntropy)
       return x

#input_dim=784
#perceptron=Perceptron(input_dim)
model=Perceptron()
print("les perceptron:\n",model)

#sortie


#output=perceptron(input)
#print("perceptron de sortie:\n",output)

#fonction perte

criterion = nn.CrossEntropyLoss() #binary cross entropy loss
#optimiseur
optimizer = optim.Adam(model.parameters(), lr=0.001)  #stochastic gradient descent


print("criterion:\n",criterion)
print("optimizer:\n",optimizer)

#entrainement

num_epochs=10
for epoch in range(num_epochs):
     # Mode entraînement
    model.train()
    running_loss=0.0
    for images,labels in train_loader:
        #optimizer.zero_grad()
        output=model(images)
        #labels=labels.view(-1,1).float()
        loss=criterion(output,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()

    print(f"epochs [{epoch+1}/{num_epochs}], Perte moyenne : {running_loss / len(train_loader):.4f}")


#evaluation
model.eval()
correct=0
total=0
with torch.no_grad():
    for images,labels in test_loader:
        output=model(images)
        _,predicted=torch.max(output.data,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
print(f"Précision sur le jeu de test : {100 * correct / total:.2f}%")
#print("accuracy: ",100*correct/total)

data_iter = iter(test_loader)
images, labels = next(data_iter)

outputs = model(images)
_, predicted = torch.max(outputs, 1)
print("Predicted: ", predicted)

# Afficher les images et les prédictions

fig, axes = plt.subplots(1, 6, figsize=(10, 4))
for i in range(6):
    axes[i].imshow(images[i].numpy().squeeze(), cmap='gray')
    axes[i].set_title(f"Vrai : {labels[i]}\nPrédit : {predicted[i]}")
    axes[i].axis('off')
plt.show()














