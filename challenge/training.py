import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.cnn import CNN
from data_managment.dataset import train_loader, valid_loader, test_loader
import matplotlib.pyplot as plt

class MusicGenreClassifier:
    def __init__(self, model, train_loader, valid_loader, test_loader, device=None, lr=1e-3, n_epochs=300):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.n_epochs = n_epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Move model to the specified device
        self.model.to(self.device)

        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
    
    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with tqdm(total=len(self.train_loader), desc="Training Epoch", leave=False) as pbar:
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                pbar.update(1)
                
        accuracy = 100 * correct / total
        return running_loss / len(self.train_loader), accuracy
    
    def validate(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.valid_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        return val_loss / len(self.valid_loader), accuracy
    
    def test(self):
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        return test_loss / len(self.test_loader), accuracy
    
    def train(self):
        with tqdm(total=self.n_epochs, desc="Training model") as epoch_pbar:
            for epoch in range(self.n_epochs):
                train_loss, train_accuracy = self.train_one_epoch()
                val_loss, val_accuracy = self.validate()

                self.train_losses.append(train_loss)
                self.train_accuracies.append(train_accuracy)
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_accuracy)

                print(f'Epoch [{epoch+1}/{self.n_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
                epoch_pbar.update(1)

        test_loss, test_accuracy = self.test()
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
        
        self.plot_metrics()
    
    def plot_metrics(self):
        epochs = range(1, self.n_epochs + 1)

        plt.figure(figsize=(14, 5))
        plt.title('Train and Validation Accuracy')
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, label='Train Loss')
        plt.plot(epochs, self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Train and Validation Loss')
        plt.legend()

        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accuracies, label='Train Accuracy')
        plt.plot(epochs, self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Validation Accuracy')
        plt.legend()

        # save
        plt.tight_layout()
        plt.savefig('challenge\\results\\training_metrics.png')
        plt.show()

if __name__ == '__main__':
    model = CNN(input_shape=(1, 128, 130))

    classifier = MusicGenreClassifier(model=model,
                                    train_loader=train_loader,
                                    valid_loader=valid_loader,
                                    test_loader=test_loader,
                                    n_epochs=300)

    if not os.path.exists('challenge\\results'):
        os.makedirs('challenge\\results') 
        
    # Train the model
    classifier.train()
    # Save trained model
    torch.save(model.state_dict(), 'challenge\\results\\music_genre_classifier.pth')
    
