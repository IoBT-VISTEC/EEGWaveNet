import torch
from torch import nn
from torch.optim import Adam
from sklearn.utils.class_weight import compute_class_weight as classweight
from sklearn.metrics import accuracy_score
import numpy as np
from torch.nn import CrossEntropyLoss
import time

class trainer:
    def __init__(self, Model, Train_set, Val_set, n_classes):

        self.Model = Model
        self.compiled = False

        self.X_train, self.y_train = Train_set
        self.X_val, self.y_val = Val_set

        self.tracker = {'train_tracker':[],'val_tracker':[]}

        weights = classweight(class_weight="balanced",classes=np.arange(n_classes),y=self.y_train.numpy())
        if torch.cuda.is_available():
            class_weights = torch.FloatTensor(weights).cuda()
        else:
            class_weights = torch.FloatTensor(weights)
        self.loss_func = CrossEntropyLoss(weight=class_weights)

    def compile(self,learning_rate):
        self.optimizer = Adam(self.Model.parameters(), lr=learning_rate)
        self.compiled = True

    def train(self, epochs, batch_size=32, patience=10, directory='model.pt'):

        wait = 0

        best_model = self.Model

        if not self.compiled:
            raise Exception("You need to compile an optimizer first before training.")

        train_loss_tracker = []
        val_loss_tracker = []

        trainset = [[self.X_train[i],self.y_train[i]] for i in range(self.X_train.size()[0])]
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

        valset = [[self.X_val[i],self.y_val[i]] for i in range(self.X_val.size()[0])]
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True)
    
        if torch.cuda.is_available():
            self.Model.cuda()

        for e in range(epochs):

            T0 = time.time()

            batch_train_loss = []

            for data, target in trainloader:
                
                if torch.cuda.is_available():

                    self.Model.train()
                    pred = self.Model(data.float().cuda())
                    self.optimizer.zero_grad()
                    train_loss = self.loss_func(pred, target.cuda())
                    train_loss.backward()       
                    self.optimizer.step()
                    
                else:
                    
                    self.Model.train()
                    pred = self.Model(data.float())
                    self.optimizer.zero_grad()
                    train_loss = self.loss_func(pred, target)
                    train_loss.backward()       
                    self.optimizer.step()

                batch_train_loss.append(train_loss)

            final_train_loss = torch.mean(torch.tensor(batch_train_loss)) 

            Training_time = time.time()-T0

            batch_val_loss = []
            
            with torch.no_grad():

                for data, target in valloader:
                    
                    if torch.cuda.is_available():

                        pred = self.Model(data.float().cuda())
                        val_loss = self.loss_func(pred, target.cuda())
                        batch_val_loss.append(val_loss)
                    
                    else:
                        
                        pred = self.Model(data.float())
                        val_loss = self.loss_func(pred, target)
                        batch_val_loss.append(val_loss)

            final_val_loss = torch.mean(torch.tensor(batch_val_loss))    

            print("Epoch Number \t: ",e)
            print("Train Loss \t:","{:.5f}".format(final_train_loss))
            print("Val Loss \t:","{:.5f}".format(final_val_loss))
            print("Training Time \t:","{:.5f}".format(Training_time))
            print("===================================================================================\n")

            if e>patience:
                if val_loss.item()<=np.min(val_loss_tracker):
                    best_model = self.Model
                    torch.save(self.Model.state_dict(), directory)
                    wait = 0
                else:
                    wait += 1
            else:
                torch.save(self.Model.state_dict(), directory)
                
            train_loss_tracker.append(final_train_loss)
            val_loss_tracker.append(final_val_loss)

            if wait >= patience:
                break

        self.tracker['train_tracker'] = train_loss_tracker
        self.tracker['val_tracker'] = val_loss_tracker
        self.Model = best_model

        return self.tracker

    def predict(X_test):

        output = []
        testloader = torch.utils.data.DataLoader(X_test, batch_size=32, shuffle=True)

        with torch.no_grad():

          for data in testloader:
              if torch.cuda.is_available():
                  pred = self.Model.cuda()(data.float().cuda())
                  pred = list(np.argmax(list(pred.cpu().numpy()), axis=1))
                  output += pred
              else:
                  pred = self.Model(data.float())
                  pred = list(np.argmax(list(pred.cpu().numpy()), axis=1))
                  output += pred

        return output
