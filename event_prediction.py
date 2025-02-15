
# Define models with the use of minibatch
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, utils
import torch.nn as nn
from scipy.special import softmax
import torchvision
from torch.autograd import Variable
from sklearn.decomposition import PCA
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(linewidth=1000)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init  as init
import pandas as pd
import random
import pprint
from torch.nn.utils.rnn import pad_sequence
import pathlib
import os
import bottleneck as bn
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support
device=torch.device('cuda:0')
plt.style.use('ggplot')


# Define an RNN model (The generator)
class LSTMGenerator(nn.Module):
    def __init__(self, seq_len, input_size, batch, hidden_size , num_layers, num_directions):
        super().__init__()
        self.input_size = input_size
        self.h = torch.randn(num_layers * num_directions ,batch, hidden_size)
        self.c = torch.randn(num_layers * num_directions ,batch, hidden_size)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.25, batch_first =True, bidirectional = False)


        latent_vector_size =50 * batch
        self.linear1 = nn.Linear(batch * seq_len *hidden_size, latent_vector_size)
        # self.linear2 = nn.Linear(latent_vector_size,batch*seq_len*hidden_size)
        self.linearHC = nn.Linear(num_layers *hidden_size *batch, latent_vector_size)
        # self.linearHCO = nn.Linear(3*latent_vector_size,batch*seq_len*hidden_size )
        self.linearHCO = nn.Linear( 3 *latent_vector_size ,batch *seq_len *input_size )




        # Define sigmoid activation and softmax output
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x):
        # x = x.view((1,x.size()[0], x.size()[1]))
        # Pass the input tensor through each of our operations
        # print("inputsize:", x.size())
        output, (h ,c) = self.lstm(x, (self.h, self.c))
        # print("inputsize:", x.size(),"output size:", output.size())
        # print("h size:", h.size(),"c size:", c.size())
        self.h = h.detach()
        self.c = c.detach()

        # Executing Fully connected network
        # print("The size of output:", output.size(), h.size(), c.size())
        u = output.reshape((output.size()[0 ] *output.size()[1 ] *output.size()[2]))
        u = self.relu(self.linear1(u))
        # print("The size of lninera1:", u.size())
        # u = self.linear2(u)

        # Flating h and feeding it into a linear layer
        uH = F.leaky_relu(self.linearHC(h.reshape((h.size()[0 ] *h.size()[1 ] *h.size()[2]))))
        uC = F.leaky_relu(self.linearHC(c.reshape((c.size()[0 ] *c.size()[1 ] *c.size()[2]))))
        uHCO = torch.cat((uH ,uC ,u))
        uHCO = self.linearHCO(uHCO)
        u= uHCO

        # output = u.view((output.size()[0],output.size()[1],output.size()[2]))
        output = u.view((output.size()[0],output.size()[1],self.input_size))
        # print("output size finally:", output.size())


        return output


####################################################################################################
# Define an RNN model (The discriminator)
class LSTMDiscriminator(nn.Module):
    def __init__(self, seq_len, input_size, batch, hidden_size, num_layers, num_directions):
        self.batch = batch
        super().__init__()
        self.h = torch.randn(num_layers * num_directions, batch, hidden_size)
        self.c = torch.randn(num_layers * num_directions, batch, hidden_size)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.25, batch_first=True, bidirectional=False)
        # h0 = torch.randn(,1, 513)
        # c0 = torch.randn(1,1, 513)

        latent_vector_size = 50 * batch
        self.linear1 = nn.Linear(batch * seq_len * hidden_size, latent_vector_size)
        self.linearHC = nn.Linear(num_layers * hidden_size * batch, latent_vector_size)
        # self.linearHCO = nn.Linear(3*latent_vector_size,batch*seq_len*input_size )
        self.linearHCO = nn.Linear(3 * latent_vector_size, batch * seq_len * input_size)
        self.linear2 = nn.Linear(batch * seq_len * input_size, 100)
        self.linear3 = nn.Linear(100, 50)
        self.linear4 = nn.Linear(50, batch)

        # h0.data *=0.001
        # c0.data *=0.001

        # Define sigmoid activation and softmax output
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x):
        # x = x.view((1,x.size()[0], x.size()[1]))
        # Pass the input tensor through each of our operations
        output, (h, c) = self.lstm(x, (self.h, self.c))
        # print("inputsize:", x.size(),"output size:", output.size())
        self.h = h.detach()
        self.c = c.detach()

        # Executing Fully connected network
        # print("The size of output:", output.size(), h.size(), c.size())
        u = output.reshape((output.size()[0] * output.size()[1] * output.size()[2]))
        u = self.relu(self.linear1(u))
        # u = self.linear2(u)

        # Flating h and feeding it into a linear layer
        uH = F.leaky_relu(self.linearHC(h.reshape((h.size()[0] * h.size()[1] * h.size()[2]))))
        uC = F.leaky_relu(self.linearHC(c.reshape((c.size()[0] * c.size()[1] * c.size()[2]))))
        uHCO = torch.cat((uH, uC, u))
        uHCO = self.linearHCO(uHCO)
        u = F.relu(self.linear2(uHCO))
        u = F.relu(self.linear3(u))
        u = self.linear4(u)

        # output = u.view((output.size()[0],output.size()[1],output.size()[2]))
        # output = u.view((output.size()[0],output.size()[1],input_size))
        output = u

        # Reshaping into (batch,-1)
        # tensor([[-0.1050],
        # [ 0.0327],
        # [-0.0260],
        # [-0.1059],
        # [-0.1055]], grad_fn=<ViewBackward>)
        output = output.reshape((self.batch, -1))

        return output

####################################################################################################
def one_hot_encoding(batch, no_events, y_truth):
    '''
    batch : the batch size
    no_events : the number of events
    y_truth : the ground truth labels

    example:
      tensor([[8.],
        [6.],
        [0.],
        [0.],
        [8.]])

    tensor([[0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]])'''

    z = torch.zeros((batch, no_events))
    for i in range(z.size()[0]):
        z[i, y_truth[i].long()] = 1

    # print(z)
    return z.view(batch, 1, -1)

####################################################################################################
####################################################################################################


def model_eval_test(modelG, mode, obj):
    '''
    This module is for validation and testing the Generator
    @param modelG: Generator neural network
    @param mode: 'validation', 'test', 'test-validation'
    @param obj: A data object created from "Input" class that contains the required information
    @return: The average accuracy and F1 score of the Generator over all mini-batches
    '''
    # Set the evaluation mode
    rnnG = modelG
    rnnG.eval()

    validation_loader = obj.validation_loader
    test_loader = obj.test_loader
    batch = obj.batch
    events = list(np.arange(0, len(obj.unique_event)))
    prefix_len = obj.prefix_len

    # Determine which data loader to use
    if mode == 'validation':
        data_loader = validation_loader
    elif mode == "test":
        data_loader = test_loader
    elif mode == 'test-validation':
        data_loader = test_loader + validation_loader

    # Initialize variables to store metrics
    accuracy_record = []
    y_truth_list = []
    y_pred_last_event_list = []

    for mini_batch in iter(data_loader):
        x = mini_batch[0]
        y_truth = mini_batch[1]
        if x.size()[0] < batch:
            continue

        # Execute LSTM
        y_pred = rnnG(x[:, :, events])
        y_pred_last = y_pred[0: batch, prefix_len - 1, :]
        y_pred_last_event = torch.argmax(F.softmax(y_pred_last.view((batch, 1, -1)), dim=2), dim=2)

        # Collect ground truth and predictions
        y_truth_list += list(y_truth.flatten().data.cpu().numpy().astype(int))
        y_pred_last_event_list += list(y_pred_last_event.flatten().data.cpu().numpy().astype(int))

        # Calculate accuracy for this mini-batch
        correct_predictions = (y_pred_last_event.flatten() == y_truth.long().flatten()).sum().item()
        batch_accuracy = correct_predictions / y_truth.numel()
        accuracy_record.append(batch_accuracy)

    rnnG.train()

    # Calculate final metrics
    avg_accuracy = np.mean(accuracy_record)  # Average of batch-level accuracies
    weighted_precision, weighted_recall, weighted_f1score, _ = precision_recall_fscore_support(
        y_truth_list, y_pred_last_event_list, average='weighted', labels=events
    )

    # Write results to a file (if mode is 'test')
    if mode == 'test':
        with open(obj.path + '/results.txt', "a") as fout:
            fout.write(
                f"Average Accuracy: {avg_accuracy:.4f}\n"
                f"Average F1 Score: {weighted_f1score:.4f}\n"
            )

    # Print average results
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average F1 Score: {weighted_f1score:.4f}")

    return avg_accuracy, weighted_f1score

####################################################################################################

# Gradient Penalty Function
def gradient_penalty(critic, real_data, fake_data, device):
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)  # Interpolation factor
    alpha = alpha.expand_as(real_data)
    interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)

    critic_interpolates = critic(interpolates)
    gradients = autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(critic_interpolates).to(device),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    return penalty


# Loss Functions
def critic_loss(critic, real_data, fake_data, device, lambda_gp=10):
    real_score = critic(real_data).mean()
    fake_score = critic(fake_data).mean()
    gp = gradient_penalty(critic, real_data, fake_data, device)
    return fake_score - real_score + lambda_gp * gp


def generator_loss(critic, fake_data):
    return -critic(fake_data).mean()


# Training Loop
'''
def train(rnnG, rnnD, optimizerD, optimizerG, obj, epoch):
    """
    Train the GAN using the WGAN-GP approach.

    Args:
        rnnG: Generator network.
        rnnD: Discriminator (critic) network.
        optimizerD: Optimizer for the discriminator (critic).
        optimizerG: Optimizer for the generator.
        obj: Object containing data-related attributes (e.g., path, batch size).
        epoch: Number of training epochs.
    """
    device = next(rnnG.parameters()).device  # Ensure proper device allocation
    train_loader = obj.train_loader
    accuracy_best = -float('inf')  # Track the best F1 score or accuracy
    path = obj.path
    critic_steps = 5  # Number of critic updates per generator update
    lambda_gp = 10    # Gradient penalty weight

    for epoch_idx in range(epoch):
        for i, real_data in enumerate(train_loader):
            real_data = real_data[0].to(device)  # Assuming the first element is the input data
            batch_size = real_data.size(0)

            # Generate fake data
            noise = torch.randn(batch_size, obj.prefix_len, len(obj.unique_event), device=device)
            fake_data = rnnG(noise)

            # --------------------
            # Update Critic
            # --------------------
            for _ in range(critic_steps):
                optimizerD.zero_grad()
                d_loss = critic_loss(rnnD, real_data, fake_data, device, lambda_gp)
                d_loss.backward()
                optimizerD.step()

            # --------------------
            # Update Generator
            # --------------------
            if i % critic_steps == 0:
                optimizerG.zero_grad()
                g_loss = generator_loss(rnnD, fake_data)
                g_loss.backward()
                optimizerG.step()

        # --------------------
        # Validation
        # --------------------
        if epoch_idx % 5 == 0:
            rnnG.eval()
            accuracy, f1_score = model_eval_test(rnnG, 'validation', obj)
            rnnG.train()

            if f1_score > accuracy_best:  # Save best model based on F1 score
                print(f"Epoch {epoch_idx}: Validation Accuracy: {accuracy}, F1 Score: {f1_score}")
                accuracy_best = f1_score

                # Save models
                if not os.path.isdir(path):
                    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
                torch.save(rnnG.state_dict(), f"{path}/rnnG_best.pth")
                torch.save(rnnD.state_dict(), f"{path}/rnnD_best.pth")

        print(f"Epoch {epoch_idx + 1}/{epoch} - Critic Loss: {d_loss.item():.4f}, Generator Loss: {g_loss.item():.4f}")

'''
def train(rnnG, rnnD, optimizerD, optimizerG, obj, epoch):
    '''
    @param rnnG: Generator neural network
    @param rnnD: Discriminator neural network
    @param optimizerD:  Optimizer of the discriminator
    @param optimizerG:  Optimizer of the generator
    @param obj:       A data object created from "Input" class that contains the training,test, and validation datasets and other required information
    @param epoch:    The number of epochs
    @return: Generator and Discriminator
    '''

    # Training Generator
    #epoch = 30
    #events = list(np.arange(0, len(obj.unique_event) + 1))
    events = list(np.arange(0, len(obj.unique_event) ))
    gen_loss_pred = []
    disc_loss_tot = []
    gen_loss_tot = []
    accuracy_best = 0
    f1_score_best = 0

    for i in tqdm(range(epoch)):
        for mini_batch in iter(obj.train_loader):

            x = mini_batch[0];
            y_truth = mini_batch[1]
            # When we create mini batches, the length of the last one probably is less than the batch size, and it makes problem for the LSTM, therefore we skip it.
            if (x.size()[0] < obj.batch):
                continue
            # print(x[:,:,events], x.size(),'\n', y_truth)
            # -----------------------------------------------------------------------------------------------------

            # Training discriminator
            optimizerD.zero_grad()

            # Executing LSTM
            y_pred = rnnG(x[:, :, events])
            # print("y_pred:\n", y_pred, y_pred.size())

            # Just taking the last predicted element from each the batch
            y_pred_last = y_pred[0:obj.batch, obj.prefix_len - 1, :]
            y_pred_last = y_pred_last.view((obj.batch, 1, -1))
            # print("y_pred:", y_pred_last)

            # Converting the labels into one hot encoding
            y_truth_one_hot = one_hot_encoding(obj.batch, len(events), y_truth)

            # Creating synthetic and realistic datasets
            ##data_synthetic = torch.cat((x[:,:,events],F.softmax(y_pred_last,dim=2)), dim=1)
            y_pred_last_event = torch.argmax(F.softmax(y_pred_last, dim=2), dim=2)
            y_pred_one_hot = one_hot_encoding(obj.batch, len(events), y_pred_last_event)
            data_synthetic = torch.cat((x[:, :, events], y_pred_one_hot), dim=1)

            # Realistinc dataset
            data_realistic = torch.cat((x[:, :, events], y_truth_one_hot), dim=1)



            # Training Discriminator on realistic dataset
            discriminator_realistic_pred = rnnD(data_realistic)
            disc_loss_realistic = F.binary_cross_entropy(F.sigmoid(discriminator_realistic_pred),
                                                         torch.ones((obj.batch, 1)), reduction='sum')
            disc_loss_realistic.backward(retain_graph=True)

            # Training Discriminator on synthetic dataset
            discriminator_synthetic_pred = rnnD(data_synthetic)
            # print("disc pred:", discriminator_synthetic_pred)
            disc_loss_synthetic = F.binary_cross_entropy(F.sigmoid(discriminator_synthetic_pred),
                                                         torch.zeros((obj.batch, 1)), reduction='sum')
            disc_loss_synthetic.backward(retain_graph=True)

            disc_loss_tot.append(disc_loss_realistic.detach() + disc_loss_synthetic.detach())

            optimizerD.step()

            if len(disc_loss_tot) % 1000 == 0:
                print("iter =------------------------------ i :", i, len(disc_loss_tot), " the Disc error is:",
                      ", the avg is:", np.mean(disc_loss_tot))

            #-------------------------------------------------------------------------------------------------------------------------

            # Training teh Generator
            # Training the prediction for the generator

            optimizerG.zero_grad()

            # Computing the loss of prediction
            lstm_loss_pred = F.binary_cross_entropy(F.sigmoid(y_pred_last), y_truth_one_hot, reduction='sum')
            gen_loss_pred.append(lstm_loss_pred.detach())
            lstm_loss_pred.backward(retain_graph=True)

            # Fooling the discriminator by presenting the synthetic dataset and considering the labels as the real ones
            discriminator_synthetic_pred = rnnD(data_synthetic)
            # print("disc pred:", discriminator_synthetic_pred)
            gen_fool_dic_loss = F.binary_cross_entropy(F.sigmoid(discriminator_synthetic_pred), torch.ones((obj.batch, 1)),
                                                       reduction='sum')
            gen_fool_dic_loss.backward(retain_graph=True)

            gen_loss_tot.append(lstm_loss_pred.detach() + gen_fool_dic_loss.detach())

            optimizerG.step()

            if len(gen_loss_tot) % 1000 == 0:
                print("iter =------------------------------ i :", i, len(gen_loss_tot), " the Gen error is:",
                      ", the avg is:", np.mean(gen_loss_tot))

        # Applying validation after several epoches
        # Early stopping (checking for every 5 iterations)
        #path = os.getcwd() + "/" + obj.dataset_name + '/event_prediction' + '/prefix_' + str(obj.prefix_len)
        path = obj.path
        #obj.path=path
        if i % 5 == 0:
            rnnG.eval()
            accuracy, f1_score = model_eval_test(rnnG, 'validation', obj)  # Get both metrics
            rnnG.train()

            print(f"The validation set accuracy is: {accuracy}")
            print(f"The validation set F1-Score is: {f1_score}")

            # Save model only if accuracy improves
            # if f1_score > f1_score_best:
            #    f1_score_best = f1_score
            if accuracy > accuracy_best:
                accuracy_best = accuracy
                print("Saving the model with the best validation accuracy...")

                # Writing down the model
                if os.path.isdir(path):
                    torch.save(rnnG, path + "/rnnG(validation).m")
                    torch.save(rnnD, path + "/rnnD(validation).m")
                else:
                    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
                    torch.save(rnnG, path + "/rnnG(validation).m")
                    torch.save(rnnD, path + "/rnnD(validation).m")

    #Saving the models after training
    torch.save(rnnG, path + "/rnnG.m")
    torch.save(rnnD, path + "/rnnD.m")

    #plot_loss(gen_loss_pred, "Prediction loss", obj)
    plot_loss(gen_loss_tot, "Generator loss total", obj)
    plot_loss(disc_loss_tot, "Discriminator loss total", obj)


#########################################################################################################

def plot_loss(data_list, title, obj):
    '''
    #Plotting the input data
    @param data_list: A list of error values or accuracy values
    @param obj:
    @param title: A description of the datalist
    @return:
    '''
    if(title == "Generator loss total" ):
        if(hasattr(obj, 'plot')):
            obj.plot+=1
        else:
            obj.plot=1
    


    #plt.figure()
    plt.plot(bn.move_mean(data_list, window=100, min_count=1), label = title)
    plt.title(title+ ' prefix =' + str(obj.prefix_len) + ',' + "batch = " + str(obj.batch))
    plt.legend()

    tt =str(datetime.now()).split('.')[0].split(':')
    strfile = obj.path+'/'+title+ ', prefix =' + str(obj.prefix_len) + ',' + "batch = " +str(obj.batch) + str(obj.plot)
    plt.savefig(strfile)

    if(title == "Discriminator loss total"):
        plt.close()





