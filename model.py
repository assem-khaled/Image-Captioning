import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        
        # set class variables
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # embedding layer
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # LSTM unit
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                            dropout = 0.5, batch_first=True) # batch_first = True => input , output need to have batch size as 1st dimension
        
        # fully-connected layer
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # initialize the weights
        self.init_weights()
        
        
    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.xavier_uniform_(self.embed.weight) 
        
        
    def forward(self, features, captions):
        # features : ouput of CNNEncoder having shape (batch_size, embed_size)
        # captions : a PyTorch tensor corresponding to the last batch of captions having shape (batch_size, caption_length)
        
        captions = captions[:, :-1] # Discard the <end> word
        
        captions = self.embed(captions) # output shape : (batch_size, caption length , embed_size)
        
        # Concatenate the features and caption inputs
        features = features.unsqueeze(1)
        inputs = torch.cat((features, captions), 1) # output shape : (batch_size, caption length, embed_size)
        
        outputs, _ = self.lstm(inputs) # output shape : (batch_size, caption length, hidden_size)
        
        outputs = self.fc(outputs) # output shape : (batch_size, caption length, vocab_size)
        
        return outputs
    

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        outputs_list = [] 
        
        for _ in range(max_len):
          
            output, states = self.lstm(inputs, states) # No need to initialize the hidden states with random values
            
            output = self.fc(output) 

            _, predicted_index = torch.max(output,-1) # Extracting the word index
            
            outputs_list.append(predicted_index.item()) # Saving the word index to a list
            
            if (predicted_index == 1): # "1" => <end>
                break
               
            inputs = self.embed(predicted_index)     
                        
            
        return outputs_list