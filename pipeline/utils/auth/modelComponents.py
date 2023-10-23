import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm

device = torch.device("cuda" if torch.cuda.is_available() else  "cpu")

edge_list_a = [(0,1), (0,2), (1,3), (2,4), (3, 5), (4, 6), (1, 7), (2, 8), (7,8)]
edge_list_b = [(j,i) for i,j in edge_list_a]
edge_list = edge_list_a + edge_list_b
spacial_edge_list = [(i + 9*inc, j + 9*inc) for (i,j) in edge_list for inc in range(128)]
temp_edge_list_a = [(i, i+9) for i in range(128 * 9 -9)]
temp_edge_list_b = [(j, i) for (i,j) in temp_edge_list_a]
temp_edge_list = temp_edge_list_a + temp_edge_list_b
total_edge_list = spacial_edge_list + temp_edge_list
total_edges= list(zip(*total_edge_list))
edges = torch.tensor(total_edges).long().to(device)

class GCNLayer(torch.nn.Module): #Could use a different GCNLayer such as the attention-based ones
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.batch_norm = BatchNorm(out_channels)   
        #Batch_Norm is part of this library. Also make sure to credit the paper this model is based on
        
    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return F.relu(x)  

class ConvLayer(nn.Module):
    def __init__(self, prior, hidden, ker):
        super().__init__()
        self.layer = nn.Sequential(nn.Conv1d(prior, hidden, ker, padding = (int(ker/2-.5))),
                                   nn.BatchNorm1d(hidden),
                                   nn.ReLU())
    def forward(self, x):
        return self.layer(x)
    

class ResidualConvLayer(nn.Module):
    def __init__(self, prior, hidden, ker):
        super().__init__()
        self.main_path = ConvLayer(prior, hidden, ker)
        self.shortcut = nn.Conv1d(prior, hidden, 1) if prior != hidden else nn.Identity()    

    def forward(self, x):
        return self.main_path(x) + self.shortcut(x)  # Apply batch normalization before ReLU


class BodyPartLayer(nn.Module):
    def __init__(self, num_parts):
        super().__init__()
        self.conv1 = ResidualConvLayer(num_parts*64, 512, 3)
        self.conv2 = ResidualConvLayer(512, 512, 3)
        self.conv3 = ResidualConvLayer(512, 256, 3)
        self.conv4 = ResidualConvLayer(256, 256, 3)
        self.conv5 = ResidualConvLayer(256, 64, 3)
        
        self.pool = nn.MaxPool1d(3,  stride = 2)
        
    def forward(self, x): 
        z = self.conv1(x)
        z = self.conv2(z)
        z = self.pool(z)
        z = self.conv3(z)
        z = self.conv4(z)
        z = self.pool(z)
        z = self.conv5(z)
        v = z.mean(dim = 2)
        return v

class Siamese3D(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.gnn1 = GCNLayer(3, 64)
        self.gnn2 = GCNLayer(64,64)
            
        self.convUpperTorso = BodyPartLayer(3)
        self.convLowerTorso = BodyPartLayer(2)
        self.convLeftArm = BodyPartLayer(3)
        self.convRightArm = BodyPartLayer(3)
#         self.convFullBody = BodyPartLayer(9)
        
        self.encode = nn.Sequential(nn.Linear(64 * 4, 256), nn.ReLU(), nn.Linear(256, 64))
#         self.convEntireBody, basically a residual layer through the hierarchy
        
    def forward(self, x):
        batch = x.shape[0]
        x = x.view(batch,-1,3)
        x = self.gnn1(x,edges)
        x = self.gnn2(x,edges)
        x = x.view(batch, 9, 64, 128) #assuming seq_len = 128
        upperTorso = x[:,[0,1,2],:,:].view(batch, -1, 128)
        lowerTorso = x[:,[7,8],:,:].view(batch, -1, 128)
        leftArm = x[:, [1, 3, 5], :, :].view(batch, -1, 128)
        rightArm = x[:, [2,  4, 6],:,:].view(batch, -1, 128)
        x2 = self.convUpperTorso(upperTorso)
        x3 = self.convLowerTorso(lowerTorso)
        x4 = self.convLeftArm(leftArm)
        x5 = self.convRightArm(rightArm)
#         x6 = self.convFullBody(x.view(batch,-1,128))
        x = torch.concat([x2,x3,x4,x5], dim = 1)
        return self.encode(x)
    
class Siamese2D(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.gnn1 = GCNLayer(2, 64)
        self.gnn2 = GCNLayer(64,64)
        self.convUpperTorso = BodyPartLayer(3)
        self.convLowerTorso = BodyPartLayer(2)
        self.convLeftArm = BodyPartLayer(3)
        self.convRightArm = BodyPartLayer(3)
        self.encode = nn.Sequential(nn.Linear(64 * 4, 256), nn.ReLU(), nn.Linear(256, 64))
        
    def forward(self, x):
        batch = x.shape[0]
        x = x.view(batch,-1,2)
        x = self.gnn1(x,edges)
        x = self.gnn2(x,edges)
        x = x.view(batch, 9, 64, 128) #assuming seq_len = 128
        upperTorso = x[:,[0,1,2],:,:].view(batch, -1, 128)
        lowerTorso = x[:,[7,8],:,:].view(batch, -1, 128)
        leftArm = x[:, [1, 3, 5], :, :].view(batch, -1, 128)
        rightArm = x[:, [2,  4, 6],:,:].view(batch, -1, 128)
        x2 = self.convUpperTorso(upperTorso)
        x3 = self.convLowerTorso(lowerTorso)
        x4 = self.convLeftArm(leftArm)
        x5 = self.convRightArm(rightArm)
        x = torch.concat([x2,x3,x4,x5], dim = 1)
        return self.encode(x)        
        
def init_siamese(deg = 3):
    if deg == 3:
        return Siamese3D()
    else:
        return Siamese2D()
