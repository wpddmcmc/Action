import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tensorboardX import SummaryWriter

class Action_Net(nn.Module):
    def __init__(self):
        super(Action_Net, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 25, 3)
            nn.Conv2d(
                in_channels=1,      # input height
                out_channels=16,    # n_filters
                kernel_size=(2,6),  # filter size
                stride=1,           # filter movement/step
                padding=1,          # padding=(kernel_size-1)/2 height and width don't change (3+2,25+2)
            ),  # output shape (16,5-2+1,27-6+1) = (16,4,22)
            nn.ReLU(),              # activation
            nn.MaxPool2d(kernel_size=(2,2)),  # sample in 2x1 space, output shape (16, 2,11)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 11, 2)
            nn.Conv2d(
                in_channels=16,  # input height
                out_channels=32,  # n_filters
                kernel_size= (1,4),  # filter size
                stride=1,  # filter movement/step
                padding=1,  # padding=(kernel_size-1)/2 height and width don't change (2+2,11+2)
            ), # output shape (32,  13-4+1, 4-1+1) = (32,4,10)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 1, 5)
        )
        self.fcon = nn.Linear( 32 * 5 * 2 , 120)
        self.fcon2 = nn.Linear(120,3) # fully connected layer, output 2 classes
        #self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fcon(x)
        output = self.fcon2 (x)
        #output = self.softmax(output)
        return output

def readdatas(file):
    print("Read file %s as input"%file)
    fr = open(file)
    lines = fr.readlines()
    data_x = []
    for n in range(len(lines)):
        data = []
        if lines[n][0] == 'B':
            data_joint = []
            for i in range(1,26):
                datas = lines[n+i].strip(' \n').split(' ')
                for i in range(3):
                    datas[i] = float(datas[i])
                data_joint.append(datas)
            add_domain = []
            add_domain.append(data_joint)
            data_x.append(add_domain)

    data_x = np.array(data_x)

    print("Data size: ",data_x.shape)
    return data_x

def generate_label(length,type):
    label = np.empty([length],dtype=int)
    for n in range(label.shape[0]):
        label[n] = type
    return  label

def generate_test(data,length):
    import random
    data_test = np.empty([length,1,25,3],dtype=np.float64)
    for i in range(length):
        rand = random.randint(0,data.shape[0]-1)
        data_test[i] = data[rand]
        data = np.delete(data,rand,0)
    return data_test,data

data_hand = readdatas("handsup.dat")
data_run = readdatas("run.dat")
data_sit = readdatas("sit.dat")

hand_test,hand_train = generate_test(data_hand,100)
run_test,run_train = generate_test(data_run,400)
sit_test,sit_train = generate_test(data_sit,100)

print("Training Input Data Size: ",hand_train.shape,run_train.shape,sit_train.shape)
print("Test Input Data Size: ",hand_test.shape,run_test.shape,sit_test.shape)



label_hand = generate_label(hand_train.shape[0],1)
label_run = generate_label(run_train.shape[0],0)
label_sit = generate_label(sit_train.shape[0],2)

label_train = np.append(label_hand,label_run)
label_train = torch.from_numpy(np.append(label_train,label_sit))

print("\nTotal Input Label Size: ",label_train.shape)

label_hand = generate_label(hand_test.shape[0],1)
label_run = generate_label(run_test.shape[0],0)
label_sit = generate_label(sit_test.shape[0],2)

label_test = np.append(label_hand,label_run)
label_test= torch.from_numpy(np.append(label_test,label_sit))
print("Total Testing Label Size: ",label_test.shape)

data_train = np.vstack((hand_train,run_train))
data_train = torch.from_numpy(np.vstack((data_train,sit_train)))

data_test = np.vstack((hand_test,run_test))
data_test = torch.from_numpy(np.vstack((data_test,sit_test)))

data_train = data_train.reshape(data_train.shape[0],1,3,25)
data_test = data_test.reshape(data_test.shape[0],1,3,25)

print("Total Input Training and Tesing Size: ",data_train.shape,data_test.shape)
num = data_train.shape[0]
train_dataset = TensorDataset(data_train, label_train)
train_loader = DataLoader(dataset=train_dataset,batch_size=20,shuffle=True)

test_dataset = TensorDataset(data_test, label_test)
test_loader = DataLoader(dataset=test_dataset,batch_size=data_test.shape[0],shuffle=True)


model = Action_Net()    # instantiate convolution neural network
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)   # optimize all cnn parameters
writer = SummaryWriter(comment='Action_Net')

for epoch in range(5):
    for step,(inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()  # clear last grad
        inputs = torch.tensor(inputs,dtype=torch.float32)
        out = model(inputs)
        loss = criterion(out, labels)  # calculate loss
        loss.backward()  # loss backward, calculate new data
        optimizer.step()  # add new weight to net parameters
        writer.add_graph(model, inputs)
        writer.add_scalar('Loss', loss, epoch*100+step)
        if step % 100 == 0:
            for i,(test_data,test_label) in enumerate(test_loader):
                test_data = torch.tensor(test_data, dtype=torch.float32)
                test_output = model(test_data)
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                accuracy = float((pred_y == test_label.data.numpy()).astype(int).sum()) / float(test_label.size(0))
                writer.add_scalar('Accuracy', accuracy, epoch*100+step)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
writer.close()
torch.save(model, 'model/net.pkl')                      # 保存整个神经网络的结构和模型参数
torch.save(model.state_dict(), 'model/net_params.pkl')  # 只保存神经网络的模型参数
