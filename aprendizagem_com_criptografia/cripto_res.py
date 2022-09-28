from numpy import savetxt
from numpy import loadtxt
import numpy as np
import torch
import random
import tenseal as ts
import matplotlib.pyplot as plt
from time import time



x_train_res = loadtxt(r'C:\Users\Pichau\OneDrive\Documentos\UFCG\PIBIC\x_train_res.csv', delimiter=',')
y_train_res = loadtxt(r'C:\Users\Pichau\OneDrive\Documentos\UFCG\PIBIC\y_train_res.csv', delimiter=',')
x_test_res = loadtxt(r'C:\Users\Pichau\OneDrive\Documentos\UFCG\PIBIC\x_test_res.csv', delimiter=',') 
y_test_res = loadtxt(r'C:\Users\Pichau\OneDrive\Documentos\UFCG\PIBIC\y_test_res.csv', delimiter=',')

y_train_res = y_train_res.reshape(len(y_train_res), -1)
y_test_res = y_test_res.reshape(len(y_test_res), -1)

x_train_res = torch.tensor(x_train_res, dtype=torch.float32)
y_train_res = torch.tensor(y_train_res, dtype=torch.float32)
x_test_res = torch.tensor(x_test_res, dtype=torch.float32)
y_test_res = torch.tensor(y_test_res, dtype=torch.float32)



parte = 128 
def split_train_test(x_train, y_train, x_test, y_test):

    idx_train = [i for i in range(len(x_train))]
    idx_test = [i for i in range(len(x_test))]

    random.shuffle(idx_train)
    random.shuffle(idx_test)

    delim1 = len(x_train) // parte
    delim2 = len(x_test) // parte

    train_idxs = idx_train[:delim1]
    test_idxs = idx_test[:delim2]

    return x_train[train_idxs], y_train[train_idxs], x_test[test_idxs], y_test[test_idxs]



def conta(y):
    cont1 = torch.abs(y) > 0.5
    cont0 = torch.abs(y) < 0.5
    print()
    print(cont1.float().mean())
    print(cont0.float().mean())
    print()



x_train, y_train, x_test, y_test = split_train_test(x_train_res, y_train_res, x_test_res, y_test_res)

savetxt('x_train_res' + str(parte) + '.csv', x_train, delimiter=',')
savetxt('x_test_res' + str(parte) + '.csv', x_test, delimiter=',')
savetxt('y_test_res' + str(parte) + '.csv', y_test, delimiter=',')

print("############# Data summary #############")
print(f"x_train has shape: {x_train.shape}")
print(f"y_train has shape: {y_train.shape}")
print("y_train has count :")
conta(y_train)
print(f"x_test has shape: {x_test.shape}")
print(f"y_test has shape: {y_test.shape}")
print("y_test has count :")
conta(y_test)
print("#######################################")


    
class LR(torch.nn.Module):

    def __init__(self, n_features):
        super(LR, self).__init__()
        self.lr = torch.nn.Linear(n_features, 1)
        
    def forward(self, x):
        out = torch.sigmoid(self.lr(x))
        return out

n_features = x_train.shape[1]

model = LR(n_features) ##########################################################################################

# use gradient descent with a learning_rate=1
optim = torch.optim.SGD(model.parameters(), lr=1)

# use Binary Cross Entropy Loss
criterion = torch.nn.BCELoss()

# define the number of epochs for both plain and encrypted training
EPOCHS = 4

def train(model, optim, criterion, x, y, epochs=EPOCHS):
    for e in range(1, epochs + 1):
        optim.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optim.step()
        print(f"Loss at epoch {e}: {loss.data}")
    return model

model = train(model, optim, criterion, x_train, y_train)


def accuracy(model, x, y):
    out = model(x)
    correct = torch.abs(y - out) < 0.5
    return correct.float().mean()

plain_accuracy = accuracy(model, x_test, y_test)
print(f"Accuracy on plain test_set: {plain_accuracy}")


class EncryptedLR():
    
    def __init__(self, torch_lr):
        self.weight = torch_lr.lr.weight.data.tolist()[0]
        self.bias = torch_lr.lr.bias.data.tolist()
        # we accumulate gradients and counts the number of iterations
        self._delta_w = 0
        self._delta_b = 0
        self._count = 0
        
    def forward(self, enc_x):
        enc_out = enc_x.dot(self.weight) + self.bias
        enc_out = EncryptedLR.sigmoid(enc_out)
        return enc_out
    
    def backward(self, enc_x, enc_out, enc_y):
        out_minus_y = (enc_out - enc_y)
        self._delta_w += enc_x * out_minus_y
        self._delta_b += out_minus_y
        self._count += 1
        
    def update_parameters(self):
        if self._count == 0:
            raise RuntimeError("You should at least run one forward iteration")
        # update weights
        # We use a small regularization term to keep the output
        # of the linear layer in the range of the sigmoid approximation
        self.weight -= self._delta_w * (1 / self._count) + self.weight * 0.05
        self.bias -= self._delta_b * (1 / self._count)
        # reset gradient accumulators and iterations count
        self._delta_w = 0
        self._delta_b = 0
        self._count = 0
    
    @staticmethod
    def sigmoid(enc_x):
        # We use the polynomial approximation of degree 3
        # sigmoid(x) = 0.5 + 0.197 * x - 0.004 * x^3
        # from https://eprint.iacr.org/2018/462.pdf
        # which fits the function pretty well in the range [-5,5]
        return enc_x.polyval([0.5, 0.197, 0, -0.004])
    
    def plain_accuracy(self, x_test, y_test):
        # evaluate accuracy of the model on
        # the plain (x_test, y_test) dataset
        w = torch.tensor(self.weight)
        b = torch.tensor(self.bias)
        out = torch.sigmoid(x_test.matmul(w) + b).reshape(-1, 1)
        correct = torch.abs(y_test - out) < 0.5
        return correct.float().mean()

    # hypotese implementqation 
    def hypotese(self, x_test):
        w = torch.tensor(self.weight)
        b = torch.tensor(self.bias)
        out = torch.sigmoid(x_test.matmul(w) + b).reshape(-1, 1)
        return out
    
    def encrypt(self, context):
        self.weight = ts.ckks_vector(context, self.weight)
        self.bias = ts.ckks_vector(context, self.bias)
        
    def decrypt(self):
        self.weight = self.weight.decrypt()
        self.bias = self.bias.decrypt()
        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

# parameters
poly_mod_degree = 8192
coeff_mod_bit_sizes = [31, 26, 26, 26, 26, 26, 26, 31]
# create TenSEALContext
ctx_training = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
ctx_training.global_scale = 2 ** 26
ctx_training.generate_galois_keys()
t_start = time()
enc_x_train = [ts.ckks_vector(ctx_training, x.tolist()) for x in x_train]
enc_y_train = [ts.ckks_vector(ctx_training, y.tolist()) for y in y_train]
t_end = time()
print(f"Encryption of the training_set took {int(t_end - t_start)} seconds")


normal_dist = lambda x, mean, var: np.exp(- np.square(x - mean) / (2 * var)) / np.sqrt(2 * np.pi * var)

def plot_normal_dist(mean, var, rmin=-10, rmax=10):
    x = np.arange(rmin, rmax, 0.01)
    y = normal_dist(x, mean, var)
    fig = plt.plot(x, y)
    
# plain distribution
lr = LR(n_features)
data = lr.lr(x_test)
mean, var = map(float, [data.mean(), data.std() ** 2])
plot_normal_dist(mean, var)
print("Distribution on plain data:")
plt.show()

# encrypted distribution
def encrypted_out_distribution(eelr, enc_x_test):
    w = eelr.weight
    b = eelr.bias
    data = []
    for enc_x in enc_x_test:
        enc_out = enc_x.dot(w) + b
        data.append(enc_out.decrypt())
    data = torch.tensor(data)
    mean, var = map(float, [data.mean(), data.std() ** 2])
    plot_normal_dist(mean, var)
    print("Distribution on encrypted data:")
    plt.show()

eelr = EncryptedLR(lr)
eelr.encrypt(ctx_training)
encrypted_out_distribution(eelr, enc_x_train)


eelr = EncryptedLR(LR(n_features))
accuracy = eelr.plain_accuracy(x_test, y_test)
torch.save(eelr, 'modelo_parcial_res0_' + str(parte) + '.pth')
print(f"Accuracy at epoch #0 is {accuracy}")

times = []
for epoch in range(EPOCHS):
    eelr.encrypt(ctx_training)
    
    # if you want to keep an eye on the distribution to make sure
    # the function approxiamation is still working fine
    # WARNING: this operation is time consuming
    encrypted_out_distribution(eelr, enc_x_train)
    
    t_start = time()
    for enc_x, enc_y in zip(enc_x_train, enc_y_train):
        enc_out = eelr.forward(enc_x)
        eelr.backward(enc_x, enc_out, enc_y)
    eelr.update_parameters()
    t_end = time()
    times.append(t_end - t_start)
    
    eelr.decrypt()
    accuracy = eelr.plain_accuracy(x_test, y_test)
    
    #salvando modelos parciais
    torch.save(eelr, 'modelo_parcial_res' + str(epoch + 1) + '_' + str(parte) + '.pth')

    print(f"Accuracy at epoch #{epoch + 1} is {accuracy}")



print(f"\nAverage time per epoch: {int(sum(times) / len(times))} seconds")
print(f"Final accuracy is {accuracy}")



diff_accuracy = plain_accuracy - accuracy
print(f"Difference between plain and encrypted accuracies: {diff_accuracy}")
if diff_accuracy < 0:
    print("Oh! We got a better accuracy when training on encrypted data! The noise was on our side...")