import torch
import tenseal as ts

from sklearn.metrics import classification_report

from numpy import loadtxt

parte = 128
epoch = 6  

x_test = loadtxt(r'C:\Users\Pichau\OneDrive\Documentos\UFCG\PIBIC\x_test_res' + str(parte)+ '.csv', delimiter=',')
y_test = loadtxt(r'C:\Users\Pichau\OneDrive\Documentos\UFCG\PIBIC\y_test_res' + str(parte)+ '.csv', delimiter=',')
x_train = loadtxt(r'C:\Users\Pichau\OneDrive\Documentos\UFCG\PIBIC\x_train_res' + str(parte)+ '.csv', delimiter=',')

y_test = y_test.reshape(len(y_test),-1)

x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
x_train = torch.tensor(x_train, dtype=torch.float32)

print(x_test.dtype)
# x_train = torch.tensor([])
# torch.load(x_train, r'C:\Users\Pichau\OneDrive\Documentos\UFCG\PIBIC\tensors.pt')
# x_test = torch.tensor([])
# torch.load(x_test, r'C:\Users\Pichau\OneDrive\Documentos\UFCG\PIBIC\tensors.pt')
# y_test = torch.tensor([])
# torch.load(y_test, r'C:\Users\Pichau\OneDrive\Documentos\UFCG\PIBIC\tensors.pt')


print("############# Data summary #############")
print(f"x_test has shape: {x_test.shape}")
print(f"y_test has shape: {y_test.shape}")
print("#######################################")


class LR(torch.nn.Module):

    def __init__(self, n_features):
        super(LR, self).__init__()
        self.lr = torch.nn.Linear(n_features, 1)
        
    def forward(self, x):
        out = torch.sigmoid(self.lr(x))
        return out

n_features = x_train.shape[1]

model = LR(n_features)



optim = torch.optim.SGD(model.parameters(), lr=1)


criterion = torch.nn.BCELoss()


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

    # hypotese implementation 
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



PATH = r"C:\Users\Pichau\OneDrive\Documentos\UFCG\PIBIC\modelo_parcial_res" + str(epoch) +"_" + str(parte) + ".pth"

model = torch.load(PATH)
# model.eval()

hipotese = model.hypotese(x_test)
hipotese_r = hipotese > 0.5 

print(classification_report(y_test, hipotese_r))