import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self,max_iter = 1000,eps=1e-3,projection = None):
        self.max_iter = max_iter
        self.eps = eps
        self.projection = projection or (lambda x: x) #projection fonction identite par default

    def fit(self,data,y):
        data = self.projection(data)
        self.w = np.random.random((1,data.shape[1]))
        self.histo_w  = np.zeros((self.max_iter,data.shape[1]))
        self.histo_f = np.zeros((self.max_iter,1))
        ylab=set(y.flat)
        if len(ylab)!=2:
            print("pas bon nombres de labels (%d)" % (ylab,))
            return
        self.labels = {-1: min(ylab), 1:max(ylab)}
        y = 2*(y!=self.labels[-1])-1
        i=0
        while i<self.max_iter:
            idx = range(len(data))
            for j in idx:
                self.w = self.w - self.get_eps()*self.loss_g(data[j],y[j:(j+1)])
            self.histo_w[i]=self.w
            self.histo_f[i]=self.loss(data,y)
            if i % 100==0: print(i,self.histo_f[i])
            i+=1
    def predict(self,data):
        data = self.projection(data)
        return np.array([self.labels[x] for x in np.sign(data.dot(self.w.T)).flat]).reshape((len(data),))
    def score(self,data,y):
        return np.mean(self.predict(data)==y)
    def get_eps(self):
        return self.eps
    def loss(self,data,y):
        return hinge(self.w,data,y)
    def loss_g(self,data,y):
        return grad_hinge(self.w,data,y)


### Test des fonctions hinge, grad_hinge
w = np.random.random((3,))
data = np.random.random((100,3))
y = np.random.randint(0,2,size = (100,1))*2-1
print(hinge(w,data,y), hinge(w,data[0],y[0]), hinge(w,data[0,:],y[0]))
print(grad_hinge(w,data,y),grad_hinge(w,data[0],y[0]).shape,grad_hinge(w,data[0,:],y[0]).shape)

### Generation de donnees
xtrain,ytrain = gen_arti(data_type=0,epsilon=0.2)
xtest,ytest = gen_arti(data_type=0,epsilon=0.2)

plt.ion()

### Apprentissage
model=Perceptron(eps=1e-2)
model.fit(xtrain,ytrain)
print("score en train : ",model.score(xtrain,ytrain))
print("score en test : ",model.score(xtrain,ytrain))

#### Tracer de frontiere
plt.figure()
plot_frontiere(xtrain,p.predict,50)
plot_data(xtrain,ytrain)

#### Visualisation de la fonction de cout
plt.figure()
plot_frontiere(data=None,f=lambda w : hinge(w,xtrain,ytrain))
