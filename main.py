
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


########################################################################
import numpy as np
import cv2			# pip3 install opencv-python
from scipy import ndimage
import warnings; warnings.filterwarnings('ignore')	# due to overflow at exp
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 	# because of TF-log-spam
import tensorflow as tf
import random as rd
import pickle

########################################################################


########################################################################
class Network_Base:
    def __init__(self):
        # Template list (all allowed input templates)
        self.pattern = [str(i) for i in range(10)]

        # Input template edge length
        self.ILE = 28

        # network parameters
        self.NPL = [ self.ILE**2, 128, len(self.pattern) ]
        self.N = len(self.NPL)

        # Batch-Size
        self.batch_size = 32

        # Data from MNIST (digits 0-9), flattening (28x28->28*28) and normalization
        # [0] Teaching
        # [1] Testing
        # [:][0] Input pattern 28x28 integers
        # [:][1] Output pattern 1 Integer from {0,1,...9}
        (self.DataTrainX, self.DataTrainY), (self.DataTestX, self.DataTestY) = tf.keras.datasets.mnist.load_data()
        self.DataTrainX     = self.DataTrainX / 255.
        self.DataTrainXflat = np.array([ m.flatten() for m in self.DataTrainX ])
        self.DataTrainYflat = np.array([ [ 1 if m==p else 0 for m in range(10) ] for p in self.DataTrainY ])
        self.DataTestX      = self.DataTestX / 255.
        self.DataTestXflat  = np.array([ m.flatten() for m in self.DataTestX ])
        self.DataTestYflat  = np.array([ [ 1 if m==p else 0 for m in range(10) ] for p in self.DataTestY ])


    def PrintPattern(self, xp, yp):
        print(yp)
        for x in range(28):
            for y in range(28):
                p = xp[y+28*x]
                if  p > 0.8:
                    print('*', end='')
                elif p > 0.6:
                    print('o', end='')
                elif p > 0.4:
                    print('·', end='')
                else:
                    print(' ', end='')
            print()
        print()
########################################################################


########################################################################
class Network_Simple(Network_Base):
    def __init__(self):
        super().__init__()

        self.W = [None] + [np.random.randn(i, j) for i, j in zip(self.NPL[1:], self.NPL[:-1])]
        self.B = [None] + [np.random.randn(ni) for ni in self.NPL[1:]]
        self.Z = [None, self.Relu, self.S]

        self.x = []
        self.a = []



    # Output function Z = sigmoidal
    def S(self, z, d=0):
        if d == 1:
            return self.S(z, 0)*(1-self.S(z, 0))
        else:
            return 1/(1 + np.exp(-z))

    # Output function Z = relu
    def Relu(self, z, d=0):
        if d == 1:
            return np.where(z > 0, 1, 0)
        else:
            return np.where(z > 0, z, 0)

    # Feedback pass for input pattern X
    def FeedForward(self, X):
        X = X.flatten()

        x_max = max(X)
        if x_max == 0:
            x_max = 1
        X = X / x_max

        # k = 0
        self.x = [X]
        self.a = [X]
        y = X
        # k > 0
        for k in range(1, self.N):
            self.x.append(y)
            self.a.append(np.dot(self.W[k], self.x[k]) + self.B[k])
            if k == 2:
                y = self.Relu(self.a[k])
            else:
                y = self.S(self.a[k])
        return y



    # access to hidden neurons' activations
    def Activation(self, X):
        return self.x[-1]

    # network training
    # -> all patterns multiple times ('epochs'), bundled into batches
    def Fit(self, ep=1, bs=32):
        self.batch_size = bs

        dataTrainexamplesXY = []
        wholeTrainSet = zip(self.DataTrainXflat, self.DataTrainYflat)
        for x, y in wholeTrainSet:
            arr = [x, y]
            dataTrainexamplesXY.append(arr)


        for epoch in range(ep):
            print('Epoch number ', epoch)
            np.random.shuffle(dataTrainexamplesXY)
            dataTrainSets = []
            for currentBatch in range(0, len(dataTrainexamplesXY), bs):
                dataTrainSets = dataTrainSets + dataTrainexamplesXY[currentBatch:(currentBatch+bs)]
            for currentBatch, number in dataTrainSets:
                self.update(currentBatch, number)
            self.Test()





    # update weights and biases with pattern of one batch
    def update(self, batch, targetNumber):
        # weight
        singleNabla = []
        for weight in self.W[1:]:
            singleNabla = singleNabla + list(np.zeros(len(weight)))
        nablaW = [None] + singleNabla

        # bias
        singleNabla = []
        for bias in self.B[1:]:
            singleNabla = singleNabla + list(np.zeros(len(bias)))
        nablaB = [None] + singleNabla

        nablaW, nablaB = self.BackProp(batch, targetNumber, nablaW, nablaB)

        for i in range(1, self.N):
            self.W[i] = self.W[i] + nablaW[i]
            self.B[i] = self.B[i] + nablaB[i]



    # calculate changes in weights and biases for one pattern
    # add to nabla_W and nabla_B parameters
    def BackProp(self, X, Y, nabla_W, nabla_B):
        #output
        delta = -(self.FeedForward(X) - Y) * self.S(self.a[self.N - 1], 1)
        eta = 1
        gamma = eta / (1 + eta * (1+self.x[self.N-1]@self.x[self.N-1]))
        nabla_W[self.N-1] = nabla_W[self.N-1] + gamma * np.transpose([delta]) * self.x[self.N-1]
        nabla_B[self.N-1] = nabla_B[self.N-1] + gamma * delta

        #hidden
        for k in range(2, self.N):
            delta = -self.S(self.a[k-1], 1) * (delta@(self.W[k]))
            gamma = eta / (1 + eta * (1+self.x[k-1]@self.x[k-1]))
            nabla_W[k - 1] = nabla_W[k - 1] + gamma * np.transpose([delta]) * self.x[k - 1]
            nabla_B[k - 1] = nabla_B[k - 1] + gamma * delta

        return [nabla_W, nabla_B]


    # Determine absolute deviation for all
    # Pattern in 'testfile'
    def Test(self):
        loss, acc = 0, 0
        N = len(self.DataTestY)
        for xt,yt,v in zip(self.DataTestXflat, self.DataTestYflat, self.DataTestY):
            y = self.FeedForward(xt)
            loss += np.sqrt(sum(y-yt)**2)
            if v == y.argmax():
                acc += 1
        print(f'Simple-NN.Test: acc = {100*acc/N:5.2f}%  loss = {loss/N:5.2f}')
########################################################################


########################################################################
class Board:
    def __init__(self, nn):
        print("\n----------------\n Keys:\n  \' \'=reset\n  \'q\'=quit\n  \'f\'=toggle blur\n----------------\n")

        self.NN = nn

        # display parameters
        self.use_blurr = False
        self.screen_width, self.screen_height = 600, 780

        # input zoom window
        self.Win1a_edge = 196
        self.Win1a_x0 = 20
        self.Win1a_y0 = self.screen_height - self.Win1a_edge - 20

        # input grab window
        self.Win1b_edge = 50

        # input draw window
        self.Win1c_edge = self.Win1a_edge
        self.Win1c_x0 = self.Win1a_x0 + self.Win1a_edge + 20
        self.Win1c_y0 = self.screen_height - self.Win1c_edge - 20

        # hidden layer window
        self.Win2_dx = 24*16
        self.Win2_dy = 24*8
        self.Win2_x0 = 20
        self.Win2_y0 = self.Win1a_y0 - self.Win2_dy - 60

        # output layer window
        self.Win3_dx = 550
        self.Win3_dy = 200
        self.Win3_x0 = 20
        self.Win3_y0 = self.Win2_y0 - self.Win3_dy - 60

        # define board
        cv2.namedWindow('Board')
        cv2.moveWindow('Board', 10, 10)
        cv2.setMouseCallback('Board', self.mouse_cb)
        self.clear_screen()


    # reset board
    def clear_screen(self):
        # Drawing
        self.pen_down, self.but3_down, self.old_x , self.old_y = False, False, 0, 0
        self.px_min = 10000
        self.px_max = 0
        self.py_min = 10000
        self.py_max = 0

        # windows
        self.board1 = np.zeros((self.screen_height, self.screen_width), np.uint8)
        self.board2 = np.zeros((self.screen_height, self.screen_width), np.uint8)
        cv2.rectangle(self.board1, (self.Win1a_x0-1, self.Win1a_y0-1), (self.Win1a_x0 + self.Win1a_edge+1, self.Win1a_y0 + self.Win1a_edge+1), color=255, thickness=1)
        cv2.rectangle(self.board2, (self.Win1c_x0-1, self.Win1c_y0-1), (self.Win1c_x0 + self.Win1c_edge+1, self.Win1c_y0 + self.Win1c_edge+1), color=25, thickness=-1)
        cv2.rectangle(self.board1, (self.Win1c_x0-1, self.Win1c_y0-1), (self.Win1c_x0 + self.Win1c_edge+1, self.Win1c_y0 + self.Win1c_edge+1), color=255, thickness=1)
        cv2.rectangle(self.board1, (self.Win2_x0-1, self.Win2_y0-1), (self.Win2_x0 + self.Win2_dx+1, self.Win2_y0 + self.Win2_dy+1), color=255, thickness=1)
        cv2.rectangle(self.board1, (self.Win3_x0-1, self.Win3_y0-1), (self.Win3_x0 + self.Win3_dx+1, self.Win3_y0 + self.Win3_dy+1), color=255, thickness=1)
        cv2.putText(self.board1, "Input-Zoom",   (self.Win1a_x0, self.Win1a_y0-14), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, cv2.LINE_AA)
        cv2.putText(self.board1, "Input Draw",   (self.Win1c_x0, self.Win1c_y0-14), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 2, cv2.LINE_AA)
        cv2.putText(self.board1, "Hidden-Layer", (self.Win2_x0, self.Win2_y0-14), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, cv2.LINE_AA)
        cv2.putText(self.board1, "Output-Layer", (self.Win3_x0, self.Win3_y0-14), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, cv2.LINE_AA)
        self.screen_plot()


    # update network information
    def screen_plot(self):
        # Input-Ausschnitt
        grab0 = self.cutout_input_win()
        # + Blurr
        if self.use_blurr:
            grab0 = ndimage.filters.gaussian_filter( grab0, sigma=2)
        # layer activations
        xin  = cv2.resize( grab0, dsize=(self.NN.ILE, self.NN.ILE), interpolation=cv2.INTER_CUBIC)
        yout = self.NN.FeedForward(xin)
        yhid = self.NN.Activation(xin)

        # output layer
        self.Win3 = np.zeros((self.Win3_dy, self.Win3_dx), np.uint8)
        yout_max = max(yout)
        yout_max = 1 if yout_max==0 else yout_max
        yout_sum = sum(yout)
        yout_sum = 1 if yout_sum==0 else yout_sum
        for i,y in enumerate(yout):
            cv2.putText(self.Win3, self.NN.pattern[i], (30+i*50, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 2, cv2.LINE_AA)
            cv2.putText(self.Win3, str(int(y*100./yout_sum))  , (20+i*50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 2, cv2.LINE_AA)
            cv2.rectangle(self.Win3, (20+i*50, 165), (60+i*50, 160-int(130*y/yout_max)), color=255, thickness=-1)
        cv2.putText(self.Win3, "%"  , (520, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 2, cv2.LINE_AA)
        self.board1[self.Win3_y0:self.Win3_y0+self.Win3_dy, self.Win3_x0:self.Win3_x0+self.Win3_dx] = self.Win3

        # hidden layer
        self.Win2 = np.zeros((self.Win2_dy, self.Win2_dx), np.uint8)
        yhid_max = max(yhid)
        yhid_max = 1 if yhid_max==0 else yhid_max
        imax, jmax, di, dj = 16,8,24,24
        for j in range(jmax):
            for i in range(imax):
                cv2.circle(self.Win2, (di*i+di//2, dj*j+dj//2), di//2, color=int(255*yhid[i+j*imax]/yhid_max), thickness=-1)
        self.board1[self.Win2_y0:self.Win2_y0+self.Win2_dy, self.Win2_x0:self.Win2_x0+self.Win2_dx] = self.Win2

        # input layer
        self.Win1a = np.zeros((self.Win1a_edge, self.Win1a_edge), np.uint8)
        xin_max = max(xin.flatten())
        xin_max = 1 if xin_max==0 else xin_max
        imax, jmax, di, dj = self.NN.ILE,self.NN.ILE,7,7
        for j in range(jmax):
            for i in range(imax):
                cv2.circle(self.Win1a, (di*i+di//2, dj*j+dj//2), di//2, color=int(255*xin[j,i]/xin_max), thickness=-1)
        self.board1[self.Win1a_y0:self.Win1a_y0+self.Win1a_edge,
        self.Win1a_x0:self.Win1a_x0+self.Win1a_edge] = self.Win1a


    # clipping for input window
    def clipping(self,x,y):
        return max(min(x, self.Win1c_x0 + self.Win1c_edge), self.Win1c_x0), max(min(y, self.Win1c_y0 + self.Win1c_edge), self.Win1c_y0)


    # win0 ausschneiden
    def cutout_input_win(self):
        xc = (self.px_max+self.px_min)
        xc = max(2*self.Win1c_x0+self.Win1b_edge, xc)
        xc = min(2*(self.Win1c_x0 + self.Win1c_edge) - self.Win1b_edge, xc)
        yc = (self.py_max+self.py_min)
        yc = max(2*self.Win1c_y0+self.Win1b_edge, yc)
        yc = min(2*(self.Win1c_y0 + self.Win1c_edge) - self.Win1b_edge, yc)
        ya = (yc - self.Win1b_edge)//2
        yb = (yc + self.Win1b_edge)//2
        xa = (xc - self.Win1b_edge)//2
        xb = (xc + self.Win1b_edge)//2
        cv2.rectangle(self.board2, (self.Win1c_x0-1, self.Win1c_y0-1), (self.Win1c_x0 + self.Win1c_edge+1, self.Win1c_y0 + self.Win1c_edge+1), color=25, thickness=-1)
        cv2.rectangle(self.board2, (xa-1, ya-1), (xb+1, yb+1), color=255, thickness=1)
        return self.board1[ya:yb,xa:xb]


    # Mouse-Callback
    def mouse_cb(self, event, x, y, flags, param):
        self.but3_down = flags & 0x4
        x,y = self.clipping(x,y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pen_down = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.pen_down:
                # line drawing
                cv2.line(self.board1, (self.old_x, self.old_y), (x,y), color=255, thickness=2)
                # for sub-area with drawing
                self.px_min = min(x, self.px_min)
                self.px_max = max(x, self.px_max)
                self.py_min = min(y, self.py_min)
                self.py_max = max(y, self.py_max)
                self.screen_plot()
            self.old_x, self.old_y = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            self.pen_down = False


    # Input Pattern for NN prediction
    def RunNet(self):
        print("Running Network...")
        self.clear_screen()
        while True:
            key = cv2.waitKey(20)
            if key in [27, ord('q')]:
                break
            elif key == ord('f'):
                self.use_blurr = not self.use_blurr
                self.screen_plot()
            elif key == ord(' '):
                self.clear_screen()
            cv2.imshow('Board', self.board1 | self.board2)

    def __del__(self):
        cv2.destroyAllWindows()
########################################################################



########################################################################
if __name__ == "__main__":

    NN = Network_Simple()

    NN.Test()
    NN.Fit(1,32)
    NN.Test()

    B = Board(NN)
    B.RunNet()
########################################################################


