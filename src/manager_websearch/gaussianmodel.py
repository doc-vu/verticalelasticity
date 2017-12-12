
# libaries
import csv
import numpy as np
import matplotlib.pyplot as pl
import pyGPs as gp
from sklearn.cluster import KMeans
from math import log, pi, sqrt
import scipy.stats
import logging
#from cloudmodel import CloudModel
from cloudmodel import *

class GaussianModel(CloudModel):
    'Cloud model that predict the latency of a webserver'

    def __init__ (self):
	return

    #def __init__(self, winSize = 600, gpWinSize = 50 ,featureWinSize=3 ,nu_cluster = 2,
    #             controlFeedback = 1, alpha = 3, beta = 1, sigma = 0.9, controllerDelay = 5 ):
        #  online Learning parameters and variables
        self.__winSize = winSize
        self.__gpWinSize = gpWinSize
        self.__featureWinSize = featureWinSize
        self.__modelInitilization  = 0  # 0: not initialozed, 1: learn and predict
        self.offlineInit = 0            # 1: use an offline data to init the model, 0: init online by buffering data
        
        self.__fileName = "result04.csv"
        
        # load data
        self.__inData = []
        self.__outData = []
        self.__statData = []
        self.__gpfX = []
        self.__gpfY = []


        self.__sysState = []
        # Number of clusters
        self.__Nc = nu_cluster
        self.__clusterFeatures = []
        self.__featurewin = []
        self.__onlineCluster = 1

        # model status
       
        self.__modelTimeIdx = 0
        self.__modelstate = 0
        # model control output
        self.__controlOut = []
        self.__controlFeedback = controlFeedback
        self.__alpha = alpha
        self.__beta = beta
        self.__sigma = sigma
        self.__controllerDelay = controllerDelay
        self.__controllerCounter = 0
        # model objects
        # state models
        self.__GPs = []

        # forecast model
        self.__gpFmdl = gp.GPR()

        # classifier model
        self.__myCluster = KMeans(n_clusters=self.__Nc ,init='random', random_state=0)



        # variables to store prediction data for evaluation and plotting:
        # forecasting workload variables
        self.__In_mean = []
        self.__In_si = []
        self.__In_pred_data = []

        # classification variables
        self.__sysModes = []

        # for output variables
        self.__out_mean =[]
        self.__out_si = []
        self.__out_pred_data = []
        # for i in self.__Nc:
        #     self.__out_mean.append([])
        #     self.__out_si.append([])
        #     self.__out_pred_data.append([])


        # other
        self.verbose = 0

        
        if self.offlineInit:
            # pass the offline data to initialize the model
            # input, output and model state indices in data
            indxIn = 36  # range(0,9)      # Input indices [1:9]
            indxOut = 38  # range(9,18)      # Output indices [10:18]
            indxS = [18, 26, 27, 35]  # range(18,36)     # State indices  [20:36]
            inData = 0
            outData = 0
            statData = []
            with open(self.__fileName, 'rt') as dataFile:
                reader = csv.reader(dataFile, delimiter=',')
                for row in reader:
                    inData = float(row[indxIn])
                    outData = float(row[indxOut])
                    statData = [float(row[i]) for i in indxS]
                    self.__initModel(inData, statData, outData)
    
                    if self.__modelInitilization:
                        break

    def __featureExtraction(self, X):
        m = len(X[0])  # number of measurments
        n = len(X)  # Window Size
        Y = []  # feature Vector

        # Calculate mean feature
        for i in range(m):
            num = []
            for j in range(n):
                num.append(X[j][i])

            Y.append(np.mean(num))


            # Calculate Variance feature
            # Y.append(np.var(num))

        return Y

    def __initModel(self, sysIn, sysStat, sysOut):

        self.__inData.append(sysIn)
        self.__outData.append(sysOut)
        self.__statData.append(sysStat)

        self.__featurewin.append(sysStat)
        while len(self.__featurewin) < self.__featureWinSize:
            self.__featurewin.append(sysStat)

        while len(self.__featurewin) > self.__featureWinSize:
            self.__featurewin.pop(0)

        curr_feature = self.__featureExtraction(self.__featurewin)
        # update feature data  (slid the window)
        self.__clusterFeatures.append(curr_feature)

        # update forecast model and system indx
        # add the new data  (slid the win)
        self.__gpfX.append(self.__modelTimeIdx)
        self.__gpfY.append(sysIn)

        while len(self.__gpfX) > self.__gpWinSize:
            # delete oldest data
            self.__gpfX.pop(0)
            self.__gpfY.pop(0)

        self.__modelTimeIdx += 1

        if len(self.__clusterFeatures) >= self.__winSize:
            # Learn the models  (align the data to input output format  Y(k+1) = f(x(k),u(k+1)))
            self.__inData.pop(0)
            self.__outData.pop(0)
            # self.__statData.pop()
            # self.__clusterFeatures.pop()

            # classify the data
            clustersX = self.__myCluster.fit_predict(self.__clusterFeatures)
            if self.verbose:
                print ("training state-space models using  GPs")

            for i in range(self.__Nc):
                gprX = []
                gprY = []

                for j in range(len(clustersX) - 1):
                    if clustersX[j] == i:
                        gprX.append([self.__inData[j]] + self.__statData[j])
                        gprY.append(self.__outData[j])
                gprX = np.array(gprX)
                gprY = np.array(gprY)
                gpmdl = gp.GPR()
                m = gp.mean.Zero()
                RBF_hyp_init = [0.5] * (len(
                    sysStat) + 2)  # [13.9310228936928,2.54640381722411,0.177686434357263,12.5490563084955,162.467937309584,3.38074333489536]
                k = gp.cov.RBFard(D=None, log_ell_list=RBF_hyp_init[:-1], log_sigma=RBF_hyp_init[-1])
                gpmdl.setPrior(mean=m, kernel=k)
                if self.verbose:
                    print ("training GP of mode: " + str(i))
                # gpmdl.getPosterior(gprX,gprY)
                gpmdl.setNoise(log_sigma=np.log(0.8))

                gpmdl.setOptimizer('Minimize')  # ('Minimize');
                gpmdl.optimize(gprX, gprY)  # ,numIterations=100)

                self.__GPs.append(gpmdl)

            if self.verbose:
                print ("training forecast Model using GP")

            try:
                k_f = gp.cov.RBF(log_ell=1, log_sigma=1)
                self.__gpFmdl.setPrior(mean=gp.mean.Zero(), kernel=k_f)
                self.__gpFmdl.setNoise(log_sigma=np.log(0.8))
                self.__gpFmdl.setOptimizer('BFGS')  # ('Minimize');
                self.__gpFmdl.optimize(np.array(self.__gpfX), np.array(self.__gpfY))  # ,numIterations=100)
            except:
                print('can quasi-newton it (forecast)')
                self.__gpFmdl = gp.GPR()
                k_f = gp.cov.RBF(log_ell=1, log_sigma=1)
                self.__gpFmdl.setPrior(mean=gp.mean.Zero(), kernel=k_f)
                self.__gpFmdl.setNoise(log_sigma=np.log(0.8))
                self.__gpFmdl.setPrior(mean=gp.mean.Zero(), kernel=k_f)
                self.__gpFmdl.setOptimizer('Minimize')  # ('Minimize');
                self.__gpFmdl.optimize(np.array(self.__gpfX), np.array(self.__gpfY))  # , numIterations=100)
            self.__sysState = self.__statData[-2]
            self.__modelInitilization = 1

    def __updateModel(self, sysIn, sysStat, sysOut):

        # update the forecast model
        if self.verbose:
            print ("update forecast Model")
        # add the new data  (slid the win)
        gpfX = np.append(self.__gpFmdl.x, np.array([self.__modelTimeIdx]).reshape(-1, 1), axis=0)
        gpfY = np.append(self.__gpFmdl.y, np.array([sysIn]).reshape(-1, 1), axis=0)

        while gpfY.size > self.__gpWinSize:
            # delete oldest data
            gpfX = np.delete(gpfX, 0, 0)
            gpfY = np.delete(gpfY, 0, 0)
        # get the old hyp
        hyp_f = self.__gpFmdl.covfunc.hyp

        # relearn the model with the old hyp as a prior model
        try:
            self.__gpFmdl = gp.GPR()
            k_f = gp.cov.RBF(log_ell=hyp_f[0], log_sigma=hyp_f[1])
            self.__gpFmdl.setPrior(mean=gp.mean.Zero(), kernel=k_f)
            self.__gpFmdl.setNoise(log_sigma=np.log(0.8))
            self.__gpFmdl.setOptimizer('SCG')
            self.__gpFmdl.optimize(gpfX, gpfY)
        except:
            print('cannot SCG it forecast')
            self.__gpFmdl = gp.GPR()
            k_f = gp.cov.RBF(log_ell=hyp_f[0], log_sigma=hyp_f[1])
            self.__gpFmdl.setPrior(mean=gp.mean.Zero(), kernel=k_f)
            self.__gpFmdl.setNoise(log_sigma=np.log(0.8))
            self.__gpFmdl.setOptimizer('Minimize')
            self.__gpFmdl.optimize(gpfX, gpfY)

        # Update cluster model
        if self.__onlineCluster:
            if self.verbose:
                print ("update cluster Model ...")
            self.__myCluster = KMeans(n_clusters=self.__Nc,
                                      init=self.__myCluster.cluster_centers_,
                                      random_state=0,n_init=1)

        # update feature window (slide the win)
        self.__featurewin.append(sysStat)
        self.__featurewin.pop(0)
        # calculate new feature
        curr_feature = self.__featureExtraction(self.__featurewin)
        # update feature data  (slid the window)
        self.__clusterFeatures.append(curr_feature)
        self.__clusterFeatures.pop(0)
        # update the clusterer
        self.__myCluster.fit(self.__clusterFeatures)

        # update system state GP model
        if self.verbose:
            print ("update system Models ...")
        # pull the model used for the last prediction
        gprMdl = self.__GPs[self.__modelstate]
        newgprX = np.array([sysIn] + self.__sysState).reshape(1, -1)
        gprX = np.append(gprMdl.x, newgprX, axis=0)
        gprY = np.append(gprMdl.y, np.array([sysOut]).reshape(1, -1), axis=0)
        # gprMdl.x = np.append(gprMdl.x, Xs, axis=0)
        # gprMdl.y = np.append(gprMdl.y, [outData[i]], axis=0)

        while gprY.size > self.__gpWinSize:
            gprX = np.delete(gprX, 0, 0)
            gprY = np.delete(gprY, 0, 0)

        hyp = gprMdl.covfunc.hyp

        gprMdl = gp.GPR()
        m = gp.mean.Zero()
        # k = gp.cov.SumOfKernel(gp.cov.RBFard(D=None, log_ell_list=hyp, log_sigma=1.),gp.cov.Noise(1))
        k = gp.cov.RBFard(D=None, log_ell_list=hyp[:-1], log_sigma=hyp[-1])
        gprMdl.setPrior(mean=m, kernel=k)
        # gprMdl.getPosterior(gprX,gprY)
        gprMdl.setNoise(log_sigma=np.log(0.81))
        try:
            gprMdl.setOptimizer('SCG')
            gprMdl.optimize(gprX, gprY)
        except:
            print('cannot SCG it ')
            gprMdl = gp.GPR()
            m = gp.mean.Zero()
            # k = gp.cov.SumOfKernel(gp.cov.RBFard(D=None, log_ell_list=hyp, log_sigma=1.),gp.cov.Noise(1))
            k = gp.cov.RBFard(D=None, log_ell_list=hyp[:-1], log_sigma=hyp[-1])
            gprMdl.setPrior(mean=m, kernel=k)
            # gprMdl.getPosterior(gprX,gprY)
            gprMdl.setNoise(log_sigma=np.log(0.81))
            gprMdl.setOptimizer('Minimize')
            gprMdl.optimize(gprX, gprY)
        self.__GPs[self.__modelstate] = gprMdl

        # Update system state
        self.__sysState = sysStat

        # save the data for Error calculation and prediction evaluation
        self.__In_pred_data.append(sysIn)
        self.__out_pred_data.append(sysOut)

    def __predictPerformance(self):

        if self.verbose:
            print ("forecasting and predicting ...")
        # forecast workLoad (predict workload input)
        INm, INsi, foom, foos2, foolp = self.__gpFmdl.predict(np.array([self.__modelTimeIdx + 1]).reshape(1, -1))

        if INm[0] < 0:  # bound the prediction
            INm[0] = 0
        self.__In_mean.append(np.asscalar(INm[0]))
        self.__In_si.append(np.asscalar(INsi[0]))

        # predict Mode
        curr_feature = self.__clusterFeatures[-1]
        predCluster = self.__myCluster.predict(np.array(curr_feature).reshape(1, -1))
        self.__modelstate = np.asscalar(predCluster[0])
        self.__sysModes.append(self.__modelstate)

        # initialize the return
        Y_mean = 0
        Y_si = 0

        # calculate the input
        Xs = np.array([np.asscalar(INm[0])] + self.__sysState).reshape(1, -1)
        # pull the model
        gprMdl = self.__GPs[self.__modelstate]
        # predict the system performance (latency)
        ym, ysi, fm, fs2, lp = gprMdl.predict(Xs)

        if ym[0] < 0:  # bound the prediction
            ym[0] = 0

        Y_mean = np.asscalar(ym[0])
        Y_si = np.asscalar(ysi[0])

        self.__out_mean.append(Y_mean)
        self.__out_si.append(Y_si)

        return Y_mean, Y_si

    def sysControl(self, sysIn, sysStat, sysOut, controlFeedback, MaxN, MinN):
        # sysIn: average Input (scalar)
        # sysStat: current utilization vector (list of four element)
        # sysOut: Average Latency (scalar)
        # controlFeedback: current DB number of CPU cores
        self.__controlFeedback = controlFeedback

        if not(self.__modelInitilization):
            self.__initModel(sysIn,sysStat,sysOut)
        else:
            self.__updateModel(sysIn,sysStat,sysOut)

        controlSignal = controlFeedback
        if not(self.__modelInitilization):
            controlSignal = self.__controlFeedback
            self.__controllerCounter = self.__controllerDelay
        else:

            ym, ysi = self.__predictPerformance()
            p_lge_alpha = (1 - scipy.stats.norm(ym, ysi).cdf(self.__alpha))
            P_lle_beta = scipy.stats.norm(ym, ysi).cdf(self.__beta)
            if self.verbose:
                print('predicted latency: N(' +str(ym)  +','+str(ysi) +')')
                print('P(L > '+ str(self.__alpha) +')= ' + str(p_lge_alpha))
                print('P(L < ' + str(self.__beta) + ')= ' + str(P_lle_beta))
    
                logging.info('predicted latency: N(' + str(ym) + ',' + str(ysi) + ')')
                logging.info('P(L > ' + str(self.__alpha) + ')= ' + str(p_lge_alpha))
                logging.info('P(L < ' + str(self.__beta) + ')= ' + str(P_lle_beta))
            if p_lge_alpha >= self.__sigma:
                if self.__controlFeedback < MaxN and self.__controllerCounter >= self.__controllerDelay:
                    controlSignal = self.__controlFeedback + 1
                    self.__controllerCounter = 0
    
            elif P_lle_beta >= self.__sigma:
                if self.__controlFeedback > MinN and self.__controllerCounter >= self.__controllerDelay:
                    controlSignal = self.__controlFeedback - 1
                    self.__controllerCounter = 0
    
            self.__controllerCounter += 1

        # Save the control history
        self.__controlOut.append(controlSignal)
        return controlSignal

    def evalModel(self):
        # Model Evaluation

        print('Prediction evaluation metrics for each mode:')
        for i in range(self.__Nc):
            print('Mode ' + str(i) + ':')
            ym = [self.__out_mean[j] for j in range(len(self.__out_pred_data)) if self.__sysModes[j] == i]
            ysi = [self.__out_si[j] for j in range(len(self.__out_pred_data)) if self.__sysModes[j] == i]
            yreal = [self.__out_pred_data[j] for j in range(len(self.__out_pred_data)) if self.__sysModes[j] == i]
            if len(ym) > 0:
                self.__cal_Error_Metrics(ym, ysi, yreal)

        # Evaluate the forecast prediction
        print('Forecasting evaluation metrics:')
        return self.__cal_Error_Metrics(self.__In_mean, self.__In_si, self.__In_pred_data)

    def __cal_Error_Metrics(self, y_m, y_si, y_real):
        MSE = 0
        MRSE = 0
        LD = 0
        n = len(y_real)
        Error = [y_m[i] - y_real[i] for i in range(n)]

        # Calculate Mean Square Error
        MSE = 1.0 / n * sum(np.power(Error, 2).tolist())

        # Calculate Mean Root Square Error
        MRSE = 1.0 * sqrt(sum(np.power(Error, 2).tolist()) / sum(np.power(y_real, 2).tolist()))

        # Calculate Log predictive density error
        LD = 0.5 * log(2 * pi) + 1 / (2 * n) * \
                                 sum(np.log(np.power(y_si[:n], 2)).tolist() + np.divide(np.power(Error, 2),
                                                                                        np.power(y_si[:n], 2)).tolist())

        print('MSE: ' + str(MSE))
        print('MRSE: ' + str(MRSE))
        print('LD: ' + str(LD))

        logging.info('MSE: ' + str(MSE))
        logging.info('MRSE: ' + str(MRSE))
        logging.info('LD: ' + str(LD))
        return MSE, MRSE, LD

    def plot_results(self):
        # Draw the Results
        n = len(self.__out_pred_data)
        colorArray = ['g', 'y', 'r']
        colors = [colorArray[i] for i in self.__sysModes]
        sigma = np.sqrt(self.__out_si[:n])

        lowerBound = np.array(self.__out_mean[:n]) - 2 * sigma
        upperBound = np.array(self.__out_mean[:n]) + 2 * sigma
        for i in range(len(lowerBound)):
            if lowerBound[i] < 0:
                lowerBound[i] = 0

        sigma = np.sqrt(self.__In_si[:n])
        lowerBoundIn = np.array(self.__In_mean[:n]) - 2 * sigma
        upperBoundIn = np.array(self.__In_mean[:n]) + 2 * sigma
        for i in range(len(lowerBoundIn)):
            if lowerBoundIn[i] < 0:
                lowerBoundIn[i] = 0
        # Create a subplot with 3 rows and 1 column
        fig, (ax1, ax2, ax3) = pl.subplots(3, 1)

        # time

        x = range(n)
        #######  PLOT latency prediction (output)
        ax1.plot(self.__out_pred_data, 'b-', markersize=5, label=u'Observations')
        ax1.plot(self.__out_mean[:n], 'r--', label=u'Prediction')
        ax1.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([lowerBound, upperBound[::-1]]),
                 #       alpha=.5, fc='b', ec='b', label='95% confidence interval')
                 alpha=.75, fc='w', ec='k', label='95% confidence interval')
        ax1.set_xlabel('$time$')
        ax1.set_ylabel('$Latency$')
        ax1.legend(loc='upper right')
        ax1.set_xlim(0, n + 1)
        ax1.grid(True)
        #### PLOT Request Rate and its predictions
        ax2.plot(self.__In_pred_data, 'b-', markersize=5, label=u'Observations')
        ax2.plot(self.__In_mean[:n], 'r--', label=u'Prediction')
        ax2.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([lowerBoundIn, upperBoundIn[::-1]]),
                 #       alpha=.5, fc='b', ec='b', label='95% confidence interval')
                 alpha=.75, fc='w', ec='k', label='95% confidence interval')

        ax2.set_xlabel('$time$')
        ax2.set_ylabel('$Request rate$')
        ax2.set_xlim(0, n + 1)
        ax2.legend(loc='upper right')
        ax2.grid(True)
        #### PLOT Cluster (mode) prediction
        ax3.scatter(x, [1] * n, marker='.', s=30, lw=0, alpha=1, c=colors[:n])
        ax3.set_xlim(0, n + 1)
        ax3.set_xlabel('$time$')
        ax3.set_ylabel('$cluster$')
        ax3.grid(True)
        # ax3.legend([1]*n,colors,loc='upper right')

        pl.show()

    def export_results(self):
        # Draw the Results
        n = len(self.__out_pred_data)
        colorArray = ['g', 'y', 'r']
        colors = [colorArray[i] for i in self.__sysModes]
        sigma = np.sqrt(self.__out_si[:n])

        lowerBound = np.array(self.__out_mean[:n]) - 2 * sigma
        upperBound = np.array(self.__out_mean[:n]) + 2 * sigma
        for i in range(len(lowerBound)):
            if lowerBound[i] < 0:
                lowerBound[i] = 0

        sigma = np.sqrt(self.__In_si[:n])
        lowerBoundIn = np.array(self.__In_mean[:n]) - 2 * sigma
        upperBoundIn = np.array(self.__In_mean[:n]) + 2 * sigma
        for i in range(len(lowerBoundIn)):
            if lowerBoundIn[i] < 0:
                lowerBoundIn[i] = 0

        # time
        l1 = self.__out_mean[:n]
        l2 = self.__out_pred_data[:n]
        l3 = lowerBound
        l4 = upperBound

        l5 = self.__In_mean[:n]
        l6 = self.__In_pred_data
        l7 = lowerBoundIn
        l8 = upperBoundIn

        l9 = colors[:n]

        l10 = self.__controlOut[:n]
        x = range(n)
        rows = zip(x, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10)
        with open('output.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'Y', 'Ym', 'Yl', 'Yu', 'U', 'Um', 'Ul', 'Uu', 'Mode', 'Control'])
            writer.writerows(rows)
