import numpy as np 


class KalmanFilter(object):
    def __init__(self, dt, ax, ay, stdacc, xstdmeas, ystdmeas):
        # dt: sampling time (1 cycle)
        # ax: acceleration in x 
        # ay: acceleration in y
        # stdacc: process noise
        # xstdmeas: standard deviation of x
        # ystdmeas: standard deviation of y

        #Sampling time & input variables
        self.dt = dt
        self.u = np.matrix([[ax],[ay]])

        #Initial state
        self.x = np.matrix([[0], [0], [0], [0]])

        #Define the state transition Matrix A
        self.A = np.matrix([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        #Define control Matrix B
        self.B = np.matrix([[(self.dt**2)/2, 0],
                            [0, (self.dt**2)/2],
                            [self.dt, 0],
                            [0, self.dt]])
        #Define measurement matrix
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])
        
        #Noise, must be computed diferently PROBABILISTIC PROGRAMMING
        self.Q = np.matrix([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                            [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                            [(self.dt**3)/2, 0, (self.dt**2), 0],
                            [0, (self.dt**3)/2, 0, (self.dt**2)]]) * stdacc**2
        #Computing variance
        #self.Q = np.matrix([[(np.var(self.x[0][0], self.x[0][0])), 0, np.dot(np.var(self.x[0][0]), np.var(self.x[0][2])), 0],
        #                    [0, (np.var(self.x[0][1], self.x[0][1])), 0, np.var(self.x[0][1], self.x[0][3])],
        #                    [np.var(self.x[0][0], self.x[0][2]), 0, (np.var(self.x[0][2], self.x[0][2])), 0],
        #                    [0, (np.dot(np.var(self.x[0][3]), self.x[0][1])), 0, (np.var(self.x[0][3], self.x[0][3]))]])
        self.R = np.matrix([[xstdmeas**2, 0],
                            [0, ystdmeas**2]])
        #Computing new variance
        #self.R = np.matrix([[(np.var(self.x[0][0], self.x[0][0])), 0],
        #                    [0, (np.var(self.x[0][1], self.x[0][1]))]])


        #Initial Covariance Matrix
        self.P = np.eye(self.A.shape[1])

    def predict(self):
        #Update time state xk = Axk-1 + Bak-1 
        #Fcuk noise
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)

        #Recalculate Q
        #self.Q = np.matrix([[(np.var(self.x[0][0], self.x[0][0])), 0, np.dot(np.var(self.x[0][0]), np.var(self.x[0][2])), 0],
        #                    [0, (np.var(self.x[0][1], self.x[0][1])), 0, np.var(self.x[0][1], self.x[0][3])],
        #                    [np.var(self.x[0][0], self.x[0][2]), 0, (np.var(self.x[0][2], self.x[0][2])), 0],
        #                    [0, (np.dot(np.var(self.x[0][3]), self.x[0][1])), 0, (np.var(self.x[0][3], self.x[0][3]))]])
        
        #Calculate error covariance P = A*P*A' + Q
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        
        return self.x[0:2]

    def update(self, z):
        #Calculate R
        #self.R = np.matrix([[(np.var(z[0], z[0])), 0],
        #                    [0, (np.var(z[1], z[1]))]])
               
        #Ajuda escriure S = H*P*H' + R
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R

        #Kalman Gain K = P*H'/S
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))
        I = np.eye(self.H.shape[1])

        #Update error covariance
        self.P = (I - (K * self.H)) * self.P
        return self.x[0:2]
    


