import numpy as np
import scipy.linalg as la
import scipy.signal as ss

class RARE(object):
    
    def __init__(self, array_parameters) -> None:
        self.M = array_parameters["number_of_array_element"]        # 阵元个数
        self.ed = array_parameters["element_distance"]              # 阵元间距   
        self.L = array_parameters["number_of_signal"]               # 信源数目
        self.N = array_parameters["number_of_snapshots"]            # 快拍数
    
    def run(self, Xs, searchrange = np.arange(-90, 90)):
        # 得到阵列信息
        d = np.arange(0, (self.M)*self.ed, self.ed)
        d = np.expand_dims(d, axis=0)

        Rxx = np.zeros([self.M, self.M], dtype=np.complex128)
        # 对每个子阵列分别计算
        for X in Xs:
            Rxx += X @ X.conj().T / self.N
        

        # 特征值分解
        evalues, evectors = la.eig(Rxx)
        idx = evalues.argsort()[::-1]
        evalues = evalues[idx]
        evectors = evectors[:,idx]

        # 取矩阵的第M+1到N列组成噪声子空间
        En = evectors[:, self.L+1:]

        def Q(z):
            zs = [z ** i for i in range(self.M)]
            return np.diag(zs)
        
        T = np.ones([8,1])

        PRARE = []
        for angle in searchrange:
            phim = angle * np.pi / 180
            z = np.exp(-1j*2*np.pi * self.ed *np.sin(phim))
            PRARE.append(1/np.linalg.det(T.T@Q(1/z)@En@En.conj().T@Q(z)@T))


        PRARE = np.array(PRARE).squeeze()
        PRARE = abs(PRARE)
        PRARE = 10 * np.log10(PRARE / max(PRARE))            # 归一化处理
        try:
            DoA_idxs = self.find_DoA(PRARE)
            return searchrange[DoA_idxs], PRARE
        except:
            return np.NaN, PRARE


    def find_DoA(self, PRARE, threshold = -5):
        DoA_idxs, properties = ss.find_peaks(PRARE, height=threshold, distance=1)

        if DoA_idxs.shape[0] > self.L:
            biggest = properties["peak_heights"].argsort()[::-1]
            DoA_idxs = DoA_idxs[biggest[:self.L]]
        if DoA_idxs.shape[0] < self.L:
            DoA_idxs = self.find_DoA(PRARE, threshold = threshold-5)
        
        return np.sort(DoA_idxs)
    