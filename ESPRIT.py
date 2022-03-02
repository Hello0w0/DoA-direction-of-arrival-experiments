import numpy as np
import scipy.linalg as la
import scipy.signal as ss

class ESPRIT(object):
    
    def __init__(self, array_parameters) -> None:
        self.M = array_parameters["number_of_array_element"]        # 阵元个数
        self.ed = array_parameters["element_distance"]              # 阵元间距   
        self.L = array_parameters["number_of_signal"]               # 信源数目
        self.N = array_parameters["number_of_snapshots"]            # 快拍数

    def run(self, Xs):
        DoAs = []
        # 对每个子阵列分别计算
        for X in Xs:
            Rxx = X @ X.conj().T / self.N
            # 特征值分解
            evalues, evectors = la.eig(Rxx)
            # 按照特征值大小排序特征向量
            idx = evalues.argsort()[::-1]
            evectors = evectors[:,idx]
            S = evectors[:, :self.L]
            # 将原阵列分为两个子阵列： [0,1,...,M-2] 与 [1,2,...,M-1]
            E0 = S[:self.M-1]
            E1 = S[1:]
            E01 = np.hstack((E0, E1))
            R_E = E01.conj().T @ E01
            evalues, evectors = la.eig(R_E)
            idx = evalues.argsort()
            evectors = evectors[:,idx]
            F = evectors[:,:self.L]
            Phi = -F[:self.L,:] @ la.pinv(F[self.L:,:])

            evalues, _ = la.eig(Phi)
            DoA = -np.arcsin(np.angle(evalues)/np.pi) / np.pi * 180
            DoAs.append(np.sort(DoA))
        
        DoAs = np.around(np.average(np.array(DoAs), axis=0))

        return DoAs



