import numpy as np
import scipy.linalg as la
import scipy.signal as ss

class MUSIC(object):
    
    def __init__(self, array_parameters) -> None:
        self.M = array_parameters["number_of_array_element"]        # 阵元个数
        self.ed = array_parameters["element_distance"]              # 阵元间距   
        self.L = array_parameters["number_of_signal"]               # 信源数目
        self.N = array_parameters["number_of_snapshots"]            # 快拍数


    def run(self, Xs, searchrange = np.arange(-90, 90)):
        # 得到阵列信息
        d = np.arange(0, (self.M)*self.ed, self.ed)
        d = np.expand_dims(d, axis=0)

        Pmusics = []

        # 对每个子阵列分别计算
        for X in Xs:
            # 计算协方差矩阵
            Rxx = X @ X.conj().T / self.N
            # 特征值分解
            evalues, evectors = la.eig(Rxx)
            # 按照特征值大小排序特征向量
            idx = evalues.argsort()[::-1]
            evectors = evectors[:,idx]
            # 取第L+1到M的特征向量组成噪声子空间
            En = evectors[:, self.L+1:]
    
            # 遍历每个角度，计算空间谱
            Pmusic = []
            for angle in searchrange:
                phim = angle * np.pi / 180
                a = np.exp(-1j*2*np.pi * d *np.sin(phim)).T              
                Pmusic.append(1/(a.conj().T@En@En.conj().T@a))

            Pmusic = np.array(Pmusic).squeeze()
            Pmusic = abs(Pmusic)
            Pmusics.append(Pmusic)
        
        # 空间谱取平均
        Pmusic_mean = np.average(np.array(Pmusics), axis = 0)
        Pmusic_mean = 10 * np.log10(Pmusic_mean / max(Pmusic_mean))            # 归一化处理
        try:
            DoA_idxs = self.find_DoA(Pmusic_mean)
            return searchrange[DoA_idxs], Pmusic_mean
        except:
            return np.NaN, Pmusic_mean

    def run_DoA(self, Xs, searchrange = np.arange(-90, 90)):
        # 得到阵列信息
        d = np.arange(0, (self.M)*self.ed, self.ed)
        d = np.expand_dims(d, axis=0)

        DoA_idxss = []

        # 对每个子阵列分别计算
        for X in Xs:
            # 计算协方差矩阵
            Rxx = X @ X.conj().T / self.N
            # 特征值分解
            evalues, evectors = la.eig(Rxx)
            # 按照特征值大小排序特征向量
            idx = evalues.argsort()[::-1]
            evectors = evectors[:,idx]
            # 取第L+1到M的特征向量组成噪声子空间
            En = evectors[:, self.L+1:]
    
            # 遍历每个角度，计算空间谱
            Pmusic = []
            for angle in searchrange:
                phim = angle * np.pi / 180
                a = np.exp(-1j*2*np.pi * d *np.sin(phim)).T              
                Pmusic.append(1/(a.conj().T@En@En.conj().T@a))

            Pmusic = np.array(Pmusic).squeeze()
            Pmusic = abs(Pmusic)
        
            try:
                DoA_idxs = self.find_DoA(Pmusic)
            except:
                DoA_idxs = np.NaN
            DoA_idxss.append(DoA_idxs)

        # DoA取平均
        DoA_idxs_mean = np.around(np.average(np.array(DoA_idxss), axis = 0)).astype(int)
        return searchrange[DoA_idxs_mean]
        

    def run_R(self, Xs, searchrange = np.arange(-90, 90)):
        # 得到阵列信息
        d = np.arange(0, (self.M)*self.ed, self.ed)
        d = np.expand_dims(d, axis=0)

        Rxx = np.zeros([self.M, self.M], dtype=np.complex128)
        # 对每个子阵列分别计算
        for X in Xs:
            Rxx += X @ X.conj().T / self.N

        # 特征值分解
        evalues, evectors = la.eig(Rxx)
        # 按照特征值大小排序特征向量
        idx = evalues.argsort()[::-1]
        evectors = evectors[:,idx]
        # 取第L+1到M的特征向量组成噪声子空间
        En = evectors[:, self.L+1:]
    
        # 遍历每个角度，计算空间谱
        Pmusic = []
        for angle in searchrange:
            phim = angle * np.pi / 180
            a = np.exp(-1j*2*np.pi * d *np.sin(phim)).T              
            Pmusic.append(1/(a.conj().T@En@En.conj().T@a))

        Pmusic = np.array(Pmusic).squeeze()
        Pmusic = abs(Pmusic)

        Pmusic = 10 * np.log10(Pmusic / max(Pmusic))            # 归一化处理
        try:
            DoA_idxs = self.find_DoA(Pmusic)
            return searchrange[DoA_idxs], Pmusic
        except:
            return np.NaN, Pmusic
    
    def find_DoA(self, Pmusic, threshold = -5):
        DoA_idxs, properties = ss.find_peaks(Pmusic, height=threshold, distance=1)

        if DoA_idxs.shape[0] > self.L:
            biggest = properties["peak_heights"].argsort()[::-1]
            DoA_idxs = DoA_idxs[biggest[:self.L]]
        if DoA_idxs.shape[0] < self.L:
            DoA_idxs = self.find_DoA(Pmusic, threshold = threshold-5)
        
        return np.sort(DoA_idxs)


