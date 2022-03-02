import numpy as np

class SignalGenerator(object):
    def __init__(self, array_parameters, signal_parameters) -> None:
        self.L = array_parameters["number_of_array"]                # 阵列个数
        self.ads = array_parameters["array_distance"]   # 阵列间距
        self.M = array_parameters["number_of_array_element"]        # 阵元个数
        self.ed = array_parameters["element_distance"]              # 阵元间距
        self.N = array_parameters["number_of_snapshots"]            # 快拍数(采样点数)

        self.S = signal_parameters["number_of_signal"]              # 信源数目
        self.D = np.expand_dims(signal_parameters["signal_degree"], axis=0) # 信源角度
        self.SNR = signal_parameters["SNR"]                         # 信噪比
 
    def generate(self):
        S = np.random.randn(self.S, self.N) + np.random.randn(self.S, self.N)*1j    #信源信号强度（高斯白噪声）
        d = np.expand_dims(np.arange(0, (self.M)*self.ed, self.ed), axis=0)

        Xs = []
        # 各个子阵列接收信号
        for ad in self.ads:
            A = np.exp(-1j*2*np.pi * (d + ad).T @ np.sin(self.D * np.pi / 180))       # 子阵列方向矢量
            X = A @ S                                                                 # 子阵列接收信号(加入高斯白噪声)
            X = X  + (np.random.randn(X.shape[0], X.shape[1]) + np.random.randn(X.shape[0], X.shape[1])*1j) / np.sqrt(10**(self.SNR/10.0))
            Xs.append(X)
        return Xs
