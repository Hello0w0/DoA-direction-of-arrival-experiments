import numpy as np
from SignalGenerator import SignalGenerator
from MUSIC import MUSIC
from ESPRIT import ESPRIT
from RARE import RARE

array_parameters = {
    "number_of_array" : 3,              #阵列个数
    "array_distance" : [0, 5, 10],      #阵列之间距离参数
    "number_of_array_element" : 8,      #阵元个数 
    "element_distance" : 0.5,           #阵元间距(倍波长)
    "number_of_snapshots" : 1024,       #快拍数
    "number_of_signal" : 1              #信源个数
}

signal_parameters = {
    "number_of_signal" : 1,             #信源个数
    "signal_degree" : np.array([0]),    #到达角
    "SNR" : 0,                          #信噪比
}

# 遍历SNR与快拍数
SNRs = np.array([-15,-13, -11, -9, -7, -5, 0, 5])
Ns = np.array([100, 200, 500, 1000, 2000, 5000])


# 初始化存储容器
Errors_ESPRIT = np.zeros([SNRs.shape[0], Ns.shape[0]])
Errors_MUSIC = np.zeros([SNRs.shape[0], Ns.shape[0]])
Errors_RARE = np.zeros([SNRs.shape[0], Ns.shape[0]])

for i in range(SNRs.shape[0]):
    signal_parameters["SNR"] = SNRs[i]
    for j in range(Ns.shape[0]):
        array_parameters["number_of_snapshots"] = Ns[j]

        # 选择使用模型
        model_MUSIC = MUSIC(array_parameters = array_parameters)
        model_ESPRIT = ESPRIT(array_parameters = array_parameters)
        model_RARE = RARE(array_parameters = array_parameters)

        # 定义搜索范围
        searchrange=np.arange(-90, 90)

        # 重复100次取平均值
        Error_ESPRIT = 0
        Error_MUSIC = 0
        Error_RARE = 0
        for k in range(1000):
            # 随机生成到达角
            signal_parameters["signal_degree"] = np.floor(np.random.uniform(-89,90,[1,]))
            # 定义信号
            signal_generator = SignalGenerator(array_parameters=array_parameters, signal_parameters=signal_parameters)
            # 生成信号
            Xs = signal_generator.generate()
            # 使用模型处理信号
            DoAs_MUSIC = model_MUSIC.run_DoA(Xs=Xs, searchrange=searchrange)
            DoAs_ESPRIT = model_ESPRIT.run(Xs=Xs)
            DoAs_RARE, _ = model_RARE.run(Xs=Xs, searchrange=searchrange)
            # 评估结果
            Error_ESPRIT += np.nanmean((signal_parameters["signal_degree"] - DoAs_ESPRIT) ** 2) / 1000
            Error_MUSIC += np.nanmean((signal_parameters["signal_degree"] - DoAs_MUSIC) ** 2) / 1000
            Error_RARE += np.nanmean((signal_parameters["signal_degree"] - DoAs_RARE) ** 2) / 1000
        Errors_ESPRIT[i,j] = Error_ESPRIT
        Errors_MUSIC[i,j] = Error_MUSIC
        Errors_RARE[i,j] = Error_RARE

np.save("E_ESPRIT", Errors_ESPRIT)
np.save("E_MUSIC", Errors_MUSIC)
np.save("E_RARE", Errors_RARE)



