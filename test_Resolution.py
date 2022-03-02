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
    "number_of_signal" : 2              #信源个数
}

signal_parameters = {
    "number_of_signal" : 2,             #信源个数
    "signal_degree" : np.array([0]),    #到达角
    "SNR" : 0,                          #信噪比
}

# 遍历SNR与快拍数
SNRs = np.array([-10, -5, 0, 5, 10, 20])
Ns = np.array([100, 200, 500, 1000, 2000, 5000])


# 初始化存储容器
RE_ESPRIT = np.zeros([SNRs.shape[0], Ns.shape[0]])
RE_MUSIC = np.zeros([SNRs.shape[0], Ns.shape[0]])
RE_RARE = np.zeros([SNRs.shape[0], Ns.shape[0]])

def Correct(DoAs, real_angle):
    DoA_number = 0
    for DoA in DoAs:
        if (DoA > real_angle - 5) and (DoA < real_angle + 10):
            DoA_number += 1
    if DoA_number == 2:
        return 1
    else:
        return 0
    

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
        Correct_ESPRIT = 0
        Correct_MUSIC = 0
        Correct_RARE = 0
        for k in range(100):
            # 随机生成到达角
            degree = np.floor(np.random.uniform(-89,85))
            signal_parameters["signal_degree"] = np.array([degree, degree + 5])
            # 定义信号
            signal_generator = SignalGenerator(array_parameters=array_parameters, signal_parameters=signal_parameters)
            # 生成信号
            Xs = signal_generator.generate()
            # 使用模型处理信号
            DoAs_MUSIC = model_MUSIC.run_DoA(Xs=Xs, searchrange=searchrange)
            DoAs_ESPRIT = model_ESPRIT.run(Xs=Xs)
            DoAs_RARE, _ = model_RARE.run(Xs=Xs, searchrange=searchrange)
            # 评估结果
            Correct_ESPRIT += Correct(DoAs_ESPRIT, degree) / 100
            Correct_MUSIC += Correct(DoAs_MUSIC, degree) / 100
            Correct_RARE += Correct(DoAs_RARE, degree) / 100
        RE_ESPRIT[i,j] = Correct_ESPRIT
        RE_MUSIC[i,j] = Correct_MUSIC
        RE_RARE[i,j] = Correct_RARE

np.save("RE_ESPRIT", RE_ESPRIT)
np.save("RE_MUSIC", RE_MUSIC)
np.save("RE_RARE", RE_RARE)



