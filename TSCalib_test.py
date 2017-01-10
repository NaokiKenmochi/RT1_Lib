import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

# 定数の設定
calib_settings = {
    'ntct': 100,  # 温度計算の分割数
    'te': np.exp(0.1 * np.arange(100)),  # 計算温度範囲[eV] ntctと同数
    'nrat': 1000,  # ??の分割数
    'll': 5,  # フィルタ数
    'maxch': 25,  # 空間チャンネル数
    'maxword': 440, #モノクロの較正波長数
    'nfil': 5,  # フィルタ数
    'maxm': 10,
    'maxlaser': 2,  # 最大レーザー台数
    'maxfl': 10,
    'nlaser': 2,  # 仕様レーザー数
    'm': 2,
    'TT': 297.15,  # 較正時のガス温度
    'maxdata': 160,  # 最大取り込みチャンネル数
    'inj_angle': np.pi / 9,  # 入射角度[rad]
    'init_wlength': 685,  # モノクロメータの初期波長[nm]
    'worder': np.array([  # V792のデータ順を並び替える配列
        0, 2, 4, 6, 8, 10,
        12, 14, 16, 18, 20, 22,
        24, 26, 28, 30, 1, 3,
        5, 7, 9, 11, 13, 15,
        17, 19, 21, 23, 25, 27,
        32, 34, 36, 38, 40, 42,
        44, 46, 48, 50, 52, 54,
        56, 58, 60, 62, 33, 35,
        37, 39, 41, 43, 45, 47,
        49, 51, 53, 55, 57, 59,
        64, 66, 68, 70, 72, 74,
        76, 78, 80, 82, 84, 86,
        88, 90, 92, 94, 65, 67,
        69, 71, 73, 75, 77, 79,
        81, 83, 85, 87, 89, 91,
        96, 98, 100, 102, 104, 106,
        108, 110, 112, 114, 116, 118,
        120, 122, 124, 99, 97, 126,
        101, 103, 105, 107, 109, 111,
        113, 115, 117, 119, 121, 123,
        128, 130, 132, 134, 136, 138,
        140, 142, 144, 146, 148, 150,
        152, 154, 156, 158, 129, 131,
        133, 135, 137, 139, 141, 143,
        145, 147, 149, 151, 153, 155,
        29, 31, 61, 63, 93, 95, 125, 127, 157, 159]),
    'int_range': np.array([  # 各チャンネルの積分範囲、フィルターの透過波長領域に対応
        360, 380,   #ch1
        250, 350,   #ch2
        10, 170,    #ch3
        330, 370,   #ch4
        140, 290,   #ch5
        10, 400])   #ch6
}


class TSCalib:
    """トムソン較正用クラス
    """

    def __init__(self, **kwargs):
        """較正のための初期値設定
        """

        # 各諸元設定
        self.ntct = kwargs['ntct']
        self.te = kwargs['te']
        self.nrat = kwargs['nrat']
        self.ll = kwargs['ll']
        self.maxch = kwargs['maxch']
        self.maxword = kwargs['maxword']
        self.nfil = kwargs['nfil']
        self.maxm = kwargs['maxm']
        self.maxlaser = kwargs['maxlaser']
        self.maxfl = kwargs['maxfl']
        self.nlaser = kwargs['nlaser']
        self.m = kwargs['m']
        self.tt = kwargs['TT']
        self.maxdata = kwargs['maxdata']
        self.inj_angle = kwargs['inj_angle']
        self.worder = kwargs['worder']
        self.int_range = kwargs['int_range']
        self.init_wlength = kwargs['init_wlength']

    def sort_rawdata(self, st_raw, raw, worder):
        """V792のデータ順序を読み取りやすいように入れ替え
        """
        for i in range(self.maxdata):
            st_raw[:, i] = raw[:, worder[i]]

    def load_calib_values(self):
        """ポリクロ波長較正データの読み出し
        """
        file_path = '/Users/kemmochi/SkyDrive/Document/Study/Thomson/DATE/Polychrometer/Data/data of polychrometors for calibration/2016/'
        st_rwdata = np.zeros((self.maxword, self.maxdata))
        clbdata = np.zeros((self.maxword, self.maxdata))

        for j in range(self.maxch):
            file_name = "Th_Raw_HJ" + str(j + 1) + ".txt"
            rwdata = np.loadtxt(file_path + file_name, comments='#')
            self.sort_rawdata(st_rwdata, rwdata, self.worder)
            clbdata[:, (self.nfil + 1) * j:(self.nfil + 1) * (j + 1) - 1] = st_rwdata[:,
                                                                            (self.nfil + 1) * j:(self.nfil + 1) * (
                                                                            j + 1) - 1]
        #            clbdata[:, 6*j:6*(j+1)-1] = st_rwdata[:, 6*j:6*(j+1)-1]

        return clbdata

    def clb_wo_offset(self):
        """較正データのオフセットを差し引く

        返り値: オフセットを差し引いたポリクロの波長感度
        """
        clbdata = self.load_calib_values()
        for i in range(self.maxdata):
            clbdata[:, i] -= np.average(clbdata[380:439, i])
        #        clbdata -= np.average(clbdata[380:439,:])
        return clbdata

    def cnt2vol(self):
        """取り込み値を電圧値に変換

        返り値: モノクロの光量で較正したポリクロの波長感度
        """
        light_power = np.loadtxt(
            '/Users/kemmochi/SkyDrive/Document/Study/Thomson/DATE/Polychrometer/Data/data of polychrometors for calibration/2016/w_for_alignment_Aug2016.txt')
        #        plt.plot(light_power)
        clbdata = self.clb_wo_offset()
        light_power = light_power[np.newaxis, :]
        clbdata = clbdata / light_power.T
        #        plt.plot(clbdata[:,1])
        #        np.savetxt('/Users/kemmochi/SkyDrive/Document/Study/Thomson/DATE/Polychrometer/Data/data of polychrometors for calibration/2016/clbdata.txt', clbdata, delimiter=',')
        np.save(
            '/Users/kemmochi/SkyDrive/Document/Study/Thomson/DATE/Polychrometer/Data/data of polychrometors for calibration/2016/clbdata.npy',
            clbdata)
        #        np.savez_compressed('/Users/kemmochi/SkyDrive/Document/Study/Thomson/DATE/Polychrometer/Data/data of polychrometors for calibration/2016/clbdata.npz', clbdata)

        return clbdata

    def thomson_shape(self):
        """トムソン散乱断面積を計算

        返り値:
            w1: 各温度におけるトムソン散乱断面積のスペクトル(横440 x 縦160)
        """
        w1 = np.zeros((self.maxword, self.ntct))
        IS = np.cos(self.inj_angle)

        wlength = np.arange(self.maxword) + self.init_wlength  # 較正波長領域[nm]

        w2 = 1064 / wlength
        x = 51193 / self.te
        x = x[np.newaxis, :]

        #        K2 = ((x / (2 * 3.14159265358979)) ** 0.5) * np.exp(x) * (1 + (15 / (8 * x)))
        q = (1 - (1 - IS) / x.T) ** 2
        #        A = (1 - 2 * w2 * IS + w2 ** 2) ** (-0.5)
        #        B = (1 + ((w2 - 1) ** 2) / (2 * w2 * (1 - IS))) ** 0.5

        w1 = np.log(q) + 0.5 * np.log((x.T / (2 * np.pi))) + x.T + np.log(1 + (15 / (8 * x.T))) + 2 * np.log(
            w2) - x.T * np.sqrt(1 + ((w2 - 1) ** 2) / (2 * w2 * (1 - IS))) - 0.5 * np.log(1 - 2 * w2 * IS + w2 ** 2)
        w1 = np.exp(w1)

        #        plt.plot(w1[80,:])
        #        plt.contourf(np.log(w1+1))
        return w1

    def load_clbdata(self):
        """cnv2volで作成した較正データを読み取り

        返り値: int_clbdata(縦160 x 横100）
        """

        #        clbdata = np.loadtxt('/Users/kemmochi/SkyDrive/Document/Study/Thomson/DATE/Polychrometer/Data/data of polychrometors for calibration/2016/clbdata.txt', delimiter=',')
        clbdata = np.load(
            '/Users/kemmochi/SkyDrive/Document/Study/Thomson/DATE/Polychrometer/Data/data of polychrometors for calibration/2016/clbdata.npy')

        return clbdata

    def cnt_photon_ltdscp(self):
        """各温度にポリクロの各チャンネルに入ってくるフォトン数を計算
        """
        #        clbdata = self.cnt2vol()
        clbdata = self.load_clbdata()
        thomson_shape = self.thomson_shape()

        num_ratio = (self.nfil + 1) * self.nfil / 2  # チャンネルの信号比の組み合わせ数

        int_clbdata = np.zeros((self.maxdata, self.ntct))
        relte = np.zeros((num_ratio * self.maxch, self.ntct))

        ll = 0

        for i in range(self.ntct):
            for j in range(self.nfil + 1):
                for k in range(self.maxch):
                    buff = clbdata[:, (self.nfil + 1) * k + j] * thomson_shape[i, :]
                    int_clbdata[(self.nfil + 1) * k + j, i] = integrate.trapz(
                        buff[self.int_range[2 * j]:self.int_range[2 * j + 1]])

            for ii in range(self.nfil + 1):
                for jj in range(self.nfil - ii):
                    for kk in range(self.maxch):
                        if np.abs(int_clbdata[(self.nfil + 1) * kk + (ii + jj + 1), i]) < 1e-35:
                            relte[num_ratio * kk + ll, i] = np.nan
                        else:
                            relte[num_ratio * kk + ll, i] = int_clbdata[(self.nfil + 1) * kk + ii, i] / int_clbdata[
                                (self.nfil + 1) * kk + (ii + jj + 1), i]
                    ll += 1
            ll = 0

        #        return int_clbdata
        plt.ylim(0, 1e3)
        plt.plot(relte[75, :])
        plt.plot(relte[76, :])
        plt.plot(relte[77, :])
        plt.plot(relte[78, :])
        plt.show()
        return relte

    def differ(self, data, n, dx, ddata):
        """差分を計算
        """
        ddata[0] = (data[1] - data[0]) / dx

        for i in range(1, n-1):
            ddata[i] = (data[i+1] - data[i-1])/(2*dx)

        ddata[n-1] = (data[n-1] - data[n-2]) / dx

    def cal_ratio(self):
        """較正係数の計算
        """


if __name__ == "__main__":
    test = TSCalib(**calib_settings)
    relte = test.cnt_photon_ltdscp()

    print("Successfully Run the program !!!")
