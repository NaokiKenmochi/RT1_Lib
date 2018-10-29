import read_wvf
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import os
import time


class DataBrowser:
    def __init__(self, date, shotNo, LOCALorPPL, isShotLog='False'):
        """

        :param date:
        :param shotNo:
        """
        self.date = date
        self.shotnum = shotNo
        self.LOCALorPPL = LOCALorPPL
        self.isShotLog = isShotLog
        ##干渉計の補正値
        #for IF1
        if(np.int(date)<20171222):
            self.a1 = 0.00
            self.b1 = 0.29
        elif(np.int(date) >= 20171222 and np.int(date)<20171223):
            self.a1 = 0.004
            self.b1 = 0.30
        elif(np.int(date) >= 20171223 and np.int(date)<20180222):
            self.a1 = 0.002
            self.b1 = 0.136
        elif(np.int(date) >= 20180222):
            self.a1 = 0.002
            self.b1 = 0.30

        #for IF2
        if(np.int(date)>=20150719 and np.int(date)<20171012):
        #19 July 2015
            self.a2 = -0.005
            self.b2 = 0.135
        elif(np.int(date) >= 20171012 and np.int(date) < 20171111):
        #12 Oct 2017
            self.a2 = -0.0035
            self.b2 = 0.0315
        elif(np.int(date) >= 20171111):
        #11 Nov 2017
            self.a2 = -0.001
            self.b2 = 0.145

        #for IF3
        self.a3 = 0.00
        if(np.int(date)>=20160718 and np.int(date)<20171012):
            self.b3 = 0.30     #18 July 2016
        elif(np.int(date) >= 20171012 and np.int(date)<20180222):
            self.b3 = 0.28      #12 Oct 2017
        elif(np.int(date) >= 20180222):
            self.b3 = 0.29

        #グラフ描写のstep数
        self.num_step = 20


        self.data_pos_name_ep01 = np.array([[7, "time", ""],
                                            [7, "8GPf", ""],
                                            [7, "8GPr", ""],
                                            [7, "2GPf", ""],
                                            [7, "2GPr", ""],
                                            [11, "Pgas", ""],
                                            [7, "8GPf_n", ""],
                                            [7, "8GPr_n", "$\mathbf{P_{ECH} [kW]}$"],
                                            [11,"SX1", ""],
                                            [9, "VG", ""],
                                            [3, "IF", ""],
                                            [3, "IF2", ""],
                                            [3, "IF3", "$\mathbf{n_eL [10^{17}m^{-2}]}$"],
                                            [5, "Pol390nm", ""],  #wall3
                                            [11,"Id", ""],
                                            [11,"Ia", ""],
                                            [11,"Ib", ""],
                                            [1, "ml1", ""],
                                            [1, "ml2", ""],
                                            [1, "ml3", ""],
                                            [1, "ml4", ""],
                                            [1, "ml5", ""],
                                            [12, "Pfwd", ""],
                                            [12, "Prev", ""],
                                            [11,"mpz", ""],
                                            [5, "Pol730nm", ""],    #trg
                                            [5, "Pol710nm", ""],     #PD
                                            [5, "Pol450nm", ""],  #1000V
                                            [11, "REFcos", ""],
                                            [11, "REFsin", ""],
                                            [1, "dmlt1", ""],
                                            [1, "mlt2", ""],
                                            [1, "mlt3", "diamag. [mWb]"]])

        self.data_pos_name_ep02 = np.array([[2, "MP1", ""],
                                            [2, "MP2", ""],
                                            [2, "MP3", ""],
                                            [2, "MP4", ""],
                                            [6, "IF_FAST", ""],
                                            [6, "IF2_FAST", ""],
                                            [10, "REFcos_FAST", ""],
                                            [10, "REFsin_FAST", ""]])

    def load_date(self, LOCALorPPL):
        """
        LOCALorPPL == "PPL"の場合:
            exp_ep01, exp_ep01から実験データをロードします

        LOCALorPPL == "LOCAL"の場合:
            ローカルに保存してあるデータを読み込みます

        :param LOCALorPPL:
        :return:
        """
        if LOCALorPPL == "PPL":
            dm_ep01 = read_wvf.DataManager("exp_ep01", 0, self.date)
            dm_ep02_MP = read_wvf.DataManager("exp_ep02", "MP", self.date)
            dm_ep02_SX = read_wvf.DataManager("exp_ep02", "SX", self.date)
            data_ep01 = dm_ep01.fetch_raw_data(self.shotnum)
            data_ep02_MP = dm_ep02_MP.fetch_raw_data(self.shotnum)
            data_ep02_SX = dm_ep02_SX.fetch_raw_data(self.shotnum)
            #np.savez_compressed("data/data_%s_%d" % (self.date, self.shotnum), data_ep01=data_ep01, data_ep02_MP=data_ep02_MP, data_ep02_SX=data_ep02_SX)
            print("Load data from PPL")

        else:
            data = np.load("data/data_%s_%d.npz" % (self.date, self.shotnum))
            data_ep01 = data["data_ep01"]
            data_ep02_MP = data["data_ep02_MP"]
            data_ep02_SX = data["data_ep02_SX"]
            print("Load data from local")

        return data_ep01, data_ep02_MP, data_ep02_SX

    def multiplot(self):
        """
        exp_ep01, exp_ep02に保存してあるRT-1の実験データを全て描写します

        :return:
        """
        data_ep01, data_ep02_MP, data_ep02_SX = self.load_date(self.LOCALorPPL)
        data_ep01 = self.adj_gain(data_ep01)
        data_ep01 = self.mag_loop(data_ep01)
        data_ep01 = self.calib_IF(data_ep01)

        if self.isShotLog is 'True':
            self.make_shotlog(data_ep01)

        fig = plt.figure(figsize=(18,10))
        #############################
        #   Date in exp_ep01        #
        #############################
        for j in range(1,33):
            ax1 = fig.add_subplot(6,2,int(self.data_pos_name_ep01[j,0]), sharex=None, sharey=None)
            ax1.set_xlim(0.5, 2.5)
            ax1.plot(data_ep01[0,::self.num_step],data_ep01[j,::self.num_step], label=self.data_pos_name_ep01[j,1])
            ax1.legend(fontsize=10, ncol=2, loc='best')
            ax1.set_ylabel(self.data_pos_name_ep01[j,2])
            if(j==22 or j==28):
                ax1.set_xlabel("Time [sec]")

        #############################
        #   Date in exp_ep02 (SX)   #
        #############################
        for j in range(1,5):
            ax1 = fig.add_subplot(6,2, int(self.data_pos_name_ep02[j+3,0]), sharex=None, sharey=None)
            ax1.set_xlim(0.5, 2.5)
            ax1.plot(data_ep02_SX[0,::20*self.num_step]+1.25,data_ep02_SX[j,::20*self.num_step]+0.1-0.10*j, label=self.data_pos_name_ep02[j+3, 1])
            plt.subplots_adjust(left=0.05, right=0.97, bottom=0.05, top=0.95, wspace=0.15, hspace=0.15)
            ax1.legend(fontsize=10)

        #############################
        #   Date in exp_ep02 (MP)   #
        #############################
        for j in range(1,5):
            ax1 = fig.add_subplot(6,2, int(self.data_pos_name_ep02[j-1,0]), sharex=None, sharey=None)
            ax1.set_xlim(0.5, 2.5)
            #ax1.plot(data_ep02_MP[0,::20*self.num_step]+0.5,data_ep02_MP[j,::20*self.num_step]+0.2-0.10*j, label=self.data_pos_name_ep02[j-1, 1])
            ax1.plot(data_ep02_MP[0,::self.num_step]+1.25,data_ep02_MP[j,::self.num_step]+0.2-0.10*j, label=self.data_pos_name_ep02[j-1, 1])
            #ax1.plot(time_ep02_MP[::100], data_ep02_MP[j,::100]+0.5-0.25*j, label=self.data_name_ep02[j-1])
            plt.subplots_adjust(left=0.05, right=0.97, bottom=0.05, top=0.95, wspace=0.15, hspace=0.15)
            ax1.legend(fontsize=10)
            if(j == 2):
                plt.title("Date: %s, Shot No.: %d" % (self.date, self.shotnum), loc='right', fontsize=36, fontname="Times New Roman")

        ax1 = fig.add_subplot(6,2,4)
        self.stft(data_ep02_MP[0,:], data_ep02_MP[3,:], self.data_pos_name_ep02[2,1], nperseg=512, vmax=1e-4, time_offset=0.25)
        ax1 = fig.add_subplot(6,2,8)
        self.stft(data_ep02_SX[0,:], data_ep02_SX[2,:], self.data_pos_name_ep02[4,1], nperseg=25000, vmax=8e-4, time_offset=0.75)
        filepath = "figure/"
        filename = "RT1_%s_%d" % (self.date, self.shotnum)
        plt.savefig(filepath + filename)

#        plt.show()

    def mag_loop(self, ml):
        """
        磁気ループの信号を反磁性信号（単位はWb）に較正します

        :param ml:
        :return:
        """
        ml[18,:] = ml[18,:]/2.82
        ml[17,:] = ml[17,:] - ml[18,:]
        ml[30,:] = ml[30,:] - ml[18,:]/3.0
        for j in range(17,22):
            ml[j,:] -= np.mean(ml[j,:6000])
            #ml[j] = [np.abs(1.0e-4*np.sum(ml[j,:i])) for i in range(len(ml[0]))]    #Unit: Wb
            ml[j] = [np.abs(1.0e-1*np.sum(ml[j,:i])) for i in range(len(ml[0]))]    #Unit: mWb
        for j in range(30,33):
            ml[j,:] -= np.mean(ml[j,:6000])
            #ml[j] = [np.abs(1.0e-4*np.sum(ml[j,:i])) for i in range(len(ml[0]))]    #Unit:Wb
            ml[j] = [np.abs(1.0e-4*np.sum(ml[j,:i])) for i in range(len(ml[0]))]    #Unit:mWb
        return ml

    def calib_IF(self, IF):
        """
        マイクロ波干渉計の位相差信号を，密度の値に較正します
        （フリンジジャンプの補正は未実装）

        :param IF:
        :return:
        """
        IF_offset = np.mean(np.arcsin((IF[10,:5000]-self.a1)/self.b1)*180/np.pi)
        #IF[10,:] = np.arcsin((IF[10,:]-self.a1)/self.b1)*180/np.pi - np.mean(np.arcsin((IF[10,:5000]-self.a1)/self.b1)*180/np.pi)
        IF[10,:] = np.arcsin((IF[10,:]-self.a1)/self.b1)*180/np.pi
        IF[11,:] = np.arcsin((IF[11,:]-self.a2)/self.b2)*180/np.pi - np.mean(np.arcsin((IF[11,:5000]-self.a2)/self.b2)*180/np.pi)
        IF[12,:] = np.arcsin((IF[12,:]-self.a3)/self.b3)*180/np.pi - np.mean(np.arcsin((IF[12,:5000]-self.a3)/self.b3)*180/np.pi)

        IF[10, :] = 180 - IF[10, :] - IF_offset

        IF[10,:] = IF[10,:]*5.58/360
        IF[11,:] = IF[11,:]*5.58/360
        IF[12,:] = IF[12,:]*5.58/360

        return IF

    def adj_gain(self, data_ep01):
        """WE7000のゲインを調整"""

        data_ep01[1,:] = 4.24561e-6+(0.00112308 *data_ep01[1,:])+(0.0247089*(data_ep01[1,:])**2)+(0.00316782*(data_ep01[1,:])**3)+(0.000294602*(data_ep01[1,:])**4)
        data_ep01[1,:] *= 1.90546e3
        data_ep01[2,:] = 1.78134e-6+(0.000992047*data_ep01[2,:])+(0.0189206*(data_ep01[2,:])**2)+(0.00316506*(data_ep01[2,:])**3)+(6.71477e-5*(data_ep01[2,:])**4)
        data_ep01[2,:] *= 1.0e2
        data_ep01[3,:] = (data_ep01[3,:])*2.0
        data_ep01[4,:] = (data_ep01[4,:])*2.0
        data_ep01[6,:] = -2.95352e-6+(0.00313776*data_ep01[6,:])+(0.0381345*(data_ep01[6,:])**2)-(0.0110572*(data_ep01[6,:])**3)+(0.00832368*(data_ep01[6,:])**4)
        data_ep01[6,:] *= 2.45471e3
        data_ep01[7,:] = 2.94379e-6+(0.00288251*data_ep01[7,:])+(0.0365269*(data_ep01[7,:])**2)-(0.0137599*(data_ep01[7,:])**3)+(0.00581889*(data_ep01[7,:])**4)
        data_ep01[7,:] *= 9.33254e1
        data_ep01[9,:] = 10**((data_ep01[9,:]-7.75)/0.75+2)
        data_ep01[17,:] = (data_ep01[17,:])/10/3
        data_ep01[18,:] = (data_ep01[18,:])/10/3
        data_ep01[19,:] = (data_ep01[19,:])/10/3
        data_ep01[20,:] = (data_ep01[20,:])/10/3
        data_ep01[21,:] = (data_ep01[21,:])/10/3
        data_ep01[30,:] = (data_ep01[30,:])/10/3
        data_ep01[31,:] = (data_ep01[31,:])/10
        data_ep01[32,:] = (data_ep01[32,:])/10/3

        return data_ep01

    def stft(self, x, y, label, nperseg, vmax, time_offset):
        """
        短時間フーリエ変換を行います
        :param x:
        :param y:
        :param label:
        :param nperseg:
        :param vmax:
        :param time_offset:
        :return:
        """
        MAXFREQ = 1e0
        N = 1e-3*np.abs(1/(x[1]-x[2]))
        f, t, Zxx =sig.spectrogram(y, fs=N, window='hamming', nperseg=nperseg)
        plt.pcolormesh(t*1e-3+time_offset, f, np.abs(Zxx), vmin=0, vmax=vmax)
        #plt.contourf(t, f, np.abs(Zxx), 200, norm=LogNorm())# vmax=1e-7)
        plt.ylabel(label + "\nFrequency [kHz]")
        plt.ylim([0, MAXFREQ])
        plt.xlim([0.5, 2.5])

    def get_max_tmax(self, data_ep01):
        if data_ep01 is None:
            data_ep01, data_ep02_MP, data_ep02_SX = self.load_date(self.LOCALorPPL)
            data_ep01 = self.adj_gain(data_ep01)
            data_ep01 = self.mag_loop(data_ep01)
            data_ep01 = self.calib_IF(data_ep01)

        t_st = 1.1
        t_ed = 1.9
        st_idx = np.abs(np.asarray(data_ep01[0, :]) - t_st).argmin()
        ed_idx = np.abs(np.asarray(data_ep01[0, :]) - t_ed).argmin()

        ml_convolved = np.zeros((data_ep01[0, :].__len__(), 5))
        PECH_convolved = np.zeros((data_ep01[0, :].__len__(), 4))
        VG_convolved = np.zeros((data_ep01[0, :].__len__(), 1))
        IF_convolved = np.zeros((data_ep01[0, :].__len__(), 3))
        IF_max_tmax = np.zeros((3, 2))
        ml_max = np.zeros(5)
        PECH_max = np.zeros(4)
        VG_max = np.zeros(1)

        num_convolve = 1000
        b = np.ones(num_convolve)/num_convolve
        for i in range(3):
            IF_convolved[:, i] = np.convolve(data_ep01[10+i, :], b, mode='same')
        for i in range(5):
            ml_convolved[:, i] = np.convolve(data_ep01[17+i, :], b, mode='same')
        for i in range(4):
            PECH_convolved[:, i] = np.convolve(data_ep01[1+i, :], b, mode='same')
        VG_convolved[:, 0] = np.convolve(data_ep01[9, :], b, mode='same')

        IF_max_tmax[:, 0] = np.max(IF_convolved[st_idx: ed_idx, :], axis=0)
        IF_max = np.max(IF_convolved[st_idx: ed_idx, :], axis=0)
        IF_max_tmax[:, 1] = np.argmax(IF_convolved[st_idx: ed_idx, :], axis=0) + st_idx
        ml_max = np.max(ml_convolved[st_idx: ed_idx, :], axis=0)
        PECH_max = np.max(PECH_convolved[st_idx: ed_idx, :], axis=0)
        VG_max = np.max(VG_convolved[st_idx: ed_idx, :], axis=0)
        #plt.plot(IF_convolved)
        #plt.show()

        return IF_max_tmax, IF_max, ml_max, PECH_max, VG_max

    def fileCheck(self, fpn):
        comment = ''  # ローカル変数を明示
        m = os.path.isfile(fpn)
        if m:  # 真の場合に実行
            comment = 'true'
        else:
            comment = 'false'

        return comment  # 戻り値

    def writeLog(self, fname, myCheck, data_ep01):
        IF_max_tmax, IF_max, ml_max, PECH_max, VG_max = self.get_max_tmax(data_ep01)

        if myCheck == 'false':
            np.savez_compressed(fname, arr_shotnum=self.shotnum, IF_max_tmax=IF_max_tmax, IF_max=IF_max, ml_max=ml_max, PECH_max=PECH_max, VG_max=VG_max)
        elif myCheck == 'true':
            data = np.load(fname)
            arr_shotnum_buf = data["arr_shotnum"]
            IF_max_tmax_buf = data["IF_max_tmax"]
            IF_max_buf = data["IF_max"]
            ml_max_buf = data["ml_max"]
            PECH_max_buf = data["PECH_max"]
            VG_max_buf = data["VG_max"]
            arr_shotnum_buf = np.hstack((arr_shotnum_buf, self.shotnum))
            IF_max_tmax_buf = np.vstack((IF_max_tmax_buf, IF_max_tmax))
            IF_max_buf = np.vstack((IF_max_buf, IF_max))
            ml_max_buf = np.vstack((ml_max_buf, ml_max))
            PECH_max_buf = np.vstack((PECH_max_buf, PECH_max))
            VG_max_buf = np.vstack((VG_max_buf, VG_max))
            #plt.plot(arr_shotnum_buf, PECH_max_buf)
            ##plt.plot(arr_shotnum_buf, PECH_max_buf[:, 1])
            ##plt.plot(arr_shotnum_buf, PECH_max_buf[:, 2])
            ##plt.plot(arr_shotnum_buf, PECH_max_buf[:, 3])
            #plt.show()
            #plt.plot(arr_shotnum_buf, ml_max_buf)
            #plt.show()
            #plt.plot(arr_shotnum_buf, VG_max_buf)
            #plt.show()
            #plt.plot(arr_shotnum_buf, IF_max_buf)
            #plt.show()

            IF_label=np.array(["IF1", "IF2", "IF3"])
            ml_label=np.array(["ml1", "ml2", "ml3", "ml4", "ml5"])
            PECH_label=np.array(["8GPf", "8GPr", "2GPf", "2GPr"])
            plt.figure(figsize=(10, 10))
            plt.subplot(411)
            plt.title("Date: %s" % (self.date), loc='right', fontsize=36, fontname="Times New Roman")
            for i in range(PECH_label.__len__()):
                plt.plot(arr_shotnum_buf, PECH_max_buf[:, i], "o", label=PECH_label[i])
            plt.ylabel(self.data_pos_name_ep01[7, 2] + "(max)")
            plt.legend()
            plt.subplot(412)
            for i in range(ml_label.__len__()):
                plt.plot(arr_shotnum_buf, ml_max_buf[:, i], "o", label=ml_label[i])
            plt.ylabel(self.data_pos_name_ep01[32, 2] + "(max)")
            plt.legend()
            plt.subplot(413)
            plt.plot(arr_shotnum_buf, 1e3*VG_max_buf, "o", label="VG")
            plt.ylabel("VG [mPa](max)")
            plt.legend()
            plt.subplot(414)
            for i in range(IF_label.__len__()):
                plt.plot(arr_shotnum_buf, IF_max_buf[:, i], "o", label=IF_label[i])
            plt.ylabel(self.data_pos_name_ep01[12, 2] + "(max)")
            plt.xlabel("Shot Number")
            plt.legend()
            #plt.pause(0.1)
            ##plt.show()
            #plt.clf()
            plt.savefig(fname + ".png")

            np.savez_compressed(fname, arr_shotnum=arr_shotnum_buf, IF_max_tmax=IF_max_tmax_buf, IF_max=IF_max_buf, ml_max=ml_max_buf, PECH_max=PECH_max_buf, VG_max=VG_max_buf)
        else:
            pass  # 上記以外の場合は何もしないことを明示
        print('myCheck -> ' + myCheck)

    def make_shotlog(self, data_ep01=None):
        fpath = os.getcwd()
        fname = "RT1_ShotLog_%s.npz" % self.date
        fpn = fpath + '/' + fname
        myCheck = self.fileCheck(fpn)
        self.writeLog(fname, myCheck, data_ep01)

def make_shotlog_series(date):
    arr_shotnum = np.arange(47, 87)
    IF_max_tmax = np.zeros((arr_shotnum.__len__(), 3, 2))
    for i, shotnum in enumerate(arr_shotnum):
        db = DataBrowser(date=date, shotNo=shotnum, LOCALorPPL="LOCAL")
        IF_max_tmax[i, :, :] = db.get_max_tmax()

    np.savez_compressed("data/IF123_max_tmax_%s_%dto%d.npz" % (date, arr_shotnum[0], arr_shotnum[-1]),
                        arr_shotnum=arr_shotnum, IF_max_tmax=IF_max_tmax)

    return arr_shotnum, IF_max_tmax

def plot_shotlog():
    data = np.load("data/IF123_max_tmax_20180223_47to86.npz")
    arr_shotnum = data["arr_shotnum"]
    IF_max_tmax = data["IF_max_tmax"]
    #arr_pulse_width = np.loadtxt("data/pulse_width_20180223_47to86.csv", delimiter=" ")
    arr_pulse_width = np.loadtxt("data/test.txt", delimiter="\t")
    arr_pulse_width[4] = np.nan

    plt.subplot(411)
    plt.plot(arr_pulse_width, IF_max_tmax[:, 0, 1]*1e-4, "o", label="IF1", color="red")
    plt.ylim(1.1, 1.4)
    plt.xlim(5, 30)
    plt.ylabel("Time(max) [sec]")
    plt.legend()
    plt.subplot(412)
    plt.plot(arr_pulse_width, IF_max_tmax[:, 1, 1]*1e-4, "v", label="IF2", color="green")
    plt.ylim(1.1, 1.4)
    plt.xlim(5, 30)
    plt.ylabel("Time(max) [sec]")
    plt.legend()
    plt.subplot(413)
    plt.plot(arr_pulse_width, IF_max_tmax[:, 2, 1]*1e-4, "^", label="IF3", color="blue")
    plt.ylim(1.1, 1.4)
    plt.xlim(5, 30)
    plt.ylabel("Time(max) [sec]")
    plt.legend()
    plt.subplot(414)
    plt.plot(arr_pulse_width, IF_max_tmax[:, 0, 0], "o", label="IF1", color="red")
    plt.plot(arr_pulse_width, IF_max_tmax[:, 1, 0], "v", label="IF2", color="green")
    plt.plot(arr_pulse_width, IF_max_tmax[:, 2, 0], "^", label="IF3", color="blue")
    plt.ylim(0, 3.5)
    plt.xlim(5, 30)
    plt.ylabel("Density(max)")
    plt.xlabel("Pulse Width [msec]")
    plt.legend()
    plt.show()

if __name__ == "__main__":
#    for i in range(20,25):
#        db = DataBrowser(date="20180223", shotNo=i, LOCALorPPL="PPL")
#        #db.load_date(LOCALorPPL="PPL")
#        db.make_shotlog()
    start = time.time()
    db = DataBrowser(date="20180829", shotNo=27, LOCALorPPL="PPL", isShotLog='False')
    db.multiplot()
    process_time = time.time() - start
    print(process_time)
