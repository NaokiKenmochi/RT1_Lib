import read_wvf
import read_wvf_ep02
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.cbook as cbook
import matplotlib.image as image
from PIL import Image
import scipy.signal as sig


class DataBrowser:
    def __init__(self, date, shotNo, LOCALorPPL):
        """

        :param date:
        :param shotNo:
        """
        self.date = date
        self.shotnum = shotNo
        self.LOCALorPPL = LOCALorPPL
        ##干渉計の補正値
        #for IF1
        self.a1 = 0.00
        self.b1 = 0.29
        #for IF2
        self.a2 = -0.005
        self.b2 = 0.135
        #for IF3
        self.a3 = 0.00
        self.b3 = 0.30

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
                                            [3, "IF3", "$\mathbf{n_eL [10^{17}m^{-3}]}$"],
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
                                            [1, "mlt3", "diamag [Wb]"]])

        self.data_pos_name_ep02 = np.array([[2, "MP1", ""],
                                            [2, "MP2", ""],
                                            [2, "MP3", ""],
                                            [6, "IF_FAST", ""],
                                            [6, "IF2_FAST", ""],
                                            [10, "SX", ""],
                                            [10, "SX", ""]])

    def load_date(self, LOCALorPPL):
        """

        :param LOCALorPPL:
        :return:
        """
        if LOCALorPPL == "PPL":
            dm_ep01 = read_wvf.DataManager(self.date)
            dm_ep02_MP = read_wvf_ep02.DataManager("MP", self.date)
            dm_ep02_SX = read_wvf_ep02.DataManager("SX", self.date)
            data_ep01 = dm_ep01.fetch_raw_data(self.shotnum)
            data_ep02_MP = dm_ep02_MP.fetch_raw_data(self.shotnum)
            data_ep02_SX = dm_ep02_SX.fetch_raw_data(self.shotnum)
            #np.savez("data_%s_%d" % (self.date, self.shotnum), data_ep01=data_ep01, data_ep02_MP=data_ep02_MP, data_ep02_SX=data_ep02_SX)
            print("Load IF from PPL")

        else:
            data = np.load("data_%s_%d.npz" % (self.date, self.shotnum))
            data_ep01 = data["data_ep01"]
            data_ep02_MP = data["data_ep02_MP"]
            data_ep02_SX = data["data_ep02_SX"]
            print("Load IF from local")

        return data_ep01, data_ep02_MP, data_ep02_SX

    def multiplot(self):
        """

        :return:
        """
        fig = plt.figure(figsize=(18,10))
        data_ep01, data_ep02_MP, data_ep02_SX = self.load_date(self.LOCALorPPL)
        data_ep01 = self.adj_gain(data_ep01)
        data_ep01 = self.mag_loop(data_ep01)
        data_ep01 = self.calib_IF(data_ep01)
        time_ep02_SX = np.arange(0,2,2/2000000)
        time_ep02_MP = np.arange(0,2,2/1000000)

#        #datafile = cbook.get_sample_data("LOGO_en_140.jpg", asfileobj=False)
#        #im = image.imread("LOGO_en_140.jpg")
#        im = image.imread("LOGO.png")
#        #im = np.array(im)
#        ax1 = fig.add_subplot(5,2,1, sharex=None, sharey=None)
#        ax1.imshow(im, aspect='auto', extent=(0.5,1.1,0.0,0.1), alpha=1.0, zorder=-1)
#        newax = fig.add_axes([0.0, 0.8, 0.1, 0.2], anchor='NE', zorder=-1)
#        newax.imshow(im)
#        newax.axis('off')

        #############################
        #   Date in exp_ep01        #
        #############################
        for j in range(1,33):
            ax1 = fig.add_subplot(6,2,int(self.data_pos_name_ep01[j,0]), sharex=None, sharey=None)
            ax1.set_xlim(0.5, 2.5)
            ax1.plot(data_ep01[0,::self.num_step],data_ep01[j,::self.num_step], label=self.data_pos_name_ep01[j,1])
            ax1.legend(fontsize=10)
            ax1.set_ylabel(self.data_pos_name_ep01[j,2])
            if(j==22 or j==28):
                ax1.set_xlabel("Time [sec]")

        #############################
        #   Date in exp_ep02 (SX)   #
        #############################
        for j in range(1,5):
            ax1 = fig.add_subplot(6,2, int(self.data_pos_name_ep02[j+2,0]), sharex=None, sharey=None)
            ax1.plot(data_ep02_SX[0,::20*self.num_step]+0.5,data_ep02_SX[j,::20*self.num_step]+0.1-0.10*j, label=self.data_pos_name_ep02[j+2, 1])
            plt.subplots_adjust(left=0.05, right=0.97, bottom=0.05, top=0.95, wspace=0.15, hspace=0.15)
            ax1.legend(fontsize=10)

        #############################
        #   Date in exp_ep02 (MP)   #
        #############################
        for j in range(1,4):
            ax1 = fig.add_subplot(6,2, int(self.data_pos_name_ep02[j-1,0]), sharex=None, sharey=None)
            ax1.plot(data_ep02_MP[0,::20*self.num_step]+0.5,data_ep02_MP[j,::20*self.num_step]+0.2-0.10*j, label=self.data_pos_name_ep02[j-1, 1])
            #ax1.plot(time_ep02_MP[::100], data_ep02_MP[j,::100]+0.5-0.25*j, label=self.data_name_ep02[j-1])
            plt.subplots_adjust(left=0.05, right=0.97, bottom=0.05, top=0.95, wspace=0.15, hspace=0.15)
            ax1.legend(fontsize=10)
            if(j == 2):
                plt.title("Date: %s, Shot No.: %d" % (self.date,self.shotnum), loc='right', fontsize=36, fontname="Times New Roman")

        ax1 = fig.add_subplot(6,2,4)
        self.stft(data_ep02_MP[0,:], data_ep02_MP[3,:], self.data_pos_name_ep02[2,1])
        ax1 = fig.add_subplot(6,2,8)
        self.stft(data_ep02_SX[0,:], data_ep02_SX[2,:], self.data_pos_name_ep02[4,1])
        plt.show()

    def mag_loop(self, ml):
        """

        :param ml:
        :return:
        """
        ml[18,:] = ml[18,:]/2.82
        ml[17,:] = ml[17,:] - ml[18,:]
        ml[30,:] = ml[30,:] - ml[18,:]/3.0
        for j in range(17,22):
            ml[j,:] -= np.mean(ml[j,:6000])
            ml[j] = [np.abs(1.0e-4*np.sum(ml[j,:i])) for i in range(len(ml[0]))]
        for j in range(30,33):
            ml[j,:] -= np.mean(ml[j,:6000])
            ml[j] = [np.abs(1.0e-4*np.sum(ml[j,:i])) for i in range(len(ml[0]))]
        return ml

    def calib_IF(self, IF):
        """

        :param IF:
        :return:
        """
        IF[10,:] = np.arcsin((IF[10,:]-self.a1)/self.b1)*180/np.pi - np.mean(np.arcsin((IF[10,:5000]-self.a1)/self.b1)*180/np.pi)
        IF[11,:] = np.arcsin((IF[11,:]-self.a2)/self.b2)*180/np.pi - np.mean(np.arcsin((IF[11,:5000]-self.a2)/self.b2)*180/np.pi)
        IF[12,:] = np.arcsin((IF[12,:]-self.a3)/self.b3)*180/np.pi - np.mean(np.arcsin((IF[12,:5000]-self.a3)/self.b3)*180/np.pi)

        IF[10,:] = IF[10,:]*5.58/360
        IF[11,:] = IF[11,:]*5.58/360
        IF[12,:] = IF[12,:]*5.58/360
#        IF[10,:] = np.arcsin((IF[10,:]-self.a1)/self.b1)*180/np.pi
#        IF[11,:] = np.arcsin((IF[11,:]-self.a2)/self.b2)*180/np.pi
#        IF[12,:] = np.arcsin((IF[12,:]-self.a3)/self.b3)*180/np.pi
#
#        IF[10,:] -= np.mean(IF[10,:6000])
#        IF[11,:] -= np.mean(IF[11,:6000])
#        IF[12,:] -= np.mean(IF[12,:6000])

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

    def stft(self, x, y, label):
        MAXFREQ = 5e4
        N = np.abs(1/(x[1]-x[2]))
        f, t, Zxx =sig.spectrogram(y, fs=N, window='hamming', nperseg=5000)
        #plt.xlim(0, 1.0)
        plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=4e-8)
        #plt.contourf(t, f, np.abs(Zxx), 200, norm=LogNorm())# vmax=1e-7)
        plt.ylabel(label + "\nFrequency [Hz]")
        #plt.xlabel("Time [sec]")
        plt.ylim([0, MAXFREQ])


if __name__ == "__main__":
    db = DataBrowser(date="20170608", shotNo=74, LOCALorPPL="LOCAL")
    db.multiplot()
