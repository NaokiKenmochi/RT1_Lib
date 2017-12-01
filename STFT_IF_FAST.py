import sys
sys.path.append('/Users/kemmochi/PycharmProjects/ControlCosmoZ')

import numpy as np
import pywt
import read_wvf
import czdec
import scipy.signal as sig
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


class STFT_FAST:
    def __init__(self, date, shotNo, LOCALorPPL):
        """

        :param date:
        :param shotNo:
        """
        self.date = date
        self.shotnum = shotNo
        self.LOCALorPPL = LOCALorPPL

    def load_ep01(self, LOCALorPPL):
        if LOCALorPPL == "PPL":
            dm_ep01 = read_wvf.DataManager("exp_ep01", 0, self.date)
            data_ep01 = dm_ep01.fetch_raw_data(self.shotnum)
            print("Load ep01 from PPL")

        else:
            data = np.load("IF_%s_%d.npz" % (self.date, self.shotnum))
            data_ep02_SX = data["data_ep02_SX"]
            filename = "GP1_20171110_107_IF1IF2FAST.txt"
            IF_FAST = np.loadtxt(filename, delimiter=",")
            print("Load SX from local")

        return data_ep01

    def load_IF_FAST(self, LOCALorPPL):
        if LOCALorPPL == "PPL":
            dm_ep02_SX = read_wvf.DataManager("exp_ep02", "SX", self.date)
            data_ep02_SX = dm_ep02_SX.fetch_raw_data(self.shotnum)
            print("Load SX from PPL")

        else:
            data = np.load("IF_%s_%d.npz" % (self.date, self.shotnum))
            data_ep02_SX = data["data_ep02_SX"]
            filename = "GP1_20171110_107_IF1IF2FAST.txt"
            IF_FAST = np.loadtxt(filename, delimiter=",")
            print("Load SX from local")

        return data_ep02_SX

    def load_MP_FAST(self, LOCALorPPL):
        if LOCALorPPL == "PPL":
            dm_ep02_MP = read_wvf.DataManager("exp_ep02", "MP", self.date)
            data_ep02_MP = dm_ep02_MP.fetch_raw_data(self.shotnum)
            print("Load MP from PPL")

        else:
            data = np.load("MP123_%s_%d.npz" % (self.date, self.shotnum))
            data_ep02_MP = data["data_ep02_MP"]
            print("Load MP from local")

        return data_ep02_MP

    def load_SX_CosmoZ(self, LOCALorPPL):
        if LOCALorPPL == "PPL":
            data = czdec.CosmoZ_DataBrowser(filepath= '/Volumes/share/Cosmo_Z_xray/', filename="", date=self.date, shotnum=self.shotnum)
            data_SX = data.read_cosmoz()
            print("Load SX from PPL")

        else:
            #data = np.load("/Users/kemmochi/PycharmProjects/ControlCosmoZ/SX_%s_%d.npz" % (self.date, self.shotnum))
            data = np.load("data/19.npz")
            data_SX = data['energy']
            time_SX = data['time']
            print("Load SX from local")

        return data_SX, time_SX

    def cwt(self):
        # TODO: 時間がかかりすぎる　要確認
        #IF_FAST = self.load_IF_FAST("PPL")
        #MP_FAST = self.load_MP_FAST("PPL")
        #y = MP_FAST[1, :]
        #x = MP_FAST[0, :]
        num_IF = 1
        y = IF_FAST[num_IF, ::1000]
        x = np.linspace(0, 2, 2000)
        #N = 1e-3*np.abs(1/(x[1]-x[2]))
        N=4000
        wfreq = np.linspace(1,N, 4000)
        #coef = sig.cwt(y, sig.ricker, wfreq)
        #coef, freqs=pywt.cwt(y, wfreq, 'cmor')
        coef, freqs=pywt.cwt(y, wfreq, 'mexh')

        MAXFREQ = 1e0
        #plt.xlim(0, 1.0)
        #plt.contourf(t, f, np.abs(Zxx), 200, norm=LogNorm())# vmax=1e-7)
        plt.ylabel("CWT Frequency of IF%d [Hz]" % (num_IF))
        plt.xlabel("Time [sec]")
        #plt.ylim([0, MAXFREQ])
        #plt.xlim([0.8, 2.2])
        #ax3.contourf(x, 200/wfreq, coef, 200)
        plt.contourf(x,N*freqs, np.sqrt(np.real(coef)**2+np.imag(coef)**2), 100, vmax=5.0)
        #ax3.contourf(x, 200/wfreq, np.sqrt(np.real(coef)**2+np.imag(coef)**2), 200, vmax=0.4)
        #plt.colorbar()
        plt.title("Date: %s, Shot No.: %d" % (self.date, self.shotnum), loc='right', fontsize=20, fontname="Times New Roman")
        plt.show()

    def stft(self):
        data_SX, time_SX = self.load_SX_CosmoZ(self.LOCALorPPL)
        time_SX_10M = np.linspace(0, 2, 2e7)
        data_SX_10M = np.zeros(2e7)
        data_SX_10M[[i for i in time_SX*1e7]] = data_SX
        y = data_SX_10M
        x = time_SX_10M
        MAXFREQ = 1e6
        N = np.abs(1/(x[1]-x[2]))

        plt.figure(figsize=(8, 5))
        f, t, Zxx =sig.spectrogram(y, fs=N, window='hamming', nperseg=500000)
        vmin = 0.0
        vmax = 5e-1
        plt.pcolormesh(t, f, np.abs(Zxx), vmin=vmin, vmax=vmax)
        sfmt=matplotlib.ticker.ScalarFormatter(useMathText=True)
        cbar = plt.colorbar(format=sfmt)
        cbar.ax.tick_params(labelsize=12)
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        plt.ylabel("Frequency of SX [Hz]")
        plt.xlabel("Time [sec]")
        plt.ylim([0, MAXFREQ])
        plt.title("Date: %s, Shot No.: %d" % (self.date, self.shotnum), loc='right', fontsize=20, fontname="Times New Roman")
        filepath = "figure/"
        filename = "STFT_SX_20171110_19"
        plt.savefig(filepath + filename)
        plt.clf()

if __name__ == "__main__":
    stft = STFT_FAST(date="20171110", shotNo=19, LOCALorPPL="LOCAL")
    stft.stft()
    #stft.cwt()
