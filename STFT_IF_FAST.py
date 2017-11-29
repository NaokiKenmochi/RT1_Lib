import numpy as np
import pywt
import read_wvf
import scipy.signal as sig
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

class STFT_FAST:
    def __init__(self, data, shotNo, LOCALorPPL):
        """

        :param data:
        :param shotNo:
        """
        self.data = data
        self.shotnum = shotNo
        self.LOCALorPPL = LOCALorPPL

    def load_ep01(self, LOCALorPPL):
        if LOCALorPPL == "PPL":
            dm_ep01 = read_wvf.DataManager("exp_ep01", 0, self.data)
            data_ep01 = dm_ep01.fetch_raw_data(self.shotnum)
            print("Load ep01 from PPL")

        else:
            data = np.load("IF_%s_%d.npz" % (self.data, self.shotnum))
            data_ep02_SX = data["data_ep02_SX"]
            filename = "GP1_20171110_107_IF1IF2FAST.txt"
            IF_FAST = np.loadtxt(filename, delimiter=",")
            print("Load SX from local")

        return data_ep01

    def load_IF_FAST(self, LOCALorPPL):
        if LOCALorPPL == "PPL":
            dm_ep02_SX = read_wvf.DataManager("exp_ep02", "SX", self.data)
            data_ep02_SX = dm_ep02_SX.fetch_raw_data(self.shotnum)
            print("Load SX from PPL")

        else:
            data = np.load("IF_%s_%d.npz" % (self.data, self.shotnum))
            data_ep02_SX = data["data_ep02_SX"]
            filename = "GP1_20171110_107_IF1IF2FAST.txt"
            IF_FAST = np.loadtxt(filename, delimiter=",")
            print("Load SX from local")

        return data_ep02_SX

    def load_MP_FAST(self, LOCALorPPL):
        if LOCALorPPL == "PPL":
            dm_ep02_MP = read_wvf.DataManager("exp_ep02", "MP", self.data)
            data_ep02_MP = dm_ep02_MP.fetch_raw_data(self.shotnum)
            print("Load MP from PPL")

        else:
            data = np.load("MP123_%s_%d.npz" % (self.data, self.shotnum))
            data_ep02_MP = data["data_ep02_MP"]
            print("Load MP from local")

        return data_ep02_MP

    def cwt(self):
        IF_FAST = self.load_IF_FAST("PPL")
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
        plt.title("Date: %s, Shot No.: %d" % (self.data,self.shotnum), loc='right', fontsize=20, fontname="Times New Roman")
        plt.show()

    def stft(self):
        #ep01 = self.load_ep01("PPL")
        #IF_FAST = self.load_IF_FAST("PPL")
        MP_FAST = self.load_MP_FAST("PPL")
        y = MP_FAST[1, :]
        x = MP_FAST[0, :]
        #num_IF = 1
        #y = IF_FAST[num_IF, :]
        #x = np.linspace(0, 2, 2000000)
        #y = ep01[11, :]
        #x = ep01[0, :]
        #plt.plot(x, MP_FAST[1, :])
        #plt.plot(x, MP_FAST[2, :])
        #plt.plot(x, MP_FAST[3, :])
        #plt.plot(x, y)
        #plt.show()
        MAXFREQ = 1e3
        N = np.abs(1/(x[1]-x[2]))
        f, t, Zxx =sig.spectrogram(y, fs=N, window='hamming', nperseg=50000)
        #f, t, Zxx =sig.spectrogram(y, fs=N, window='hamming', nperseg=500)
        #plt.xlim(0, 1.0)
        #plt.pcolormesh(t+0.76316, f, np.abs(Zxx), vmin=0, vmax=2e-6)
        #plt.contourf(t+0.76316, f, np.abs(Zxx), 10, norm=LogNorm(), vmax=2e-7)
        #plt.ylabel("Frequency of IF%d [Hz]" % (num_IF))
        plt.ylabel("Frequency of MP1 [Hz]")
        plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=3e-8)
        plt.xlabel("Time [sec]")
        plt.ylim([0, MAXFREQ])
        #plt.xlim([0.8, 2.2])
        plt.title("Date: %s, Shot No.: %d" % (self.data,self.shotnum), loc='right', fontsize=20, fontname="Times New Roman")
        #plt.show()
        filepath = "figure/"
        filename = "STFT_MP1_%s_%d" % (self.data, self.shotnum)
        plt.savefig(filepath + filename)

if __name__ == "__main__":
    stft = STFT_FAST(data="20171111", shotNo=31, LOCALorPPL="PPL")
    stft.stft()
    #stft.cwt()
