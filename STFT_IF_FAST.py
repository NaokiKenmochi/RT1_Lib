import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

class STFT_FAST:
    def __init__(self, data, shotNo, LOCALorPPL):
        """

        :param data:
        :param shotNo:
        """
        self.data = data
        self.shotnum = shotNo
        self.LOCALorPPL = LOCALorPPL

    def load_IF_FAST(self):
        filename = "GP1_20171110_107_IF1IF2FAST.txt"
        IF_FAST = np.loadtxt(filename, delimiter=",")

        return IF_FAST

    def load_MP_FAST(self):
        data = np.load("MP123_%s_%d.npz" % (self.data, self.shotnum))
        data_ep02_MP = data["data_ep02_MP"]
        print("Load MP from local")

        return data_ep02_MP

    def stft(self, label):
        #IF_FAST = self.load_IF_FAST()
        MP_FAST = self.load_MP_FAST()
        #y = IF_FAST[:,0]
        x = MP_FAST[0, :]
        y = MP_FAST[3, :]
        #x = np.linspace(0, 2, 1000000)
        #plt.plot(x, MP_FAST[1, :])
        #plt.plot(x, MP_FAST[2, :])
        #plt.plot(x, MP_FAST[3, :])
        #plt.show()
        MAXFREQ = 1e1
        N = 1e-3*np.abs(1/(x[1]-x[2]))
        f, t, Zxx =sig.spectrogram(y, fs=N, window='hamming', nperseg=10000)
        #plt.xlim(0, 1.0)
        plt.pcolormesh(t*1e-3+0.76316, f, np.abs(Zxx), vmin=0, vmax=4e-5)
        #plt.contourf(t, f, np.abs(Zxx), 200, norm=LogNorm())# vmax=1e-7)
        plt.ylabel(label + "\nFrequency [kHz]")
        plt.ylim([0, MAXFREQ])
        #plt.xlim([0.8, 2.2])
        plt.show()

if __name__ == "__main__":
    stft = STFT_FAST(data="20171110", shotNo=107, LOCALorPPL="PPL")
    stft.stft("MP1")
