from RT1DataBrowser import DataBrowser
from matplotlib import gridspec
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
import matplotlib as mpl
#mpl.use('Qt4Agg')


class STFT_RT1(DataBrowser):
    def __init__(self, date, shotNo, LOCALorPPL):
        """

        :param date:
        :param shotNo:
        """
        super().__init__(date, shotNo, LOCALorPPL)
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

    def load_SX_FAST(self, LOCALorPPL):
        if LOCALorPPL == "PPL":
            dm_ep02_SX = read_wvf.DataManager("exp_ep02", "SX", self.date)
            data_ep02_SX = dm_ep02_SX.fetch_raw_data(self.shotnum)
            print("Load SX from PPL")

        else:
            data = np.load("SX123_%s_%d.npz" % (self.date, self.shotnum))
            data_ep02_MP = data["data_ep02_SX"]
            print("Load SX from local")

        return data_ep02_SX

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

    def moving_average(self, x, N):
        # Take a moving average along axis=1 with window width N.
        x = np.pad(x, ((0, 0), (N, 0)), mode='constant')
        cumsum = np.cumsum(x, axis=1)
        return (cumsum[:, N:] - cumsum[:, :-N]) / N

    def cross_spectrum(self):
        data_ep01 = self.load_ep01("PPL")
        data_ep01 = self.adj_gain(data_ep01)
        data_ep01 = self.calib_IF(data_ep01)
        IF = data_ep01[9:12:2, :].T
        N = np.abs(1/(data_ep01[0, 1]-data_ep01[0, 2]))
        sampling_time = 1/N

        #IF_FAST = self.load_IF_FAST("PPL")
        #IF = IF_FAST[1:3, :].T
        #sampling_time = 1e-6
        #f, t, Pxx = sig.spectrogram(IF, axis=0, fs=1/sampling_time, window='hamming', nperseg=128, noverlap=64, mode='complex')
        #f, t, Pxx = sig.spectrogram(IF, axis=0, fs=1/sampling_time, window='hamming', nperseg=2**15, noverlap=512, mode='complex')
        f, t, Pxx = sig.spectrogram(IF, axis=0, fs=1/sampling_time, window='hamming', nperseg=2**9, noverlap=16, mode='complex')
        #Pxx_run = self.moving_average(Pxx[:, 0] * np.conj(Pxx[:, 1]), 8)
        Pxx_run = self.moving_average(Pxx[:, 0] * np.conj(Pxx[:, 1]), 2)

        DPhase = 180/np.pi*np.arctan2(Pxx_run.imag, Pxx_run.real)

        plt.pcolormesh(t, f, np.log(np.abs(Pxx_run)))
        #plt.pcolormesh(t, f, DPhase)
        plt.xlim(0.5, 2.5)
        #plt.clim(-16, -13)
        plt.clim(-28, -25)
        plt.ylim(0, 2000)
        plt.colorbar()
        plt.xlabel('Time [sec]')
        plt.ylabel('Frequency [Hz]')
        plt.show()

    def stft(self, IForMPorSX="IF", num_ch=1):

        time_offset_stft = 0.0
        if(IForMPorSX=="IF"):
            data_ep01 = self.load_ep01("PPL")
            data_ep01 = self.adj_gain(data_ep01)
            data_ep01 = self.calib_IF(data_ep01)

            y = data_ep01[10:13, :]
            x = data_ep01[0, :]
            filename = "STFT_IF_%s_%d" % (self.date, self.shotnum)
            vmin = 0.0
            vmax = 1e-5
            NPERSEG = 2**9
            time_offset = 0.0

        if(IForMPorSX=="POL"):
            """
            390nm, 730nm, 710nm, 450nmの順で格納
            390nm, 450nmの比を用いて電子密度を計算
            730nm, 710nmの比を用いて電子温度を計算
            """
            data_ep01 = self.load_ep01("PPL")
            data_ep01 = self.adj_gain(data_ep01)

            y_buf1 = np.array([data_ep01[13, :]])
            y_buf2 = np.array(data_ep01[25:28, :])
            y = np.r_[y_buf1, y_buf2]
            x = data_ep01[0, :]
            filename = "STFT_POL_%s_%d" % (self.date, self.shotnum)
            vmin = 0.0
            vmax = 3e-7
            NPERSEG = 2**9
            time_offset = 0.0

        if(IForMPorSX=="IF_FAST"):
            IF_FAST = self.load_IF_FAST("PPL")
            y = IF_FAST[1:, :]
            x = np.linspace(0, 2, 2000000)
            filename = "STFT_IF_FAST_%s_%d" % (self.date, self.shotnum)
            vmin = 0.0
            vmax = 5e-7
            NPERSEG = 2**15
            time_offset = 0.75
            time_offset_stft = 0.75

        if(IForMPorSX=="MP"):
            MP_FAST = self.load_MP_FAST("PPL")
            y = MP_FAST[1:, :]
            x = MP_FAST[0, :]
            filename = "STFT_MP_%s_%d" % (self.date, self.shotnum)
            vmin = 0.0
            vmax = 1e-7
            NPERSEG = 1024
            time_offset = 1.25
            time_offset_stft = 0.25
            #plt.plot(x, MP_FAST[1, :]+1, label="MP1")
            #plt.plot(x, MP_FAST[2, :], label="MP2")
            #plt.plot(x, MP_FAST[3, :]-1, label="MP3")
            #plt.legend()
            #plt.show()

        if(IForMPorSX=="SX"):
            data_SX, time_SX = self.load_SX_CosmoZ(self.LOCALorPPL)
            y = data_SX[:, 4]
            x = data_SX[:, 0]
            time_SX_10M = np.linspace(0, 2, 2e7)
            data_SX_10M = np.zeros(2e7)
            data_SX_10M[[i for i in time_SX*1e7]] = data_SX
            y = data_SX_10M
            x = time_SX_10M
            time_offset_stft = 1.00
            time_offset = 1.25
            #plt.plot(x, y)
            #plt.show()
            filename = "STFT_SX4_%s_%d" % (self.date, self.shotnum)

        if(IForMPorSX=="REF"):
            SX_FAST = self.load_SX_FAST("PPL")
            y = SX_FAST[3:, :]
            x = SX_FAST[0, :]
            filename = "STFT_REF_%s_%d" % (self.date, self.shotnum)
            vmin = 0.0
            vmax = 5e-7
            NPERSEG = 2**14
            time_offset_stft = 0.75
            time_offset = 1.25
            #plt.plot(x, SX_FAST[3, :]+1, label="REF_COS")
            #plt.plot(x, SX_FAST[4, :], label="REF_SIN")
            #plt.plot(SX_FAST[3, ::10000], SX_FAST[4, ::10000])
            #plt.legend()
            #plt.show()


        N = np.abs(1/(x[1]-x[2]))

        plt.figure(figsize=(16, 5))
        gs = gridspec.GridSpec(4, num_ch)
        gs.update(hspace=0.4, wspace=0.3)
        for i in range(num_ch):

            ax0 = plt.subplot(gs[0:3, i])
            f, t, Zxx =sig.spectrogram(y[i, :], fs=N, window='hamming', nperseg=NPERSEG)
            plt.pcolormesh(t + time_offset_stft, f, np.abs(Zxx), vmin=vmin, vmax=vmax)
            sfmt=matplotlib.ticker.ScalarFormatter(useMathText=True)
            cbar = plt.colorbar(format=sfmt)
            cbar.ax.tick_params(labelsize=12)
            cbar.formatter.set_powerlimits((0, 0))
            cbar.update_ticks()
            ax0.set_xlabel("Time [sec]")
            ax0.set_ylabel("Frequency of %s%d [Hz]" % (IForMPorSX, i+1))
            ax0.set_xlim(0.5, 2.5)
            ax0.set_ylim([0, 2000])
            if(i==num_ch-1):
                plt.title("%s" % (filename), loc='right', fontsize=20, fontname="Times New Roman")

            ax1 = plt.subplot(gs[3, i])
            ax1.plot(x+time_offset, y[i, :])
            ax1.set_xlabel("Time [sec]")
            ax1.set_xlim(0.5, 3.0)
        #plt.show()
        filepath = "figure/"
        plt.savefig(filepath + filename)

if __name__ == "__main__":
    for i in range(47, 87):
        stft = STFT_RT1(date="20180223", shotNo=i, LOCALorPPL="PPL")
        stft.stft(IForMPorSX="POL", num_ch=4)
    #stft = STFT_RT1(date="20180223", shotNo=47, LOCALorPPL="PPL")
    #stft.stft(IForMPorSX="IF", num_ch=3)
    #stft.cwt()
    #stft.cross_spectrum()
