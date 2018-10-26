from RT1DataBrowser import DataBrowser
from matplotlib import gridspec
from matplotlib import mlab
from scipy.fftpack import fft
import sys
#sys.path.append('/Users/kemmochi/PycharmProjects/ControlCosmoZ')

import numpy as np
import pywt
import read_wvf
#import czdec
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
        MP_FAST = self.load_MP_FAST("PPL")
        IF_FAST = self.load_IF_FAST("PPL")
        #plt.plot(MP_FAST[0, :], MP_FAST[3, :])
        #plt.plot(MP_FAST[3, :])
        #plt.show()
        #IF = data_ep01[11:13:1, :].T
        IF = data_ep01[11:13:1, :].T
        #IF[:,0] *= -1
        IF_MP = np.zeros((28000, 2))
        #IF_MP[:, 0] = data_ep01[10, 8000:22000].T
        #IF_MP[:, 1] = data_ep01[12, 8000:22000].T
        #IF_MP[:, 1] = MP_FAST[3, 265000:965000:50].T

        #fs = 2e4
        #N = 2.8e4
        #time = np.arange(N)/float(fs)
        #x1 = np.sin(2*np.pi*600*time)
        #x2 = np.sin(2*np.pi*600*(time-0.00005))

        #IF_MP[:, 0] = x1[::1]
        #IF_MP[:, 1] = x2[::1]
        IF_MP[:, 0] = MP_FAST[1, 10500:38500:1].T
        IF_MP[:, 1] = MP_FAST[6, 10500:38500:1].T
        IF = IF_MP
        IF = IF_FAST[1:4:2, :].T
        #IF = data_ep01[11:13, :].T
        N = 2*np.abs(1/(data_ep01[0, 1]-data_ep01[0, 2]))
        N = 1e6 #IF_FAST
        #N = np.abs(1/(data_ep01[0, 1]-data_ep01[0, 2]))
        sampling_time = 1/N
        plt.plot(IF, label="IF")
        plt.legend()
        plt.show()

        #sampling_time = 1e-6
        #f, t, Pxx = sig.spectrogram(IF, axis=0, fs=1/sampling_time, window='hamming', nperseg=128, noverlap=64, mode='complex')
        #f, t, Pxx = sig.spectrogram(IF, axis=0, fs=1/sampling_time, window='hamming', nperseg=2**15, noverlap=512, mode='complex')
        #f, t, Pxx = sig.spectrogram(IF, axis=0, fs=1/sampling_time, window='hamming', nperseg=2**8, noverlap=16, mode='complex')    #MP
        f, t, Pxx = sig.spectrogram(IF, axis=0, fs=1/sampling_time, window='hamming', nperseg=2**14, noverlap=16, mode='complex')    #IF_FAST
        Pxx_run = self.moving_average(Pxx[:, 0] * np.conj(Pxx[:, 1]), 8)
        weight = Pxx_run
        weight = np.where(np.log(np.abs(Pxx_run)) > -17.0, 1, 0)
        #weight = np.where(np.log(np.abs(Pxx_run)) > -14.5, 1, 0)

        #２列目の位相が進んでいる場合にDPhaseは正になる
        DPhase = 180/np.pi*np.arctan2(Pxx_run.imag, Pxx_run.real)

        #plt.pcolormesh(t, f, np.abs(Pxx[:, 0]))
        plt.subplot(211)
        plt.title("Date: %s, Shot No.: %d" % (self.date, self.shotnum), loc='right', fontsize=16, fontname="Times New Roman")
        #plt.pcolormesh(t, f, np.log(np.abs(Pxx_run)), vmin=-18.5, vmax=-17)
        plt.pcolormesh(t+0.8, f, np.log(np.abs(Pxx_run)), vmin=-17, vmax=-16)
        #plt.pcolormesh(t+0.8, f, np.log(np.abs(Pxx_run)))
        plt.ylim(0, 2000)
        plt.xlim(0.5, 2.5)
        plt.ylabel('Cross-Spectrum b/w IF2 and REF \nFrequency [Hz]')
        #plt.ylabel('Cross-Spectrum b/w IF23 \nFrequency [Hz]')
        plt.colorbar()
        plt.subplot(212)
        plt.pcolormesh(t+0.8, f, DPhase*weight, cmap='bwr', vmin=-180, vmax=180)
        #plt.pcolormesh(t+0.8, f, 120/(DPhase*weight), cmap='Set1', vmin=-8, vmax=8)
        #plt.pcolormesh(t+0.8, f, weight)
        #plt.xlim(0.5, 2.5)
        #plt.clim(-16, -14.5)
        #plt.clim(-26, -24.5)
        plt.ylim(0, 2000)
        plt.xlim(0.5, 2.5)
        plt.colorbar()
        plt.xlabel('Time [sec]')
        plt.ylabel('Phase Difference b/w IF2 and REF\nFrequency [Hz]')
        #plt.ylabel('Phase Difference b/w IF23 \nFrequency [Hz]')
        filepath = "figure/"
        filename = "CSD_PD_MP16_%s_%d" % (self.date, self.shotnum)
        plt.savefig(filepath + filename)
        #plt.show()
        plt.clf()

        #f, t, Zxx =sig.spectrogram(IF_MP[:,1], fs=N, window='hamming', nperseg=2**8)
        ##FS = 1/(MP_FAST[0, 1] - MP_FAST[0, 0])
        ##f, t, Zxx =sig.spectrogram(MP_FAST[3, :], fs=FS, window='hamming', nperseg=2**14)
        #plt.pcolormesh(t, f, Zxx, vmax=2e-7)
        #plt.ylim(0, 2000)
        #plt.show()

        #for i in range(80):
        #    #f, Cxy = sig.coherence(IF_MP[2000+i*500:2500+i*500, 0], IF_MP[2000+i*500:2500+i*500, 1], N, nperseg=2**6)
        #    f, Cxy = sig.coherence(IF_MP[4000+i*250:4250+i*250, 0], IF_MP[4000+i*250:4250+i*250, 1], N, nperseg=2**7)
        #    if i == 0:
        #        Cxy_buf = Cxy
        #    else:
        #        Cxy_buf = np.c_[Cxy_buf, Cxy]
        #    plt.plot(f, Cxy_buf)
        #t = np.arange(1, 2, 0.0125)
        #T, F = np.meshgrid(t, f)
        #plt.pcolormesh(T, F, Cxy_buf)
        #plt.xlim(1, 2)
        #plt.ylim(0, 2000)
        #plt.colorbar(orientation="vertical")
        #plt.show()
        #f, Cxy = sig.coherence(IF_MP[4000:5000, 0], IF_MP[4000:5000, 1], N, nperseg=2**7)
        ##f, Cxy = sig.coherence(data_ep01[10, 12000:14000], data_ep01[11, 12000:14000], N, nperseg=2**8)
        ##f, Cxy = sig.coherence(data_ep01[10, 12000:13000], MP_FAST[3, 390000:400000:10], N, nperseg=2**8)
        ##plt.semilogy(f, Cxy)
        #plt.plot(f, Cxy)
        #plt.title("coherence")
        #plt.show()

    def phase_diff(self, x, y, f, t, t_offset):
        N = np.abs(1/(x[1]-x[2]))
        plt.plot(y[0])
        plt.plot(y[2])
        plt.show()
        f, t, Sxx1 = sig.spectrogram(y[0], fs=N, mode='angle', window='hamming', nperseg=2**10)
        f, t, Sxx2 = sig.spectrogram(y[1], fs=N, mode='angle', window='hamming', nperseg=2**10)
        plt.pcolormesh(t+t_offset, f, Sxx1)
        plt.colorbar()
        plt.ylim(0, 2000)
        plt.show()
        plt.pcolormesh(t+t_offset, f, Sxx2)
        plt.colorbar()
        plt.ylim(0, 2000)
        plt.show()
        phase_diff = Sxx1-Sxx2#np.angle(Sxx2) - np.angle(Sxx1)
        plt.pcolormesh(t+t_offset, f, phase_diff)#, vmin=-np.pi, vmax=np.pi)
        #plt.pcolormesh(t, f, np.angle(Sxx2-Sxx1))
        plt.colorbar()
        plt.ylim(0, 2000)
        plt.show()

    def phase_diff_test(self):
        fs = 2e5
        N = 2e5
        time = np.arange(N)/float(fs)
        x1 = np.sin(2*np.pi*600*time)
        x2 = np.sin(2*np.pi*600*(time-0.0002))
        x1[1e5:] = 0
        x2[1e5:] = 0

        #yf1 = fft(x1)
        #yf2 = fft(x2)

        #plt.plot(np.linspace(1, N, N), np.angle(yf1) - np.angle(yf2))
        #plt.axis('tight')
        #plt.xlim(0, 2000)
        #plt.xlabel("data number")
        #plt.ylabel("phase[deg]")
        #plt.show()

        plt.plot(time, x1)
        plt.plot(time, x2)
        plt.xlim(0, 0.01)
        plt.show()
        f, t, Sxx1 = sig.spectrogram(x1, fs, mode='phase', window='hamming', nperseg=2**11)
        f, t, Sxx2 = sig.spectrogram(x2, fs, mode='phase', window='hamming', nperseg=2**11)
        plt.pcolormesh(t, f, Sxx1)
        plt.colorbar()
        plt.ylim(0, 2000)
        plt.show()
        #plt.pcolormesh(t, f, np.angle(Sxx2))
        plt.pcolormesh(t, f, Sxx2)
        plt.colorbar()
        plt.ylim(0, 2000)
        plt.show()
        phase_diff = Sxx1-Sxx2#np.angle(Sxx2) - np.angle(Sxx1)
        plt.pcolormesh(t, f, phase_diff, vmin=-np.pi, vmax=np.pi)
        #plt.pcolormesh(t, f, np.angle(Sxx2-Sxx1))
        plt.colorbar()
        plt.ylim(0, 2000)
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
            coef_vmax = 0.8
            NPERSEG = 2**9
            time_offset = 0.0

        elif(IForMPorSX=="IF_FAST"):
            IF_FAST = self.load_IF_FAST("PPL")
            y = IF_FAST[1:, :]
            x = np.linspace(0, 2, 2000000)
            filename = "STFT_IF_FAST_%s_%d" % (self.date, self.shotnum)
            vmin = 0.0
            vmax = 5e-7
            coef_vmax = 0.8
            NPERSEG = 2**16
            time_offset = 0.75
            time_offset_stft = 0.75

        elif(IForMPorSX=="POL"):
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
            vmax = 3e-3
            coef_vmax = 0.8
            NPERSEG = 2**9
            time_offset = 0.0

        elif(IForMPorSX=="POL_RATIO"):
            """
            390nm, 450nmの比を用いて電子密度を計算
            730nm, 710nmの比を用いて電子温度を計算
            """
            data_ep01 = self.load_ep01("PPL")
            data_ep01 = self.adj_gain(data_ep01)

            y_Te = np.array([(data_ep01[27, :]+1.0)/(data_ep01[13, :]+1.0)])
            y_ne = np.array([(data_ep01[25, :]+1.0)/(data_ep01[26, :]+1.0)])
            y = np.r_[y_Te, y_ne]
            x = data_ep01[0, :]
            filename = "STFT_POL_RATIO_woffset_%s_%d" % (self.date, self.shotnum)
            vmin = 0.0
            vmax = 1e-6
            coef_vmax = 1.0e0
            NPERSEG = 2**8
            time_offset = 0.0


        elif(IForMPorSX=="MP"):
            MP_FAST = self.load_MP_FAST("PPL")
            y = MP_FAST[1:, :]
            x = MP_FAST[0, :]
            filename = "STFT_MP_%s_%d" % (self.date, self.shotnum)
            vmin = 0.0
            vmax = 1e-7
            coef_vmax = 0.8
            #NPERSEG = 2**14
            NPERSEG = 2**10
            #NPERSEG = 1024
            time_offset = 1.25
            time_offset_stft = 0.25
            plt.plot(x, MP_FAST[1, :], label="MP1")
            plt.plot(x, MP_FAST[3, :], label="MP3")
            plt.plot(x, MP_FAST[6, :], label="MP6")
            plt.legend()
            plt.show()

        elif(IForMPorSX=="SX"):
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
            vmin = 0.0
            vmax = 1e-7
            coef_vmax = 0.8

        elif(IForMPorSX=="REF"):
            SX_FAST = self.load_SX_FAST("PPL")
            y = SX_FAST[3:, :]
            x = SX_FAST[0, :]
            filename = "STFT_REF_%s_%d" % (self.date, self.shotnum)
            vmin = 0.0
            vmax = 5e-7
            coef_vmax = 0.7
            NPERSEG = 2**14
            time_offset_stft = 0.75
            time_offset = 1.25
            #plt.plot(x, SX_FAST[3, :]+1, label="REF_COS")
            #plt.plot(x, SX_FAST[4, :], label="REF_SIN")
            #plt.plot(SX_FAST[3, ::10000], SX_FAST[4, ::10000])
            #plt.legend()
            #plt.show()


        N = np.abs(1/(x[1]-x[2]))

        for i in range(num_ch):

            #f, t, Zxx =sig.spectrogram(y[i, :], fs=N, window='hamming', nperseg=NPERSEG, mode='complex')
            f, t, Zxx =sig.spectrogram(y[i, :], fs=N, window='hamming', nperseg=NPERSEG)
            #f, t, Zxx =sig.stft(y[i, :], fs=N, window='hamming', nperseg=NPERSEG)
            if(i == 0):
                #Zxx_3D = np.zeros((np.shape(Zxx)[0], np.shape(Zxx)[1], num_ch), dtype=complex)
                Zxx_3D = np.zeros((np.shape(Zxx)[0], np.shape(Zxx)[1], num_ch))
            Zxx_3D[:, :, i] = Zxx[:, :]

        return f, t, Zxx_3D, filename, vmax, coef_vmax,  vmin, time_offset, time_offset_stft, x, y

    def plot_stft(self, IForMPorSX="IF", num_ch=4):
        f, t, Zxx_3D, filename, vmax, coef_vmax, vmin, time_offset, time_offset_stft, x, y = self.stft(IForMPorSX=IForMPorSX, num_ch=num_ch)

        #vmaxを求める際の時間(t)，周波数(f)の範囲とそのindexを取得
        t_st = 1.2
        t_ed = 1.4
        f_st = 550
        f_ed = 650
        idx_tst = np.abs(np.asarray(t - t_st)).argmin()
        idx_ted = np.abs(np.asarray(t - t_ed)).argmin()
        idx_fst = np.abs(np.asarray(f - f_st)).argmin()
        idx_fed = np.abs(np.asarray(f - f_ed)).argmin()

        plt.figure(figsize=(16, 5))
        gs = gridspec.GridSpec(4, num_ch)
        gs.update(hspace=0.4, wspace=0.3)
        for i in range(num_ch):
            ax0 = plt.subplot(gs[0:3, i])
            try:
                vmax_in_range = np.max(np.abs(Zxx_3D[idx_fst:idx_fed, idx_tst:idx_ted, i])) * coef_vmax
            except ValueError:
                vmax_in_range = 1e-10
            plt.pcolormesh(t + time_offset_stft, f, np.abs(Zxx_3D[:, :, i]), vmin=vmin, vmax=vmax_in_range)
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
        plt.clf()
        #plt.plot(x+time_offset, y[6, :], label="MP7(CS)")
        #plt.plot(x+time_offset, y[7, :], label="MP8(CS)")
        #plt.xlabel("Time [sec]")
        #plt.legend()
        #plt.show()

def make_stft_profile(date):
    r_pol = np.array([379, 432, 484, 535, 583, 630, 689, 745, 820])
    #num_shots = np.array([97, 68, 69, 70, 71, 72, 73, 74, 75])      #For 23Dec2017
    num_shots = np.array([87, 54, 89, 91, 93, 95, 97, 99, 101])    #For 23Feb2018

    for i in range(9):
        stft = STFT_RT1(date=date, shotNo=num_shots[i], LOCALorPPL="PPL")
        #f, t, Zxx_3D,_,_,_,_,_,_,_,_ = stft.stft(IForMPorSX="POL", num_ch=4)
        f, t, Zxx_3D,_,_,_,_,_,_,_,y = stft.stft(IForMPorSX="POL_RATIO", num_ch=2)
        if i == 0:
            y_profile = np.zeros((2, y[0].__len__(), 9))
        y_profile[:, :, i] = y
        if(i == 0):
            #Zxx_4D = np.zeros((np.shape(Zxx_3D)[0], np.shape(Zxx_3D)[1], np.shape(Zxx_3D)[2], r_pol.__len__()), dtype=complex)
            Zxx_4D = np.zeros((np.shape(Zxx_3D)[0], np.shape(Zxx_3D)[1], np.shape(Zxx_3D)[2], r_pol.__len__()))
        Zxx_4D[:, :, :, i] = Zxx_3D

    #filename = 'Pol_ratio_woffset_stft_%s_%dto%d.npz' % (date, num_shots[0], num_shots[-1])
    #np.savez_compressed(filename, r_pol=r_pol, f=f, t=t, Zxx_4D=Zxx_4D)

    N = 10000
    f, Cxy = sig.coherence(y_profile[1, 15000:16000, 8], y_profile[1, 15000:16000, 2], N, nperseg=2**7)
    plt.plot(f, Cxy)
    plt.show()

if __name__ == "__main__":
    #for i in range(46,105):
    #    stft = STFT_RT1(date="20180622", shotNo=i, LOCALorPPL="PPL")
    #    stft.cross_spectrum()
    #    stft.plot_stft(IForMPorSX="IF", num_ch=3)
    #    stft.plot_stft(IForMPorSX="MP", num_ch=4)
    #    stft.plot_stft(IForMPorSX="POL", num_ch=4)
    #    stft.plot_stft(IForMPorSX="POL_RATIO", num_ch=2)
    stft = STFT_RT1(date="20180921", shotNo=13, LOCALorPPL="PPL")
    #stft.phase_diff()
    #stft.plot_stft(IForMPorSX="IF", num_ch=3)
    #f, t,_,_,_,_,_,_,time_offset_stft, x, y = stft.stft(IForMPorSX="MP", num_ch=4)
    #stft.phase_diff(x, y, f, t, time_offset_stft)
    stft.plot_stft(IForMPorSX="MP", num_ch=8)
    #make_stft_profile(date="20180223")
    #stft.cross_spectrum()
