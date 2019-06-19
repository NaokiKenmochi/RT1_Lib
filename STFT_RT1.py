from RT1DataBrowser import DataBrowser
from matplotlib import gridspec, rc
from matplotlib import mlab
from scipy.fftpack import fft
import sys
#sys.path.append('/Users/kemmochi/PycharmProjects/ControlCosmoZ')

import numpy as np
import pywt
import read_wvf
import copy
import os
#import czdec
import scipy.signal as sig
import matplotlib.pyplot as plt
import matplotlib.ticker
import scipy.optimize
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

    def cross_spectrum(self, name_data1, name_data2, chnum_data1, chnum_data2, vmax):
        #data_ep01 = self.load_ep01("PPL")
        #data_ep01 = self.adj_gain(data_ep01)
        #data_ep01 = self.calib_IF(data_ep01)
        #MP_FAST = self.load_MP_FAST("PPL")
        #IF_FAST = self.load_IF_FAST("PPL")
        ##IF = data_ep01[11:13:1, :].T
        #array_2data = data_ep01[11:13:1, :].T
        ##IF[:,0] *= -1
        #if name_data1=="MP" and name_data2=="MP":
        #    array_2data= np.zeros((28000, 2))
        #    array_2data[:, 0] = MP_FAST[chnum_data1, 10500:38500:1].T
        #    array_2data[:, 1] = MP_FAST[chnum_data2, 10500:38500:1].T
        #    N = 2.0e4 #IF_FAST
        ##IF_MP[:, 0] = data_ep01[10, 8000:22000].T
        ##IF_MP[:, 1] = data_ep01[12, 8000:22000].T
        ##IF_MP[:, 1] = MP_FAST[3, 265000:965000:50].T

        fs = 2e4
        N = 2.8e4
        time = np.arange(N)/float(fs)
        x1 = np.sin(2*np.pi*600*time)
        x2 = np.sin(2*np.pi*600*time-1*np.pi/1)

        IF_MP = np.zeros((N, 2))
        IF_MP[:, 0] = x1[::1]
        IF_MP[:, 1] = x2[::1]
        array_2data = IF_MP
        #IF = IF_FAST[1:4:2, :].T
        #IF = data_ep01[11:13, :].T
        #N = 2*np.abs(1/(data_ep01[0, 1]-data_ep01[0, 2]))
        #N = np.abs(1/(data_ep01[0, 1]-data_ep01[0, 2]))
        sampling_time = 1/N
        plt.plot(time, array_2data[:, 0], label=name_data1+str(chnum_data1) + " and 1" + name_data2+str(chnum_data2))
        plt.plot(time, array_2data[:, 1], label=name_data1+str(chnum_data1) + " and " + name_data2+str(chnum_data2))
        plt.xlim(0, 0.01)
        plt.legend()
        plt.show()

        #sampling_time = 1e-6
        #f, t, Pxx = sig.spectrogram(IF, axis=0, fs=1/sampling_time, window='hamming', nperseg=128, noverlap=64, mode='complex')
        #f, t, Pxx = sig.spectrogram(IF, axis=0, fs=1/sampling_time, window='hamming', nperseg=2**15, noverlap=512, mode='complex')
        f, t, Pxx = sig.spectrogram(array_2data, axis=0, fs=1/sampling_time, window='hamming', nperseg=2**8, noverlap=16, mode='complex')    #MP
        #f, t, Pxx = sig.spectrogram(IF, axis=0, fs=1/sampling_time, window='hamming', nperseg=2**14, noverlap=16, mode='complex')    #IF_FAST
        Pxx_run = self.moving_average(Pxx[:, 0] * np.conj(Pxx[:, 1]), 8)
        #weight = Pxx_run
        weight = np.where(np.log(np.abs(Pxx_run)) > vmax-0.5, 1, 0)

        #２列目の位相が進んでいる場合にDPhaseは正になる
        DPhase = 180/np.pi*np.arctan2(Pxx_run.imag, Pxx_run.real)

        #plt.pcolormesh(t, f, np.abs(Pxx[:, 0]))
        plt.subplot(211)
        plt.title("Date: %s, Shot No.: %d" % (self.date, self.shotnum), loc='right', fontsize=16, fontname="Times New Roman")
        #plt.pcolormesh(t, f, np.log(np.abs(Pxx_run)), vmin=-18.5, vmax=-17)
        #plt.pcolormesh(t+0.7, f, np.log(np.abs(Pxx_run)), vmin=-13, vmax=-12)
        plt.pcolormesh(t+0.7, f, np.log(np.abs(Pxx_run)), vmin=vmax-1, vmax=vmax)
        #plt.pcolormesh(t+0.7, f, np.log(np.abs(Pxx_run)))
        plt.ylim(0, 2000)
        plt.xlim(0.5, 2.5)
        plt.ylabel("Cross-Spectrum \nb/w" + name_data1+str(chnum_data1) + " and " + name_data2+str(chnum_data2) + "\nFrequency [Hz]")
        plt.colorbar()
        plt.subplot(212)
        plt.pcolormesh(t, f, DPhase*weight, cmap='bwr', vmin=-180, vmax=180)
        #plt.pcolormesh(t+0.7, f, DPhase*weight, cmap='bwr', vmin=-180, vmax=180)
        #plt.pcolormesh(t+0.8, f, 120/(DPhase*weight), cmap='Set1', vmin=-8, vmax=8)
        #plt.pcolormesh(t+0.8, f, weight)
        plt.ylim(0, 2000)
        plt.xlim(0.5, 2.5)
        plt.colorbar()
        plt.xlabel('Time [sec]')
        plt.ylabel("Phase Difference \nb/w" + name_data1+str(chnum_data1) + " and " + name_data2+str(chnum_data2) + "\nFrequency [Hz]")
        filepath = "figure/"
        filename = "CSD_PD_bw_%s%d_and_%s%d_%s_%d" % (name_data1, chnum_data1, name_data2, chnum_data2, self.date, self.shotnum)
        plt.tight_layout()
        #plt.savefig(filepath + filename, format='png', dpi=600)
        plt.show()
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

    def set_Pech(self):

        Pech = np.zeros(120)
        Pech[18:24] = 6
        Pech[24:26] = 8
        Pech[26:29] = 10
        Pech[29:32] = 12
        Pech[32:35] = 14
        Pech[35:38] = 16
        Pech[38:69] = 18
        Pech[69:71] = 6
        Pech[71:73] = 8
        Pech[73:76] = 10
        Pech[76:79] = 12
        Pech[79:82] = 14
        Pech[82:85] = 16
        Pech[85:88] = 18
        Pech[88:90] = 5
        Pech[90:93] = 7
        Pech[93:96] = 9
        Pech[96:99] = 11
        Pech[99:102] = 13
        Pech[102:105] = 15
        Pech[105:108] = 17
        Pech[108:112] = 18

        return Pech

    def set_pulse_width(self):
        pulse_width = np.zeros(120)
        pulse_width[18:41] = 5
        pulse_width[41:44] = 8
        pulse_width[44:47] = 10
        pulse_width[47:50] = 12
        pulse_width[50:53] = 14
        pulse_width[53:56] = 16
        pulse_width[56:59] = 18
        pulse_width[59:62] = 20
        pulse_width[62:65] = 22
        pulse_width[65:68] = 24
        pulse_width[68:88] = 10
        pulse_width[88:108] = 5
        pulse_width[108:111] = 20

        return pulse_width


    def cross_spectrum_MParray(self, vmax, freq_border, freq_border_2=None, num_ch=7):
        rc('text', usetex=True)
        MP_FAST = self.load_MP_FAST("PPL")

        if freq_border_2==None:
            freq_border_2=freq_border

        #fpn = "Parameters_Errors_all_ltfqb_mid_gtfqb_fqb12_%s" % self.date
        #fpn = "Parameters_Errors_all_ltfqb_mid_gtfqb_fqb12_pw5ms_MP1to%d_%s" % (num_ch, self.date)
        fpn = "Parameters_Errors_all_ltfqb_mid_gtfqb_fqb12_pw10ms_MP1to%d_%s" % (num_ch, self.date)
        #fpn = "Parameters_Errors_all_ltfqb_mid_gtfqb_fqb12_Pech18kW_MP1to%d_%s" % (num_ch, self.date)
        #fpn = "Parameters_Errors_all_ltfqb_gtfqb_fqb_%s" % self.date
        if self.fileCheck(fpn + ".npz") == 'true':
            Parameters_Errors = np.load(fpn + '.npz')
            Parameters_shtNo_all_ltfqb_mid_gtfqb = Parameters_Errors['parameters']
            Errors_shtNo_all_ltfqb_mid_gtfqb = Parameters_Errors['errors']
        else:
            Parameters_shtNo_all_ltfqb_mid_gtfqb = np.zeros((120, 11))
            Errors_shtNo_all_ltfqb_mid_gtfqb = np.zeros((120, 11))


        Parameters_shtNo_all_ltfqb_mid_gtfqb[self.shotnum, 0] = self.shotnum
        Errors_shtNo_all_ltfqb_mid_gtfqb[self.shotnum, 0] = self.shotnum

        num_st = 5650
        num_ed = num_st + 22000
        array_2data = MP_FAST[1:7, num_st:num_ed:1].T
        #array_2data = array_2data_buf[:, [3,1,2,0,4,5]]
        #array_2data = MP_FAST[1:7, 10500:38500:1].T
        #plt.plot(array_2data[:, 0], label='MP1')
        #plt.plot(array_2data[:, 1], label='MP2')
        #plt.plot(array_2data[:, 2], label='MP3')
        #plt.plot(array_2data[:, 3], label='MP4')
        #plt.legend()
        #plt.show()
        #array_2data[:, 2] *= -1
        #array_2data[:, 4:] *= -1
        N = np.abs(1/(MP_FAST[0, 1]-MP_FAST[0, 2]))
        sampling_time = 1/N

        array_offset = np.arange(6)

        #average_phase = [0]
        #average_phase = [-179.999]
        #average_phase_lt_fqb = [-179.999]
        #average_phase_mid = [-179.999]
        #average_phase_gt_fqb = [-179.999]
        #average_phase = [-179.999]
        average_phase = [0.001]
        average_phase_lt_fqb = [0.001]
        average_phase_mid = [0.001]
        average_phase_gt_fqb = [0.001]

        plt.figure(figsize=(12, 16))
        for i in range(5):
            f, t, Pxx = sig.spectrogram(array_2data[:, :i+2:i+1], axis=0, fs=1/sampling_time, window='hamming', nperseg=2**8, noverlap=16, mode='complex')    #MP
            Pxx_run = self.moving_average(Pxx[:, 0] * np.conj(Pxx[:, 1]), 8)
            weight = np.where(np.log(np.abs(Pxx_run)) > vmax[i]-0.5, 1, 0)
            weight[:, :32] = 0
            weight_gt_fqb = copy.copy(weight)
            weight_mid = copy.copy(weight)
            weight_lt_fqb = copy.copy(weight)
            weight_gt_fqb[:freq_border_2, :] = 0
            weight_mid[:freq_border] = weight_mid[freq_border_2:] = 0
            weight_lt_fqb[freq_border:, :] = 0

            #２列目の位相が進んでいる場合にDPhaseは正になる
            DPhase = 180/np.pi*np.arctan2(Pxx_run.imag, Pxx_run.real) + 180

            plt.subplot(7, 2, 2*i+1)
            #plt.pcolormesh(t+0.7, f, np.log(np.abs(Pxx_run)), vmin=vmax[i]-1, vmax=vmax[i])
            plt.pcolormesh(t+0.9, f, np.log(np.abs(Pxx_run)), vmin=vmax[i]-1, vmax=vmax[i])
            plt.ylim(0, 2000)
            #plt.xlim(0.5, 2.5)
            plt.xlim(1, 2)
            plt.ylabel("Cross-Spectrum \nb/w MP1" + " and MP" +str(i+2) + "\nFrequency [Hz]")
            plt.xlabel('Time [sec]')
            plt.colorbar()
            plt.subplot(7, 2, 2*i+2)
            if i==0:
                plt.title("Date: %s, Shot No.: %d" % (self.date, self.shotnum), loc='right', fontsize=16, fontname="Times New Roman")
            #plt.pcolormesh(t+0.7, f, DPhase*weight, cmap='bwr', vmin=-180, vmax=180)
            plt.pcolormesh(t+0.9, f, DPhase*weight, cmap='hot_r', vmin=0, vmax=360)
            #plt.hlines(781.25, 0.5, 2.5, linestyles='dotted')
            plt.hlines(f[freq_border], 0.5, 2.5, linestyles='dotted')
            plt.hlines(f[freq_border_2], 0.5, 2.5, linestyles='dotted')
            plt.ylim(0, 2000)
            #plt.xlim(0.5, 2.5)
            plt.xlim(1, 2)
            plt.colorbar()
            plt.ylabel("Phase Difference \nb/w MP1" + " and MP" +str(i+2) + "\nFrequency [Hz]")
            plt.xlabel('Time [sec]')
            average_phase.append(np.sum(DPhase*weight)/np.sum(weight))
            average_phase_gt_fqb.append(np.sum(DPhase*weight_gt_fqb)/np.sum(weight_gt_fqb))
            average_phase_mid.append(np.sum(DPhase*weight_mid)/np.sum(weight_mid))
            average_phase_lt_fqb.append(np.sum(DPhase*weight_lt_fqb)/np.sum(weight_lt_fqb))

        #average_phase[2] = 179.999
        #average_phase_lt_fqb[2] = 179.999
        #average_phase_mid[2] = 179.999
        #average_phase_gt_fqb[2] = 179.999

        plt.subplot(7, 1, 7)
        plt.plot(array_2data + array_offset + 1)#, label=name_data1+str(chnum_data1) + " and " + name_data2+str(chnum_data2))
        plt.ylabel('Raw data \n(MP1to6)')
        plt.subplot(7, 1, 6)
        #average_phase_array = np.where((np.array(average_phase) > -180) & (np.array(average_phase) < 180), np.array(average_phase), 1000)
        #average_phase_array_lt_fqb = np.where((np.array(average_phase_lt_fqb) > -180) & (np.array(average_phase_lt_fqb) < 180), np.array(average_phase_lt_fqb), 1000)
        #average_phase_array_mid = np.where((np.array(average_phase_mid) > -180) & (np.array(average_phase_mid) < 180), np.array(average_phase_mid), 1000)
        #average_phase_array_gt_fqb= np.where((np.array(average_phase_gt_fqb) > -180) & (np.array(average_phase_gt_fqb) < 180), np.array(average_phase_gt_fqb), 1000)
        average_phase_array = np.where((np.array(average_phase) > 0) & (np.array(average_phase) <360), np.array(average_phase), 1000)
        average_phase_array_lt_fqb = np.where((np.array(average_phase_lt_fqb) > 0) & (np.array(average_phase_lt_fqb) < 360), np.array(average_phase_lt_fqb), 1000)
        average_phase_array_mid = np.where((np.array(average_phase_mid) > 0) & (np.array(average_phase_mid) < 360), np.array(average_phase_mid), 1000)
        average_phase_array_gt_fqb= np.where((np.array(average_phase_gt_fqb) > 0) & (np.array(average_phase_gt_fqb) < 360), np.array(average_phase_gt_fqb), 1000)
        position = np.array([22.5, 37.5, 37.5, 37.5, 52.5, 67.5])
        position_z = np.array([0.0, 100.0, 0.0, -100.0, 0.0, 0.0])
        #position = np.array([22.5, 37.5, 52.5, 67.5, 45, 135, 270])
        #average_phase_array = np.array([90, 156, 156, 90, 180, -180, 0])
        #position = np.array([30, 60, 75, 45, 135, 270])
        position_fitfunc = np.linspace(0, 360, 3600)

        print("===== Results of MP1 to MP%d =====" % num_ch)
        #parameter_initial = np.array([2, 22.5])
        #parameter_optimal, covariance = scipy.optimize.curve_fit(self.fit_func_sin, position[:num_ch], average_phase_array[:num_ch], p0=parameter_initial, bounds=([1, 22.5], [10, 22.6]))
        #Parameters_shtNo_all_ltfqb_mid_gtfqb[self.shotnum, 1:3] = parameter_optimal
        #Errors_shtNo_all_ltfqb_mid_gtfqb[self.shotnum, 1:3] = np.sqrt(np.diag(covariance))
        #print("Parameter(All freq.): a=%.3f, b=%.3f" % (parameter_optimal[0], parameter_optimal[1]))
        ##y = self.fit_func_sin(position_fitfunc, 1, 45)#parameter_optimal[0], parameter_optimal[1])
        #y = self.fit_func_sin(position_fitfunc, parameter_optimal[0], parameter_optimal[1])
        #plt.plot(position_fitfunc, y)
        #plt.plot(position[:num_ch], average_phase_array[:num_ch], 'o')
        #plt.text(250, 100, r'$\displaystyle y = -180cos(\frac{2\pi%.3f(x-%.3f)}{360})+180$' % (parameter_optimal[0], parameter_optimal[1]))
        ##plt.hlines(0, xmin=0, xmax=360,linestyles='dotted')
        plt.xlabel('Position of MP [degree]')
        plt.ylabel('Phase Difference \nfrom MP1 [degree]')
        #plt.ylim(-180, 180)
        plt.ylim(0, 360)
        plt.xlim(0, 360)

        parameter_initial = np.array([2, 22.5])
        parameter_optimal, covariance = scipy.optimize.curve_fit(self.fit_func_sin, position[:num_ch], average_phase_array_lt_fqb[:num_ch], p0=parameter_initial, bounds=([1, 22.5], [10, 22.6]))
        Parameters_shtNo_all_ltfqb_mid_gtfqb[self.shotnum, 3:5] = parameter_optimal
        Errors_shtNo_all_ltfqb_mid_gtfqb[self.shotnum, 3:5] = np.sqrt(np.diag(covariance))
        print("Parameter (freq.<%dHz): a=%.3f, b=%.3f" % (f[freq_border], parameter_optimal[0], parameter_optimal[1]))
        y = self.fit_func_sin(position_fitfunc, parameter_optimal[0], parameter_optimal[1])
        plt.plot(position_fitfunc, y, label=r'$<$%dHz' % f[freq_border], color='blue')
        plt.plot(position[:num_ch], average_phase_array_lt_fqb[:num_ch], 'b^', label=r'$<$%dHz' % f[freq_border])
        plt.text(180, 270, r'$\displaystyle y(f<%dHz) = -180cos(\frac{2\pi%.3f(x-%.3f)}{360})+180$' % (f[freq_border], parameter_optimal[0], parameter_optimal[1]))

        parameter_initial = np.array([2, 22.5])
        parameter_optimal, covariance = scipy.optimize.curve_fit(self.fit_func_sin, position[:num_ch], average_phase_array_mid[:num_ch], p0=parameter_initial, bounds=([1, 22.5], [10, 22.6]))
        Parameters_shtNo_all_ltfqb_mid_gtfqb[self.shotnum, 5:7] = parameter_optimal
        Errors_shtNo_all_ltfqb_mid_gtfqb[self.shotnum, 5:7] = np.sqrt(np.diag(covariance))
        print("Parameter (%d<freq.<%dHz): a=%.3f, b=%.3f" % (f[freq_border], f[freq_border_2], parameter_optimal[0], parameter_optimal[1]))
        y = self.fit_func_sin(position_fitfunc, parameter_optimal[0], parameter_optimal[1])
        plt.plot(position_fitfunc, y, label='%d-%dHz' % (f[freq_border], f[freq_border_2]), color='red')
        plt.plot(position[:num_ch], average_phase_array_mid[:num_ch], 'ro', label='%d-%dHz' % (f[freq_border], f[freq_border_2]))
        plt.text(180, 180, r'$\displaystyle y(%d<f<%dHz) = -180cos(\frac{2\pi%.3f(x-%.3f)}{360})+180$' % (f[freq_border], f[freq_border_2], parameter_optimal[0], parameter_optimal[1]))

        parameter_initial = np.array([2, 22.5])
        parameter_optimal, covariance = scipy.optimize.curve_fit(self.fit_func_sin, position[:num_ch], average_phase_array_gt_fqb[:num_ch], p0=parameter_initial, bounds=([1, 22.5], [10, 22.6]))
        Parameters_shtNo_all_ltfqb_mid_gtfqb[self.shotnum, 7:9] = parameter_optimal
        Errors_shtNo_all_ltfqb_mid_gtfqb[self.shotnum, 7:9] = np.sqrt(np.diag(covariance))
        print("Parameter (freq.>%dHz): a=%.3f, b=%.3f" % (f[freq_border_2], parameter_optimal[0], parameter_optimal[1]))
        y = self.fit_func_sin(position_fitfunc, parameter_optimal[0], parameter_optimal[1])
        plt.plot(position_fitfunc, y, label=r'$>$%dHz' % f[freq_border_2], color='green')
        plt.plot(position[:num_ch], average_phase_array_gt_fqb[:num_ch], 'gs', label=r'$>$%dHz' % f[freq_border_2])
        plt.text(180, 90, r'$\displaystyle y(f>%dHz) = -180cos(\frac{2\pi%.3f(x-%.3f)}{360})+180$' % (f[freq_border_2], parameter_optimal[0], parameter_optimal[1]))
        plt.legend(bbox_to_anchor=(0.88, 0.95), loc='upper left', borderaxespad=0)

        filepath = "figure/"
        filename = "CSD_PD_bwMP1to7_l%d_mid_h%dHz_MP1to%d_%s_%d" % (f[freq_border], f[freq_border_2], num_ch, self.date, self.shotnum)
        plt.tight_layout()
        #plt.show()
        plt.savefig(filepath + filename + '.png', format='png', dpi=100)
        plt.clf()

        Parameters_shtNo_all_ltfqb_mid_gtfqb[self.shotnum, 9] = f[freq_border]
        Errors_shtNo_all_ltfqb_mid_gtfqb[self.shotnum, 9] = f[freq_border]
        Parameters_shtNo_all_ltfqb_mid_gtfqb[self.shotnum, 10] = f[freq_border_2]
        Errors_shtNo_all_ltfqb_mid_gtfqb[self.shotnum, 10] = f[freq_border_2]
        np.savez(fpn, parameters = Parameters_shtNo_all_ltfqb_mid_gtfqb, errors = Errors_shtNo_all_ltfqb_mid_gtfqb)
        print("Save %s.png" % filename)


        #Pech = self.set_Pech()
        ##pulse_width = self.set_pulse_width()
        ##freq_range = (Parameters_shtNo_all_ltfqb_mid_gtfqb[:, 9] + Parameters_shtNo_all_ltfqb_mid_gtfqb[:, 10])/2
        ##plt.plot(Pech, freq_range)
        ##plt.show()
        ##Pech = self.set_Pech()
        ##plt.plot(Pech, Parameters_shtNo_all_ltfqb_mid_gtfqb[:, 1], '^', label='all freq.')
        ##plt.errorbar(Pech, Parameters_shtNo_all_ltfqb_mid_gtfqb[:, 1], yerr=Errors_shtNo_all_ltfqb_mid_gtfqb[:, 1], fmt='b^', label='all freq.')
        ##plt.plot(Pech, Parameters_shtNo_all_ltfqb_mid_gtfqb[:, 3], 'o', label='ltfqb')
        #plt.errorbar(Pech[:], Parameters_shtNo_all_ltfqb_mid_gtfqb[:, 3], yerr=Errors_shtNo_all_ltfqb_mid_gtfqb[:, 3], fmt='ro', label=r'$freq.<%dHz$' % f[freq_border])
        ##plt.plot(Pech, Parameters_shtNo_all_ltfqb_mid_gtfqb[:, 5], 's', label='gtfqb')
        ##plt.errorbar(Pech, Parameters_shtNo_all_ltfqb_mid_gtfqb[:, 5], yerr=Errors_shtNo_all_ltfqb_mid_gtfqb[:, 5], fmt='gs', label='gtfqb')
        #plt.errorbar(Pech[:], Parameters_shtNo_all_ltfqb_mid_gtfqb[:, 5], yerr=Errors_shtNo_all_ltfqb_mid_gtfqb[:, 5], fmt='gs', label=r'$%d<freq.<%dHz$' % (f[freq_border], f[freq_border_2]))
        #plt.errorbar(Pech[:], Parameters_shtNo_all_ltfqb_mid_gtfqb[:, 7], yerr=Errors_shtNo_all_ltfqb_mid_gtfqb[:, 7], fmt='bs', label=r'$freq.>%dHz$' % f[freq_border_2])
        #plt.legend()
        #plt.ylim(1, 10)
        #plt.xlim(5, 20)
        #plt.xlabel(r'$P_{ECH(inj.)}$ [kW]')
        ##plt.xlabel('Pulse width [msec]')
        #plt.ylabel('')
        #plt.show()

        #plt.xlabel('Toroidal Position of MP [degree]')
        #plt.ylabel('Phase Difference \nfrom MP1 [degree]')
        #plt.ylim(0, 360)
        #plt.xlim(0, 360)

        #parameter_initial = np.array([2, 22.5])
        #parameter_optimal, covariance = scipy.optimize.curve_fit(self.fit_func_sin, position[:num_ch], average_phase_array_mid[:num_ch], p0=parameter_initial, bounds=([1, 22.5], [10, 22.6]))
        #Parameters_shtNo_all_ltfqb_mid_gtfqb[self.shotnum, 5:7] = parameter_optimal
        #Errors_shtNo_all_ltfqb_mid_gtfqb[self.shotnum, 5:7] = np.sqrt(np.diag(covariance))
        #print("Parameter (%d<freq.<%dHz): a=%.3f, b=%.3f" % (f[freq_border], f[freq_border_2], parameter_optimal[0], parameter_optimal[1]))
        #y = self.fit_func_sin(position_fitfunc, parameter_optimal[0], parameter_optimal[1])
        #plt.plot(position_fitfunc-22.5, y+180, label='%d-%dHz' % (f[freq_border], f[freq_border_2]), color='blue', linestyle='dashed')
        #plt.plot(position[:num_ch+1]-22.5, average_phase_array_mid[:num_ch+1]+180, 'ro', label='%d-%dHz' % (f[freq_border], f[freq_border_2]))
        #plt.show()


    def fit_func_sin(self, x, a, b):

        #return -180*np.cos(2*np.pi*a*(x-b)/360)
        return -189*np.cos(2*np.pi*a*(x-b)/360) + 180


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
            time_offset = 0.75
            time_offset_stft = 0.25

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

        time_freq_amp_3D = np.zeros((3, 10, num_ch))
        for i in range(num_ch):

            #f, t, Zxx =sig.spectrogram(y[i, :], fs=N, window='hamming', nperseg=NPERSEG, mode='complex')
            f, t, Zxx =sig.spectrogram(y[i, :], fs=N, window='hamming', nperseg=NPERSEG)
            #f, t, Zxx =sig.stft(y[i, :], fs=N, window='hamming', nperseg=NPERSEG)
            if(i == 0):
                #Zxx_3D = np.zeros((np.shape(Zxx)[0], np.shape(Zxx)[1], num_ch), dtype=complex)
                Zxx_3D = np.zeros((np.shape(Zxx)[0], np.shape(Zxx)[1], num_ch))
            Zxx_3D[:, :, i] = Zxx[:, :]

            time_freq_amp_3D[:,:,i] = self.fft_w_window( y[i, :])

        return f, t, Zxx_3D, filename, vmax, coef_vmax,  vmin, time_offset, time_offset_stft, x, y, time_freq_amp_3D

    def fft_w_window(self, y):
        N = len(y)
        t_st = 1.0
        t_ed = 1.1
        #t = np.linspace(0.275, 2.275, N)
        #plt.plot(t, y)
        #plt.show()

        N_st = np.int((t_st-0.275)*N/2)
        N_ed = np.int((t_ed-0.275)*N/2)
        hammingWindow = np.hamming(N_ed-N_st)
        freqList = np.fft.fftfreq(N_ed - N_st, d=2/N)
        time_freq_amp = np.zeros((3,10))
        for i in range(10):
            y_shorttime = y[N_st+2000*i:N_ed+2000*i]
            #yf = fft(y_shorttime)
            yfw = fft(y_shorttime*hammingWindow)
            Label = 't=%.1f-%.1f' % (t_st+0.1*i, t_ed+0.1*i)
            plt.plot(freqList, np.abs(yfw), label=Label)
            #print(freqList[np.argmax(np.abs(yfw[10:len(yfw)/2]))+10])
            #print(np.max(np.abs(yfw[10:len(yfw)/2])))
            time_freq_amp[0,i] = t_st+0.1*i
            time_freq_amp[1,i] = freqList[np.argmax(np.abs(yfw[10:len(yfw)/2]))+10]
            time_freq_amp[2,i] = np.max(np.abs(yfw[10:len(yfw)/2]))
        #plt.plot(freqList, np.abs(yf), color='red')
        #plt.xlim(0, 2000)
        #plt.legend()
        #plt.show()

        #plt.plot(time_freq_amp[0, :], time_freq_amp[1, :])
        #plt.show()
        #plt.plot(time_freq_amp[0, :], time_freq_amp[2, :])
        #plt.show()

        return time_freq_amp



    def plot_stft(self, IForMPorSX="IF", num_ch=4):
        f, t, Zxx_3D, filename, vmax, coef_vmax, vmin, time_offset, time_offset_stft, x, y, _ = self.stft(IForMPorSX=IForMPorSX, num_ch=num_ch)

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
        array_vmax_in_range = np.zeros(num_ch)
        for i in range(num_ch):
            ax0 = plt.subplot(gs[0:3, i])
            try:
                vmax_in_range = np.max(np.abs(Zxx_3D[idx_fst:idx_fed, idx_tst:idx_ted, i])) * coef_vmax
                array_vmax_in_range[i] = vmax_in_range
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
        plt.plot(x+time_offset, y[7, :], label="MP8")
        plt.plot(x+time_offset, -y[6, :]/10, label="MP7")
        plt.xlabel("Time [sec]")
        plt.legend()
        plt.show()

        return array_vmax_in_range

    def fileCheck(self, fpn):
        comment = ''  # ローカル変数を明示
        m = os.path.isfile(fpn)
        if m:  # 真の場合に実行
            comment = 'true'
        else:
            comment = 'false'

        return comment  # 戻り値

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

def plot_mode_MvsN(m, n):
    line_MvsN = np.zeros((100, 3))#m*n))
    #for i in range(m*n):
    #    line_MvsN[:, i] = np.linspace(0, 2*n*np.pi, 100) + 2*n*np.pi*(i)/m - np.pi*n*n
    line_MvsN[:, 0] = np.linspace(0, 360, 100)
    line_MvsN[:, 1] = np.linspace(0, 360, 100) - 180*(2*n/m)
    line_MvsN[:, 2] = np.linspace(0, 360, 100) + 180*(2-2*n/m)

    plt.plot(np.linspace(0, 360, 100), line_MvsN[:,0], color='red', label='Wave 1')
    plt.plot(np.linspace(0, 360, 100), line_MvsN[:,1], color='blue', label='Wave 2')
    plt.plot(np.linspace(0, 360, 100), line_MvsN[:,2], color='blue', label='Wave 2')
    plt.xlim(0, 360)
    plt.ylim(0, 360)
    plt.xlabel('Toroidal angle [degree]')
    plt.ylabel('[degree]')
    plt.legend()
    plt.show()

def plot_AmpFluc_Wp():
    plt.figure(figsize=(6,4), dpi=300)

    fpn_FreqAmp = 'sn_time_freq_amp_4D_MP1to7_20181107'
    freq_amp = np.load(fpn_FreqAmp + '.npz')
    sn_time_freq_amp_4D = freq_amp['arr_0']

    #fpn = "Parameters_Errors_all_ltfqb_mid_gtfqb_fqb12_%s" % ("20181107")
    #fpn = "Parameters_Errors_all_ltfqb_mid_gtfqb_fqb12_pw5ms_MP1to%d_%s" % (4, "20181107")
    fpn = "Parameters_Errors_all_ltfqb_mid_gtfqb_fqb12_Pech18kW_MP1to%d_%s" % (4, "20181107")
    #fpn = "Parameters_Errors_all_ltfqb_gtfqb_fqb_%s" % self.date
    Parameters_Errors = np.load(fpn + '.npz')
    Parameters_shtNo_all_ltfqb_mid_gtfqb = Parameters_Errors['parameters']
    Errors_shtNo_all_ltfqb_mid_gtfqb = Parameters_Errors['errors']
    array_MP = np.load('vmax_in_range_shotnum_maxMP1to7_20181107.npz')
    array_ShotLog = np.load('RT1_ShotLog_20181107_t12to13.npz')
    vmax_in_range_all = array_MP['arr_0']
    arr_shotnum = array_ShotLog['arr_shotnum']
    PECH_max = array_ShotLog['PECH_max']
    IF_max = array_ShotLog['IF_max']
    VG_max = array_ShotLog['VG_max']
    ml_max = array_ShotLog['ml_max']

    Parameters_shtNo_all_ltfqb_mid_gtfqb[66:68, 9] = Parameters_shtNo_all_ltfqb_mid_gtfqb[65, 9]
    Parameters_shtNo_all_ltfqb_mid_gtfqb[66:68, 10] = Parameters_shtNo_all_ltfqb_mid_gtfqb[65, 10]
    freq_range = (Parameters_shtNo_all_ltfqb_mid_gtfqb[:, 9] + Parameters_shtNo_all_ltfqb_mid_gtfqb[:, 10])/2

    t = np.linspace(1.0, 1.9, 10)
    #plt.plot(t, sn_time_freq_amp_4D[2,:,5,38:41])
    #plt.plot(t, sn_time_freq_amp_4D[2,:,5,105:108])
    #plt.plot(t[1:7], sn_time_freq_amp_4D[1,1:7,5,38:41])
    #plt.plot(t[1:7], sn_time_freq_amp_4D[1,1:7,5,105:108])
    #plt.plot(sn_time_freq_amp_4D[1,:,5,22:41])
    #plt.plot(sn_time_freq_amp_4D[1,:,5,92:108])
    #plt.plot(ml_max[:19, 2], sn_time_freq_amp_4D[1,4,5,22:41], 'bo')
    #plt.plot(ml_max[70:86, 2], sn_time_freq_amp_4D[1,4,5,92:108], 'bo')
    #plt.plot(ml_max[:, 2], (sn_time_freq_amp_4D[1,4,5,22:112] + sn_time_freq_amp_4D[1,3,5,22:112])/2, 'bo')
    #plt.plot(ml_max[:, 2], sn_time_freq_amp_4D[1,1,5,22:112], 'bo')
    #plt.plot(ml_max[:, 2], sn_time_freq_amp_4D[2,2,5,22:112], 'ro')
    #plt.plot(18*np.sqrt(1e-3*ml_max[:, 2]), sn_time_freq_amp_4D[2,2,5,22:112], 'ro')
    plt.plot(0.18*ml_max[:, 2], sn_time_freq_amp_4D[1,2,5,22:112], 'go')
    #plt.plot(ml_max[:, 2], sn_time_freq_amp_4D[1,2,5,22:112], 'go')
    #plt.plot(ml_max[:, 2], sn_time_freq_amp_4D[1,4,5,22:112], 'ro')
    #plt.xlabel('diamagnetism [mWb]')
    #plt.xlabel('Time [sec]')
    #plt.ylabel('Frequency [Hz]')
    plt.xlabel(r'$\beta$')
    plt.ylabel('Amplitude of \nmagnetic fluctuation [a.u.]')
    #plt.ylim(0, 15)
    plt.ylim(600, 1200)
    plt.xlim(0, 0.2)
    plt.show()

    plt.errorbar(ml_max[19:46, 2], Parameters_shtNo_all_ltfqb_mid_gtfqb[41:68, 5], yerr=Errors_shtNo_all_ltfqb_mid_gtfqb[41:68, 5], fmt='bo', label=r'$%dHz<freq.<%dHz$' % (859, 1093))

    fpn = "Parameters_Errors_all_ltfqb_mid_gtfqb_fqb12_pw5ms_MP1to%d_%s" % (4, "20181107")
    Parameters_Errors = np.load(fpn + '.npz')
    Parameters_shtNo_all_ltfqb_mid_gtfqb = Parameters_Errors['parameters']
    Errors_shtNo_all_ltfqb_mid_gtfqb = Parameters_Errors['errors']
    plt.errorbar(ml_max[:19, 2], Parameters_shtNo_all_ltfqb_mid_gtfqb[22:41, 5], yerr=Errors_shtNo_all_ltfqb_mid_gtfqb[22:41, 5], fmt='bo', label=r'$%dHz<freq.<%dHz$' % (859, 1093))
    plt.errorbar(ml_max[70:86, 2], Parameters_shtNo_all_ltfqb_mid_gtfqb[92:108, 5], yerr=Errors_shtNo_all_ltfqb_mid_gtfqb[92:108, 5], fmt='bo', label=r'$%dHz<freq.<%dHz$' % (859, 1093))

    fpn = "Parameters_Errors_all_ltfqb_mid_gtfqb_fqb12_pw10ms_MP1to%d_%s" % (4, "20181107")
    Parameters_Errors = np.load(fpn + '.npz')
    Parameters_shtNo_all_ltfqb_mid_gtfqb = Parameters_Errors['parameters']
    Errors_shtNo_all_ltfqb_mid_gtfqb = Parameters_Errors['errors']
    plt.errorbar(ml_max[46:66, 2], Parameters_shtNo_all_ltfqb_mid_gtfqb[68:88, 5], yerr=Errors_shtNo_all_ltfqb_mid_gtfqb[68:88, 5], fmt='bo', label=r'$%dHz<freq.<%dHz$' % (859, 1093))
    #plt.errorbar(ml_max[:19, 2], Parameters_shtNo_all_ltfqb_mid_gtfqb[22:41, 3], yerr=Errors_shtNo_all_ltfqb_mid_gtfqb[22:41, 3], fmt='ro', label=r'$freq.<%dHz$' % 859)
    #plt.errorbar(ml_max[:19, 2], Parameters_shtNo_all_ltfqb_mid_gtfqb[22:41, 7], yerr=Errors_shtNo_all_ltfqb_mid_gtfqb[22:41, 7], fmt='bs', label=r'$freq.>%dHz$' % 1093)
    #plt.errorbar(ml_max[70:86, 2], Parameters_shtNo_all_ltfqb_mid_gtfqb[92:108, 3], yerr=Errors_shtNo_all_ltfqb_mid_gtfqb[92:108, 3], fmt='ro', label=r'$freq.<%dHz$' % 859)
    #plt.errorbar(ml_max[70:86, 2], Parameters_shtNo_all_ltfqb_mid_gtfqb[92:108, 7], yerr=Errors_shtNo_all_ltfqb_mid_gtfqb[92:108, 7], fmt='bs', label=r'$freq.>%dHz$' % 1093)
    plt.legend()
    plt.ylim(1, 10)
    plt.xlim(0, 1.1)
    plt.xlabel('diamagnetism [mWb]')
    plt.ylabel('Toroidal mode number')
    plt.tight_layout()
    plt.show()

    plt.plot(18*np.sqrt(1e-3*ml_max[:19, 2]), vmax_in_range_all[22:41, 5]*1e6, 'ro')
    plt.plot(18*np.sqrt(1e-3*ml_max[70:86, 2]), vmax_in_range_all[92:108, 5]*1e6, 'ro')
    plt.xlabel(r'$\beta$')
    plt.ylabel('Amplitude of \nmagnetic fluctuation [a.u.]')
    plt.show()

    plt.plot(ml_max[:19, 2]/IF_max[:19, 0]**2, vmax_in_range_all[22:41, 5]*1e6, 'ro')
    plt.plot(ml_max[70:86, 2]/IF_max[70:86, 0]**2, vmax_in_range_all[92:108, 5]*1e6, 'ro')
    #plt.plot(arr_shotnum[:90], vmax_in_range_all[22:112, 5], 'ro')
    plt.xlabel('diamag/IF1**2')
    plt.ylabel('Amplitude of \nmagnetic fluctuation [a.u.]')
    plt.show()

    plt.plot(ml_max[:19, 2]/IF_max[:19, 0], vmax_in_range_all[22:41, 5]*1e6, 'ro')
    plt.plot(ml_max[70:86, 2]/IF_max[70:86, 0], vmax_in_range_all[92:108, 5]*1e6, 'ro')
    #plt.plot(arr_shotnum[:90], vmax_in_range_all[22:112, 5], 'ro')
    plt.xlabel('diamag/IF1')
    plt.ylabel('Amplitude of \nmagnetic fluctuation [a.u.]')
    plt.show()

    plt.plot(ml_max[:19, 2], vmax_in_range_all[22:41, 5]*1e6, 'ro')
    plt.plot(ml_max[70:86, 2], vmax_in_range_all[92:108, 5]*1e6, 'ro')
    #plt.plot(arr_shotnum[:90], vmax_in_range_all[22:112, 5], 'ro')
    plt.xlabel('diamagnetism [mWb]')
    plt.ylabel('Amplitude of \nmagnetic fluctuation [a.u.]')
    plt.show()

    plt.plot(ml_max[19:46, 2]/IF_max[19:46, 0]**2, freq_range[41:68], 'ro')
    fpn = "Parameters_Errors_all_ltfqb_mid_gtfqb_fqb12_pw5ms_MP1to%d_%s" % (4, "20181107")
    Parameters_Errors = np.load(fpn + '.npz')
    Parameters_shtNo_all_ltfqb_mid_gtfqb = Parameters_Errors['parameters']
    freq_range = (Parameters_shtNo_all_ltfqb_mid_gtfqb[:, 9] + Parameters_shtNo_all_ltfqb_mid_gtfqb[:, 10])/2
    plt.plot(ml_max[:19, 2]/IF_max[:19, 0]**2, freq_range[22:41], 'ro')
    plt.plot(ml_max[70:86, 2]/IF_max[70:86, 0]**2, freq_range[92:108], 'ro')

    #plt.plot(arr_shotnum[:90], vmax_in_range_all[22:112, 5], 'ro')
    #plt.xlabel('diamagnetism [mWb]')
    plt.xlabel('diamag/IF1**2')
    plt.ylabel('Amplitude of magnetic fluctuation [a.u.]')
    plt.show()

if __name__ == "__main__":
    #vmax_in_range_all = np.zeros((120, 8))
#    sn_time_freq_amp_4D = np.zeros((3, 10, 7, 120))
#    for i in range(22,112):
#        stft = STFT_RT1(date="20181107", shotNo=i, LOCALorPPL="PPL")
#    #    stft.cross_spectrum()
#    #    stft.plot_stft(IForMPorSX="IF", num_ch=3)
#        _,_,_,_,_,_,_,_,_,_,_, time_freq_amp_3D = stft.stft(IForMPorSX="MP", num_ch=7)
#        sn_time_freq_amp_4D[:,:,:,i] = time_freq_amp_3D
#        print('Now analyzing shot no. %d' % i)
    #    array_vmax_in_range = stft.plot_stft(IForMPorSX="MP", num_ch=7)
    #    vmax_in_range_all[i, 0] = i
    #    vmax_in_range_all[i, 1:] = array_vmax_in_range
    ##    stft.plot_stft(IForMPorSX="POL", num_ch=4)
    ##    stft.plot_stft(IForMPorSX="POL_RATIO", num_ch=2)
    #plot_mode_MvsN(8, 5)
    stft = STFT_RT1(date="20190619", shotNo=39, LOCALorPPL="PPL")
    vmax = np.array([-32/2, -33/2, -33/2, -32/2, -32/2, -34/2])
    fb1=8
    fb2=10
    stft.cross_spectrum_MParray(vmax=vmax, freq_border=fb1, freq_border_2=fb2, num_ch=8)
    #stft.cross_spectrum_MParray(vmax=vmax, freq_border=fb1, freq_border_2=fb2, num_ch=5)
    #stft.cross_spectrum_MParray(vmax=vmax, freq_border=fb1, freq_border_2=fb2, num_ch=4)
    #stft.phase_diff()
    #stft.plot_stft(IForMPorSX="MP", num_ch=8)
    #f, t,_,_,_,_,_,_,time_offset_stft, x, y = stft.stft(IForMPorSX="MP", num_ch=7)
    #stft.phase_diff(x, y, f, t, time_offset_stft)
    #stft.plot_stft(IForMPorSX="MP", num_ch=8)
    #make_stft_profile(date="20180223")
    #stft.cross_spectrum(name_data1="MP", name_data2="MP", chnum_data1=2, chnum_data2=3, vmax=-14)
    #np.savez('vmax_in_range_shotnum_maxMP1to7_20181107', vmax_in_range_all)
    #np.savez('sn_time_freq_amp_4D_MP1to7_20181107', sn_time_freq_amp_4D)

    #plot_AmpFluc_Wp()
