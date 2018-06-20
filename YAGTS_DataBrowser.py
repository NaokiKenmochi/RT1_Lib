from scipy import integrate, interpolate, fftpack
import numpy as np
import matplotlib.pyplot as plt
import pandas
import csv

class YAGTS_DataBrowser:
    def __init__(self, date, shotNo, shotSt):
        self.date = date
        self.shotNo = shotNo
        self.shotSt = shotSt
        #self.filepath = '/Volumes/share/DPO4054B/' + str(self.date) + '/tek' + str(self.shotNo-self.shotSt).zfill(4) + 'ALL.csv'
        self.filepath = '/Volumes/share/DPO4054B/tek' + str(self.shotNo-self.shotSt).zfill(4) + 'ALL.csv'

    def open_with_pandas(self):
        df = pandas.read_csv(self.filepath)
        header = df.columns.values.tolist()
        data = df.values

        return df, header, data

    def open_with_numpy(self):
        data = np.loadtxt(self.filepath, delimiter=',', skiprows=18)
        return data

    def show_graph(self, isIntegrate=False):
        data = self.open_with_numpy()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(1e9*data[:, 0], 5.0e-2*(data[:, 1]-np.mean(data[:4000,1])), label='CH1')
        ax.plot(1e9*data[:, 0], data[:, 2]-np.mean(data[:4000,2]), label='CH2')
        ax.plot(1e9*data[:, 0], data[:, 3]-np.mean(data[:4000,3]), label='CH3')
        ax.plot(1e9*data[:, 0], data[:, 4]-np.mean(data[:4000,4]), label='CH4')
        ax.legend(loc='lower right')
        max_ch1 = np.max(data[:, 1]-np.mean(data[:4000, 1]))*1.0e3
        min_ch2 = np.min(data[:, 2]-np.mean(data[:4000, 2]))*1.0e3
        min_ch3 = np.min(data[:, 3]-np.mean(data[:4000, 3]))*1.0e3
        min_ch4 = np.min(data[:, 4]-np.mean(data[:4000, 4]))*1.0e3
        ax.text(-1500, 5.0e-5*max_ch1, 'CH1: %.3f mV' % (max_ch1))
        ax.text(-1500, 1.0e-3*min_ch2, 'CH2: %.3f mV' % min_ch2)
        ax.text(-1500, 1.0e-3*min_ch3, 'CH3: %.3f mV' % min_ch3)
        ax.text(-1500, 1.0e-3*min_ch4, 'CH4: %.3f mV' % min_ch4)
        ax.text(0.75, 0.45, 'CH2/CH3: %.3f' % (min_ch2/min_ch3), transform=ax.transAxes)
        ax.text(0.75, 0.4, 'CH2/CH4: %.3f' % (min_ch2/min_ch4), transform=ax.transAxes)
        ax.text(0.75, 0.35, 'CH3/CH4: %.3f' % (min_ch3/min_ch4), transform=ax.transAxes)
        plt.title("Date: %d, Shot No.: %d" % (self.date,self.shotNo), loc='right', fontsize=20, fontname="Times New Roman")
        print('CH1: %.5f V' % max_ch1)
        print('CH2: %.5f V' % min_ch2)
        print('CH3: %.5f V' % min_ch3)
        print('CH4: %.5f V' % min_ch4)
        #plt.plot(1e9*data[:, 0], data[:, 2])
        #plt.plot(1e9*data[:, 0], data[:, 3])
        #plt.plot(1e9*data[:, 0], data[:, 4])
        plt.xlim(-2000, 2000)
        plt.xlabel('Time [nsec]')
        plt.ylabel('Output[V]')
        filepath = "figure/"
        filename = "YAGTS_%d_%d" % (self.date, self.shotNo)
        plt.savefig(filepath + filename)
        plt.clf()

    def plot_shotlog(self, num_st, num_ed):
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['xtick.top'] = 'True'
        plt.rcParams['ytick.right'] = 'True'
        plt.rcParams['ytick.direction'] = 'in'

        filename = "YAGTS_log_%d_%d_%d.npz" % (self.date, num_st, num_ed)
        shot_log = np.load(filename)
        file_num = shot_log['file_num']
        max_ch1 = np.array(shot_log['max_ch1'])
        min_ch2 = np.array(shot_log['min_ch2'])
        min_ch3 = np.array(shot_log['min_ch3'])
        min_ch4 = np.array(shot_log['min_ch4'])
        #shot_list_1 = np.array([30, 53, 61, 64, 66, 68, 69, 72, 73, 75, 76, 78, 91, 92, 94, 96, 97, 106])
        #pressure_mPa_1 = np.array([3, 3, 3, 3, 3, 3, 3, 0.7, 0.7, 0.7, 0.7, 5, 7, 7, 7, 7, 7, 0.7])
        #shot_list_2 = np.arange(32, 56)
        #pressure_mPa_2 = np.zeros(56-32)
        #shot_list = np.r_[shot_list_1, shot_list_2]
        #pressure_mPa = np.r_[pressure_mPa_1, pressure_mPa_2]
        pressure_mPa = np.zeros(120)
        pressure_mPa[2:11] = -2.0
        pressure_mPa[14:24] = 1.0
        pressure_mPa[24:27] = 2.0
        pressure_mPa[27:29] = 3.0
        pressure_mPa[30:32] = 3.0
        pressure_mPa[56:70] = 3.0
        pressure_mPa[70:77] = 0.7
        pressure_mPa[77:85] = 5.0
        pressure_mPa[85:104] = 7.0
        pressure_mPa[104:121] = 0.7
        pressure_mPa[98:100] = -1.0
        pressure_mPa[108:118] = -1.0
        pressure_mPa[101:103] = 0.0
        pressure_mPa[112] = -1.0
        pressure_mPa[0] = -1.0
        pressure_mPa[1] = -1.0
        pressure_mPa[12] = -1.0
        #shot_list = np.array([30, 53, 61, 66, 68, 72, 73, 75, 76, 78, 91, 92, 94, 96, 97, 106])
        #pressure_mPa = np.array([3, 3, 3, 3, 3, 0.7, 0.7, 0.7, 0.7, 5, 7, 7, 7, 7, 7, 0.7])
        #shot_list = np.array([78, 91, 92, 96, 97, 106])
        #for i,x in enumerate(shot_list):
        #    #plt.plot(file_num[x], min_ch2[x]/min_ch3[x], "o", color='red')
        #    #plt.plot(file_num[x], min_ch2[x]/min_ch4[x], "x", color='blue')
        #    #plt.plot(file_num[x], min_ch3[x]/min_ch4[x], "^", color='green')
        #    if(min_ch2[i] < -15):
        #        plt.plot(pressure_mPa[i], min_ch2[x]/min_ch3[x], "o", color='red', label='ch2/ch3')
        #        plt.plot(pressure_mPa[i], min_ch2[x]/min_ch4[x], "x", color='blue', label='ch2/ch4')
        #        plt.plot(pressure_mPa[i], min_ch3[x]/min_ch4[x], "^", color='green', label='ch3/ch4')
        #threshold = -15
        #for i in range(120):
        #    if(min_ch2[i] < threshold):
        #        plt.plot(pressure_mPa[i], min_ch2[i]/min_ch3[i], "o", color='red', label='ch2/ch3')
        #        plt.plot(pressure_mPa[i], min_ch2[i]/min_ch4[i], "x", color='blue', label='ch2/ch4')
        #        plt.plot(pressure_mPa[i], min_ch3[i]/min_ch4[i], "^", color='green', label='ch3/ch4')
        #        #plt.plot(i, min_ch2[i]/min_ch3[i], "o", color='red', label='ch2/ch3')
        #        #plt.plot(i, min_ch2[i]/min_ch4[i], "x", color='blue', label='ch2/ch4')
        #        #plt.plot(i, min_ch3[i]/min_ch4[i], "^", color='green', label='ch3/ch4')
        #plt.plot(min_ch2, color='red', label='ch2')
        #plt.plot(min_ch3, color='blue', label='ch3')
        #plt.plot(min_ch4, color='green', label='ch4')
        plt.plot(min_ch2/min_ch3, color='red', label='ch2/ch3')
        plt.plot(min_ch2/min_ch4, color='blue', label='ch2/ch4')
        plt.plot(min_ch3/min_ch4, color='green', label='ch3/ch4')
        plt.xlabel("shot No.")
        #plt.title("ch2 < %d" % threshold, loc='right')
        #plt.xlabel("Pressure [mPa]")
        plt.ylabel("Ratio")
        #plt.ylabel("Signal [mV]")
        #plt.xlim(-0.5, 8)
        #plt.ylim(0, 8)
        plt.legend()
        plt.show()


def make_shotlog(date, num_st, num_ed):
    file_num = []
    max_ch1 = []
    min_ch2 = []
    min_ch3 = []
    min_ch4 = []

    for i in range(num_st, num_ed):
        file_num.append(i)
        ytdb = YAGTS_DataBrowser(date=date, shotNo=i, shotSt=0)
        print('Load' + '/tek' + str(i).zfill(4))
        data = ytdb.open_with_numpy()
        max_ch1.append(np.max(data[:, 1]-np.mean(data[:4000, 1]))*1.0e3)
        min_ch2.append(np.min(data[:, 2]-np.mean(data[:4000, 2]))*1.0e3)
        min_ch3.append(np.min(data[:, 3]-np.mean(data[:4000, 3]))*1.0e3)
        min_ch4.append(np.min(data[:, 4]-np.mean(data[:4000, 4]))*1.0e3)
    filename = "YAGTS_log_%d_%d_%d" % (date, num_st, num_ed)
    np.savez(filename, file_num=file_num, max_ch1=max_ch1, min_ch2=min_ch2, min_ch3=min_ch3, min_ch4=min_ch4)

def integrate_SL(date, num_st, num_ed, isSerial=True):
    st_integrate = 4000
    #shot_list = np.array([12, 15, 16, 17, 64])
    shot_list = np.arange(65, 71)
    #shot_list = np.r_[shot_list, np.arange(29, 32)]
    #shot_list = np.arange(9, 12)
    #shot_list = np.r_[shot_list, np.arange(18, 29)]
    #shot_list = np.r_[shot_list, np.arange(32, 50)]
    #shot_list = np.r_[np.arange(59, 61), np.arange(55, 59)]
    #shot_list = np.arange(85, 88)
    if isSerial == True:
        for i in range(num_st, num_ed):
            ytdb = YAGTS_DataBrowser(date=date, shotNo=i, shotSt=0)
            print('Load' + ' tek' + str(i).zfill(4))
            data = ytdb.open_with_numpy()
            if i == num_st:
                data_integrated = np.zeros(np.shape(data))
            data_integrated += data

        data = data_integrated/(num_ed-num_st)
    elif isSerial == False:
        for i,x in enumerate(shot_list):
            ytdb = YAGTS_DataBrowser(date=date, shotNo=x, shotSt=0)
            print('Load' + ' tek' + str(x).zfill(4))
            data = ytdb.open_with_numpy()
            if i == 0:
                data_integrated = np.zeros(np.shape(data))
            data_integrated += data

        data = data_integrated/shot_list.size

    data[:, 1:] -= np.mean(data[:st_integrate, 1:], axis=0)
    #plt.plot(data_integrated)
    #plt.show()
    data_integrate_cumtrapz = integrate.cumtrapz(data[st_integrate:st_integrate+2**13+1, :], axis=0)
    print("ch1[1500]: %.5f" % data_integrate_cumtrapz[5100-st_integrate, 1])
    print("min. ch2: %.5f" % np.min(data_integrate_cumtrapz[:, 2]))
    print("min. ch3: %.5f" % np.min(data_integrate_cumtrapz[:, 3]))
    print("min. ch4: %.5f" % np.min(data_integrate_cumtrapz[:, 4]))
    print("ch2/LP: %.5f" % (np.min(data_integrate_cumtrapz[:, 2])/data_integrate_cumtrapz[5100-st_integrate, 1]))
    print("ch3/LP: %.5f" % (np.min(data_integrate_cumtrapz[:, 3])/data_integrate_cumtrapz[5100-st_integrate, 1]))
    print("ch4/LP: %.5f" % (np.min(data_integrate_cumtrapz[:, 4])/data_integrate_cumtrapz[5100-st_integrate, 1]))

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['xtick.top'] = 'True'
    plt.rcParams['ytick.right'] = 'True'
    plt.rcParams['ytick.direction'] = 'in'
    #plt.plot(1e9*data[st_integrate+1:st_integrate+2**13,0], data_integrate_cumtrapz/data_integrate_cumtrapz[5100-st_integrate, 1])
    #plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(1e9*data[:, 0], 1.0e-3*(data[:, 1]), label='CH1')
    ax.plot(1e9*data[:, 0], data[:, 2], label='CH2')
    ax.plot(1e9*data[:, 0], data[:, 3], label='CH3')
    ax.plot(1e9*data[:, 0], data[:, 4], label='CH4')
    ax.legend(loc='lower right')
    max_ch1 = np.max(data[:, 1])*1.0e3
    min_ch2 = np.min(data[:, 2])*1.0e3
    min_ch3 = np.min(data[:, 3])*1.0e3
    min_ch4 = np.min(data[:, 4])*1.0e3
    ax.text(-1500, 1.0e-6*max_ch1, 'CH1: %.3f mV' % (max_ch1))
    ax.text(-1500, 1.0e-3*min_ch2, 'CH2: %.3f mV' % min_ch2)
    ax.text(-1500, 1.0e-3*min_ch3, 'CH3: %.3f mV' % min_ch3)
    ax.text(-1500, 1.0e-3*min_ch4, 'CH4: %.3f mV' % min_ch4)
    ax.text(0.75, 0.90, 'CH2/LP: %.5f' % (min_ch2/max_ch1), transform=ax.transAxes)
    ax.text(0.75, 0.85, 'CH3/LP: %.5f' % (min_ch3/max_ch1), transform=ax.transAxes)
    ax.text(0.75, 0.80, 'CH4/LP: %.5f' % (min_ch4/max_ch1), transform=ax.transAxes)
    ax.text(0.75, 0.75, 'CH2/CH3: %.3f' % (min_ch2/min_ch3), transform=ax.transAxes)
    ax.text(0.75, 0.70, 'CH2/CH4: %.3f' % (min_ch2/min_ch4), transform=ax.transAxes)
    ax.text(0.75, 0.65, 'CH3/CH4: %.3f' % (min_ch3/min_ch4), transform=ax.transAxes)
    print('CH1: %.5f V' % max_ch1)
    print('CH2: %.5f V' % min_ch2)
    print('CH3: %.5f V' % min_ch3)
    print('CH4: %.5f V' % min_ch4)
    print('CH1/LP: %.5f V' % (max_ch1/max_ch1))
    print('CH2/LP: %.5f V' % (min_ch2/max_ch1))
    print('CH3/LP: %.5f V' % (min_ch3/max_ch1))
    print('CH4/LP: %.5f V' % (min_ch4/max_ch1))
    if isSerial == True:
        plt.title("Date: %d, File No.: %d - %d" % (date, num_st, num_ed-1), loc='right', fontsize=20, fontname="Times New Roman")
        filename = "YAGTS_integrated_%d_FileNo%dto%d" % (date, num_st, num_ed-1)
    else:
        plt.title("Date: %d, File No.: %d - %d" % (date, shot_list[0], shot_list[-1]), loc='right', fontsize=20, fontname="Times New Roman")
        filename = "YAGTS_integrated_%d_FileNo%dto%d_discrete_%dshots" % (date, shot_list[0], shot_list[-1], shot_list.size)

    plt.xlim(-2000, 2000)
    plt.xlabel('Time [nsec]')
    plt.ylabel('Output[V]')
    fig.tight_layout()
    #plt.show()
    filepath = "figure/"
    plt.savefig(filepath + filename)
    plt.clf()

    filename = "YAGTS_integrated_%d_FileNo%dto%d_discrete_%dshots.npz" % (date, shot_list[0], shot_list[-1], shot_list.size)
    np.savez(filename, shot_list=shot_list, data=data)

def subtract_straylight():
    st_integrate = 4900
    ed_integrate = 5200
    filename_stray = "YAGTS_integrated_20180328_FileNo12to31_discrete_6shots.npz"   #6Pa
    filename_plasma = "YAGTS_integrated_20180328_FileNo9to49_discrete_32shots.npz"  #6Pa
    #filename_stray = "YAGTS_integrated_20180329_FileNo85to87_discrete_3shots.npz"   #へそ外し
    #filename_stray = "YAGTS_integrated_20180329_FileNo80to84_discrete_5shots.npz"   #へそ外し
    #filename_stray = "YAGTS_integrated_20180329_FileNo15to49_discrete_6shots.npz"
    #filename_stray = "YAGTS_integrated_20180329_FileNo47to49_discrete_3shots.npz"
    #filename_plasma = "YAGTS_integrated_20180329_FileNo18to46_discrete_29shots.npz"
    #filename_plasma = "YAGTS_integrated_20180329_FileNo50to79_discrete_30shots.npz"
    #filename_plasma = "YAGTS_integrated_20180328_FileNo81to95_discrete_8shots.npz"  #6mPa 浮上
    stray = np.load(filename_stray)
    plasma = np.load(filename_plasma)
    data_stray = stray['data']
    data_plasma = plasma['data']
    data_stray_integrate_cumtrapz = integrate.cumtrapz(data_stray[st_integrate:st_integrate+2**13+1, :], axis=0)
    data_plasma_integrate_cumtrapz = integrate.cumtrapz(data_plasma[st_integrate:st_integrate+2**13+1, :], axis=0)
    plt.plot(1e9*data_plasma[st_integrate+1:st_integrate+2**13,0], data_plasma_integrate_cumtrapz[:, 1]/data_plasma_integrate_cumtrapz[5100-st_integrate, 1], color='blue', label='CH1')
    plt.plot(1e9*data_plasma[st_integrate+1:st_integrate+2**13,0], data_plasma_integrate_cumtrapz[:, 2]/data_plasma_integrate_cumtrapz[5100-st_integrate, 1], color='orange', label='CH2')
    plt.plot(1e9*data_plasma[st_integrate+1:st_integrate+2**13,0], data_plasma_integrate_cumtrapz[:, 3]/data_plasma_integrate_cumtrapz[5100-st_integrate, 1], color='green', label='CH3')
    plt.plot(1e9*data_plasma[st_integrate+1:st_integrate+2**13,0], data_plasma_integrate_cumtrapz[:, 4]/data_plasma_integrate_cumtrapz[5100-st_integrate, 1], color='red', label='CH4')
    plt.plot(1e9*data_stray[st_integrate+1:st_integrate+2**13,0], data_stray_integrate_cumtrapz[:, 1]/data_stray_integrate_cumtrapz[5100-st_integrate, 1], linestyle='dashed', color='blue', label='CH1(Stray)')
    plt.plot(1e9*data_stray[st_integrate+1:st_integrate+2**13,0], data_stray_integrate_cumtrapz[:, 2]/data_stray_integrate_cumtrapz[5100-st_integrate, 1], linestyle='dashed', color='orange', label='CH2(Stray)')
    plt.plot(1e9*data_stray[st_integrate+1:st_integrate+2**13,0], data_stray_integrate_cumtrapz[:, 3]/data_stray_integrate_cumtrapz[5100-st_integrate, 1], linestyle='dashed', color='green', label='CH3(Stray)')
    plt.plot(1e9*data_stray[st_integrate+1:st_integrate+2**13,0], data_stray_integrate_cumtrapz[:, 4]/data_stray_integrate_cumtrapz[5100-st_integrate, 1], linestyle='dashed', color='red', label='CH4(Stray)')
    plt.xlabel('Time [nsec]')
    plt.ylabel('Accumulation value [a.u.]')
    plt.legend(loc="lower right")
    plt.title(filename_plasma)
    plt.show()
    data_SL_integrated_ch1 = data_plasma_integrate_cumtrapz[:, 1]/data_plasma_integrate_cumtrapz[5100-st_integrate, 1] - data_stray_integrate_cumtrapz[:, 1]/data_stray_integrate_cumtrapz[5100-st_integrate, 1]
    data_SL_integrated_ch2 = data_plasma_integrate_cumtrapz[:, 2]/data_plasma_integrate_cumtrapz[5100-st_integrate, 1] - data_stray_integrate_cumtrapz[:, 2]/data_stray_integrate_cumtrapz[5100-st_integrate, 1]
    data_SL_integrated_ch3 = data_plasma_integrate_cumtrapz[:, 3]/data_plasma_integrate_cumtrapz[5100-st_integrate, 1] - data_stray_integrate_cumtrapz[:, 3]/data_stray_integrate_cumtrapz[5100-st_integrate, 1]
    data_SL_integrated_ch4 = data_plasma_integrate_cumtrapz[:, 4]/data_plasma_integrate_cumtrapz[5100-st_integrate, 1] - data_stray_integrate_cumtrapz[:, 4]/data_stray_integrate_cumtrapz[5100-st_integrate, 1]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(1e9*data_stray[st_integrate+1:st_integrate+2**13,0], data_SL_integrated_ch1, label='ch1')
    ax.plot(1e9*data_stray[st_integrate+1:st_integrate+2**13,0], data_SL_integrated_ch2, label='ch2')
    ax.plot(1e9*data_stray[st_integrate+1:st_integrate+2**13,0], data_SL_integrated_ch3, label='ch3')
    ax.plot(1e9*data_stray[st_integrate+1:st_integrate+2**13,0], data_SL_integrated_ch4, label='ch4')
    min_ch2 = np.min(data_SL_integrated_ch2[ed_integrate-st_integrate])
    min_ch3 = np.min(data_SL_integrated_ch3[ed_integrate-st_integrate])
    min_ch4 = np.min(data_SL_integrated_ch4[ed_integrate-st_integrate])
    #min_ch2 = np.min(data_SL_integrated_ch2)
    #min_ch3 = np.min(data_SL_integrated_ch3)
    #min_ch4 = np.min(data_SL_integrated_ch4)
    ax.text(0.75, 0.90, 'CH2/LP: %.5f' % (min_ch2), transform=ax.transAxes)
    ax.text(0.75, 0.85, 'CH3/LP: %.5f' % (min_ch3), transform=ax.transAxes)
    ax.text(0.75, 0.80, 'CH4/LP: %.5f' % (min_ch4), transform=ax.transAxes)
    ax.text(0.75, 0.75, 'CH2/CH3: %.3f' % (min_ch2/min_ch3), transform=ax.transAxes)
    ax.text(0.75, 0.70, 'CH2/CH4: %.3f' % (min_ch2/min_ch4), transform=ax.transAxes)
    ax.text(0.75, 0.65, 'CH3/CH4: %.3f' % (min_ch3/min_ch4), transform=ax.transAxes)
    ax.vlines(1e9*data_plasma[ed_integrate, 0], 0, min_ch2*2, linestyles='dashed')
    #plt.title("Date: %d, Shot No.: %d" % (self.date,self.shotNo), loc='right', fontsize=20, fontname="Times New Roman")
    print("Load stray: %s" % filename_stray)
    print("Load plasma: %s" % filename_plasma)
    print('CH2: %.5f V' % min_ch2)
    print('CH3: %.5f V' % min_ch3)
    print('CH4: %.5f V' % min_ch4)
    te_12, te_14, te_24 = get_Te(min_ch2/min_ch4, min_ch2/min_ch3, min_ch4/min_ch3)
    plt.title(filename_plasma)
    plt.xlabel('Time [nsec]')
    plt.ylabel('Integrated Value')
    plt.legend()
    plt.show()
    max_ch1_stray = np.max(data_stray[:, 1]-np.mean(data_stray[:4000, 1]))#*1.0e3
    max_ch1_plasma = np.max(data_plasma[:, 1]-np.mean(data_plasma[:4000, 1]))#*1.0e3
    data_ch2 = max_ch1_plasma*((data_plasma[:, 2]-np.mean(data_plasma[:4000,2]))/max_ch1_plasma - (data_stray[:, 2]-np.mean(data_stray[:4000,2]))/max_ch1_stray)
    #data_ch2 = (data_stray[:, 2]-np.mean(data_stray[:4000,2]))/max_ch1_stray
    data_ch3 = max_ch1_plasma*((data_plasma[:, 3]-np.mean(data_plasma[:4000,3]))/max_ch1_plasma - (data_stray[:, 3]-np.mean(data_stray[:4000,3]))/max_ch1_stray)
    data_ch4 = max_ch1_plasma*((data_plasma[:, 4]-np.mean(data_plasma[:4000,4]))/max_ch1_plasma - (data_stray[:, 4]-np.mean(data_stray[:4000,4]))/max_ch1_stray)
    data = max_ch1_plasma*(data_plasma/max_ch1_plasma - data_stray/max_ch1_stray)
    plt.plot(data)
    plt.show()
    filename_data = filename_plasma[:-4] + "_woStray.txt"
    np.savetxt(filename_data, data, delimiter=',')
    #plt.plot(data_ch2)
    #plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(1e9*data_plasma[:, 0], data_ch2, label='CH2')
    ax.plot(1e9*data_plasma[:, 0], data_ch3, label='CH3')
    ax.plot(1e9*data_plasma[:, 0], data_ch4, label='CH4')
    ax.legend(loc='lower right')
    min_ch2 = np.min(data_ch2-np.mean(data_ch2[:4000]))*1.0e3
    min_ch3 = np.min(data_ch3-np.mean(data_ch3[:4000]))*1.0e3
    min_ch4 = np.min(data_ch4-np.mean(data_ch4[:4000]))*1.0e3
    ax.text(-1500, 1.0e-3*min_ch2, 'CH2: %.3f mV' % min_ch2)
    ax.text(-1500, 1.0e-3*min_ch3, 'CH3: %.3f mV' % min_ch3)
    ax.text(-1500, 1.0e-3*min_ch4, 'CH4: %.3f mV' % min_ch4)
    ax.text(0.75, 0.45, 'CH2/CH3: %.3f' % (min_ch2/min_ch3), transform=ax.transAxes)
    ax.text(0.75, 0.4, 'CH2/CH4: %.3f' % (min_ch2/min_ch4), transform=ax.transAxes)
    ax.text(0.75, 0.35, 'CH3/CH4: %.3f' % (min_ch3/min_ch4), transform=ax.transAxes)
    #plt.title("Date: %d, Shot No.: %d" % (self.date,self.shotNo), loc='right', fontsize=20, fontname="Times New Roman")
    print('CH2: %.5f V' % min_ch2)
    print('CH3: %.5f V' % min_ch3)
    print('CH4: %.5f V' % min_ch4)
    #plt.plot(1e9*data[:, 0], data[:, 2])
    #plt.plot(1e9*data[:, 0], data[:, 3])
    #plt.plot(1e9*data[:, 0], data[:, 4])
    plt.xlim(-2000, 2000)
    plt.xlabel('Time [nsec]')
    plt.ylabel('Output[V]')
    plt.show()
    #filepath = "figure/"
    #filename = "YAGTS_woStray_integrated_20180328_FileNo9to49_discrete_32shots"
    #plt.savefig(filepath + filename)
    num_convolve = 150
    v = np.ones(num_convolve)/num_convolve
    data_ch2_convolved = np.convolve(data_ch2, v, mode='same')
    data_ch3_convolved = np.convolve(data_ch3, v, mode='same')
    data_ch4_convolved = np.convolve(data_ch4, v, mode='same')
    plt.plot(1e9*data_plasma[:, 0], data_ch2_convolved, label='CH2')
    plt.plot(1e9*data_plasma[:, 0], data_ch3_convolved, label='CH3')
    plt.plot(1e9*data_plasma[:, 0], data_ch4_convolved, label='CH4')
    plt.show()

    n = data_ch2.__len__()
    dt = data_plasma[1, 0] - data_plasma[0, 0]
    yf = fftpack.fft(data_ch2)/(n/2)
    freq = fftpack.fftfreq(n, dt)

    fs = 4e7
    yf2 = np.copy(yf)
    yf2[(freq > fs)] = 0
    yf2[(freq < 0)] = 0

    y2 = np.real(fftpack.ifft((yf2)*n))
    plt.plot(1e9*data_plasma[:, 0], y2, label='CH2')
    plt.show()
    plt.clf()

def load_ratio_ptncnt():
    file_path = "Ratio_PtnCnt_P25.txt"
    ratio_ptncnt = np.loadtxt(file_path, delimiter='\t', skiprows=1)
    #plt.plot(ratio_ptncnt[:, 0], ratio_ptncnt[:, 1], label='ch1/ch2')
    #plt.plot(ratio_ptncnt[:, 0], ratio_ptncnt[:, 3], label='ch1/ch4')
    #plt.plot(ratio_ptncnt[:, 0], ratio_ptncnt[:, 6], label='ch2/ch4')
    #plt.legend()
    #plt.ylim(0, 10)
    #plt.xscale("log")
    #plt.show()
    return ratio_ptncnt

def load_cofne():
    num_pol = 25
    num_pnts = 1000
    file_path = "Cofne_Mar2018_HJPol25.txt"
    cofne = np.loadtxt(file_path, delimiter='\t', skiprows=num_pol*num_pnts+(num_pol-1)*num_pnts+1)
    #te = np.linspace(10, 8000, 1000)

    #plt.plot(te, cofne)
    #plt.ylim(0, 5)
    #plt.xscale("log")
    #plt.show()

    return cofne


def get_Te(ratio_12, ratio_14, ratio_24):
    ratio_ptncnt = load_ratio_ptncnt()
    te = np.linspace(1, 9000, 90000-1)
    ratio_ptncnt_spline_1 = interpolate.interp1d(ratio_ptncnt[:, 0], ratio_ptncnt[:, 1])
    ratio_ptncnt_spline_3 = interpolate.interp1d(ratio_ptncnt[:, 0], ratio_ptncnt[:, 3])
    ratio_ptncnt_spline_6 = interpolate.interp1d(ratio_ptncnt[:, 0], ratio_ptncnt[:, 6])
    #plt.plot(te, ratio_ptncnt_spline_1(te), label='ch1/ch2')
    #plt.plot(te, ratio_ptncnt_spline_3(te), label='ch1/ch4')
    #plt.plot(te, ratio_ptncnt_spline_6(te), label='ch2/ch4')
    #plt.legend()
    #plt.ylim(0, 10)
    #plt.xscale("log")
    #plt.show()
    idx_12 = getNearestValue(ratio_ptncnt_spline_1(te[10:]), ratio_12)
    idx_14 = getNearestValue(ratio_ptncnt_spline_3(te[10:]), ratio_14)
    idx_24 = getNearestValue(ratio_ptncnt_spline_6(te[10:]), ratio_24)
    print("Te(ch1/ch2) = %.2f [eV]" % te[10+idx_12])
    print("Te(ch1/ch4) = %.2f [eV]" % te[10+idx_14])
    print("Te(ch2/ch4) = %.2f [eV]" % te[10+idx_24])
    return te[idx_12], te[idx_14], te[idx_24]

def get_ne(Te1_eV, Te4_eV, Te2_eV, output1_mV, output4_mV, output2_mV):
    cofne = load_cofne()
    te = np.linspace(10, 8000, 1000)
    te_interp = np.linspace(10, 8000, 100000)
    cofne_spline_1 = interpolate.interp1d(te, cofne[:, 0])
    #cofne_spline_4 = interpolate.interp1d(te, cofne[:, 3])
    #cofne_spline_2 = interpolate.interp1d(te, cofne[:, 1])
    print("ne(ch1) = %.4f [x10^16 m^-3]" % (output1_mV*cofne_spline_1(Te1_eV)))
    #print("ne(ch4) = %.2f [x10^16 m^-3]" % (output4_mV*cofne_spline_4(Te4_eV)))
    #print("ne(ch2) = %.2f [x10^16 m^-3]" % (output2_mV*cofne_spline_2(Te2_eV)))

    #return output1_mV*cofne_spline_1[idx_1], output4_mV*cofne_spline_4[idx_4], output2_mV*cofne_spline_2[idx_2]

def getNearestValue(list, num):
    """
    概要: リストからある値に最も近い値を返却する関数
    @param list: データ配列
    @param num: 対象値
    @return 対象値に最も近い値
    """

    # リスト要素と対象値の差分を計算し最小値のインデックスを取得
    idx = np.abs(np.asarray(list) - num).argmin()
    #return list[idx]
    return idx


if __name__ == "__main__":
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['xtick.top'] = 'True'
    plt.rcParams['ytick.right'] = 'True'
    plt.rcParams['ytick.direction'] = 'in'
    #ytdb = YAGTS_DataBrowser(date=20180328, shotNo=15, shotSt=0)
    #ytdb.plot_shotlog(0, 120)
    #ytdb.open_with_pandas()
    #ytdb.show_graph()
    #make_shotlog(date=20180327, num_st=0, num_ed=120)
    integrate_SL(date=20180619, num_st=1, num_ed=4, isSerial=False)
    #subtract_straylight()
    #get_Te()
    #get_ne(40.3, 40.3, 40.3, 0.6546303, 0.6546303, 0.6546303)
    #get_ne(124.79, 40.3, 40.3, (0.06300-0.04959)*143.243, 0.6546303, 0.6546303)
    #get_ne(21.3, 40.3, 40.3, (0.03693-0.03673)*144.98, 0.6546303, 0.6546303)
