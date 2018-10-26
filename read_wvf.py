'''
WE7000が出すバイナリファイル(.wvf)をそのヘッダファイル(.hdr)を用いて
読み取るプログラム
.hdrファイルに関する詳細は最後に記述する
2016/08/29
by Noriki Takahashi

exp_ep01, exp_ep02を区別して読み込む
2017/6/30
by Naoki Kenmochi

Edit:
package化に向けて編集開始
それまでのものはread_wvf_old.pyとした
読み込みに大して時間がかからないので，全てをメモリに読み込んで呼び出す
という方針でいく
'''

import numpy as np
from io import StringIO
import struct
import os, os.path
import re
import subprocess
import platform

class DataManager:

#    if(platform.system() == 'Darwin'):
#        _base_dir = os.path.expanduser('~/mount_point/exp_ep01')
#    elif(platform.system() == 'Windows'):
#        _base_dir = os.path.expanduser('//EXP_EP01/d/WEDATA')  #for windows(in same Network)

    def __init__(self, exp_ep01or02, MPorSX, date):
        ##################################
        #              定数              #
        ##################################
        # ヘッダーファイルから情報を引き出してバイナリを読む
        # hfile = '00064.hdr' # hedder file
        # bfile = '00064.wvf' # binary file

        # 以下のものだけ取り出せれば十分(igorでは少なくともそうだった)
        # 他のWEの製品が生成するものに対応したければ，エンディアンなどを調べる必要もある．
        # VR -> VResolution: 縦軸(vertical axis)の1目盛り分
        # VO -> VOffset    : 縦軸(vertical axis)のオフセット分
        # ticks * VR + VO が計測値(電圧値)
        # HR -> HResolution: 横軸(horizontal axis)の1目盛り分
        # HO -> HOffset    : 横軸(horizontal axis)のオフセット分
        # ticks * HR + HO が計測値(時間)
        self.VR_ID = 5
        self.VO_ID = 6
        self.HR_ID = 14
        self.HO_ID = 15

        # BlockSizeがある行
        self.BS_ID = 4

        # headerの読み取り位置
        self.START = 11
        self.STEP  = 19

        self.date = date
        self.exp_ep01or02 = exp_ep01or02
        self.MPorSX = MPorSX
#        self._set_date()
        self._mount()

        if(self.exp_ep01or02 == 'exp_ep01'):
            # チャンネル数(固定)
            self.CH_NUM = 32
            # Groupの数
            self.GROUP_NUM = 8
            if(platform.system() == 'Darwin'):
                #self._base_dir = os.path.expanduser('/Volumes/WEDATA')
                self._base_dir = os.path.expanduser('~/mount_point/exp_ep01/WEDATA')
            elif(platform.system() == 'Windows'):
                self._base_dir = os.path.expanduser('//EXP_EP01/d/WEDATA')  #for windows(in same Network)

        elif(self.exp_ep01or02 == 'exp_ep02'):
            if(self.MPorSX=="MP"):
                # チャンネル数(固定)
                self.CH_NUM = 8
                # Groupの数
                self.GROUP_NUM = 2
            elif(self.MPorSX=="SX"):
                # チャンネル数(固定)
                self.CH_NUM = 4
                # Groupの数
                self.GROUP_NUM = 1
            if(platform.system() == 'Darwin'):
                #self._base_dir = os.path.expanduser('/Volumes/D/WEDATA')
                self._base_dir = os.path.expanduser('~/mount_point/exp_ep02/WEDATA')
            elif(platform.system() == 'Windows'):
                self._base_dir = os.path.expanduser('//Exp_ep02/D/WEDATA')  #for windows in same network

        self._set_dir()

    def fetch_raw_ch_data(self, shot_nums, ch_nums):
        data_list = []
        for shot_num in shot_nums:
            data_list.append(self.fetch_raw_data(shot_num)[ch_nums])
        time_data = self.fetch_time_data(shot_nums[0])
        return time_data, data_list

    def fetch_time_data(self, shot_num):
        return self.fetch_raw_data(shot_num)[0]

    def fetch_raw_data(self, shot_num):
        path = self._generate_path(shot_num)
        return self.read_wvf(path)

    def read_wvf_header(self, bfile, hfile):
        '''
        headerファイルとbinaryファイルの名前を入れると，
        data(Groupごとの配列になっている)が返ってくる。
        '''

        header_list = self._make_header_list(hfile)
        chs = np.array([i + 1 for i in range(self.CH_NUM)])
        gid, cid = self._ch2indexes(chs)

        vr, vo, hr, ho, bs = (np.zeros_like(gid) * 1.0 for _ in range(5))
        for i in range(self.CH_NUM):
                vr[i] = np.loadtxt(StringIO(header_list[gid[i]][self.VR_ID]), usecols=(cid[i],))
                vo[i] = np.loadtxt(StringIO(header_list[gid[i]][self.VO_ID]), usecols=(cid[i],))
                hr[i] = np.loadtxt(StringIO(header_list[gid[i]][self.HR_ID]), usecols=(cid[i],))
                ho[i] = np.loadtxt(StringIO(header_list[gid[i]][self.HO_ID]), usecols=(cid[i],))
                bs[i] = np.loadtxt(StringIO(header_list[gid[i]][self.BS_ID]), usecols=(cid[i],))

        # BlockSizeはすべてのデータで同じとする
        # BlockSizeはデータの大きさ
        bs0 = int(bs[0])

        data_raw = self._read_binary(bfile, bs0)

        data = np.zeros_like(data_raw)
        # timeは全て同じとする
        t = hr[0] * np.linspace(0, bs0 - 1, bs0) + ho[0]

        data = (vr * data_raw.T + vo).T

        # 元のigorの形式と対応がつくように時間を0に入れる
        return np.vstack((t, data))

    def read_wvf(self, bfile):
        return self.read_wvf_header(bfile, bfile.replace('.wvf', '.hdr'))

    def _set_date(self):
        date_regex = re.compile(r'[0-9]{8}')
        # dirname = os.path.basename(os.path.dirname(__file__))
        dirname = os.path.basename(os.getcwd())
        date = ''
        if date_regex.match(dirname):
            date = dirname
        else:
            while not date_regex.match(date):
                date = input('date(yyyymmdd or today): ')
                if date == 'today' or date == '':
                    import datetime
                    date = datetime.date.today().strftime("%Y%m%d")
        self.date = date

    def _mount(self):
        """
        Mac OSでexp_ep01, exp_ep02を自動でマウントする処理
        始めに，ロールのホームディレクトリに"~/mount_point/exp_ep01/WEDATA", "~/mount_point/exp_ep02/WEDATA"を作成しておく
        場合によっては，"mount_smbfs..."のPC名（exp_ep01, exp_ep02）をそれぞれのIPアドレスに変更する
        :return:
        """
        try:
            if(self.exp_ep01or02 == 'exp_ep01'):
                if(platform.system() == 'Darwin'):
                    cmd = 'mount_smbfs //rt-1:ringtrap@exp_ep01/WEDATA ~/mount_point/exp_ep01/WEDATA'
                    subprocess.check_call(cmd, shell=True)
                    #subprocess.check_call(cmd.split(" "), shell=True)

            elif(self.exp_ep01or02 == 'exp_ep02'):
                if(platform.system() == 'Darwin'):
                    cmd = 'mount_smbfs //rt-1:ringtrap@exp_ep02/D/WEDATA ~/mount_point/exp_ep02/WEDATA'
                    subprocess.check_call(cmd, shell=True)
        except Exception as e:
            if(e.args[0] == 64):
                print("!!!%s is already mounted !!!" % self.exp_ep01or02)
            elif(e.args[0] == 68):
                print("Error; mount_smbfs: server connection failed: No route to host")
            else:
                print("!!!!!!!!!!!!!!!!! ERROR !!!!!!!!!!!!!!!!!!!!")
                print(e.args)
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    def _set_dir(self):
        if(self.exp_ep01or02 == "exp_ep01"):
            we_dir = 'd7273'
            self.dir_path = os.path.join(self._base_dir, 'd' + self.date, we_dir)
        elif(self.exp_ep01or02 == "exp_ep02"):
            if(self.MPorSX=="MP"):
                we_dir = 'FC'       #MP
            elif(self.MPorSX=="SX"):
                we_dir = 'd7116'    #SX
            self.dir_path = os.path.join(self._base_dir, 'd' + self.date + '_2', we_dir)

    def _generate_path(self, shot_num):
        return os.path.join(self.dir_path, '{0:05d}.wvf'.format(shot_num))

    def _ch2indexes(self, ch):
        """
        input: ch
        output: index of group (header_list[*])
                index of usecols (np.loadtxt(-, usecols=(*,))
        """
        gid = (ch - 1)// 4
        cid = ch - gid * 4
        return gid, cid

    def _make_header_list(self, hfile):
        # グループごとのリストをつくる
        header_list = [[] for _ in range(self.GROUP_NUM)]

        # ヘッダーファイルを読みこむ
        with open(hfile, 'r') as f:
            header = f.readlines()

        for i in range(self.GROUP_NUM):
            header_list[i] = header[self.START + i * self.STEP: self.START + (i + 1) * self.STEP]
        return header_list

    def _read_binary(self, bfile, blocksize):
        bf = open(bfile, 'br')
        dt = np.dtype('<h')
        data_raw = np.zeros((self.CH_NUM, blocksize))
        for i in range(self.CH_NUM):
                # 2バイトずつ，符号なし整数で読んでいく
                data_raw[i] = np.frombuffer(bf.read(blocksize * 2), dt)
        bf.close()
        return data_raw

    def _read_binary_chs(self, bfile, blocksize, chs):
        bf = open(bfile, 'br')
        dt = np.dtype('<h')
        ch_num = len(chs)
        data_raw = {ch: np.zeros((blocksize,)) for ch in chs}
        for i in range(self.CH_NUM):
            # 2バイトずつ，符号なし整数で読んでいく
            if (i + 1) in chs.keys():
                data_raw[i + 1] = np.frombuffer(bf.read(blocksize * 2), dt)
            else:
                bf.read(blocksize * 2)
        bf.close()
        return data_raw
'''
.hdrは以下のような構造になっている．
プログラムでVR_IDなどとなっているのは，各グループで(VResolutionが)何行目に来ているか
を示している．


//YOKOGAWA ASCII FILE FORMAT

$PublicInfo
FormatVersion     1.01
Model             WE7273
Endian            Ltl
DataFormat        Block
GroupNumber       8
TraceTotalNumber  32
DataOffset        0

$Group1
TraceNumber       4
BlockNumber       1
TraceName         CH1            CH2            CH3            CH4
BlockSize         40000          40000          40000          40000
VResolution       6.241028522e-005  6.249511757e-005  3.123682197e-004  3.122219273e-004
VOffset           -1.560257130e-003 -8.749316460e-004 3.123682197e-004  -2.497775419e-003
VDataType         IS2            IS2            IS2            IS2
VUnit             V              V              V              V
VPlusOverData     32072          32017          32015          32036
VMinusOverData    -32022         -31990         -32014         -32023
VIllegalData      NAN            NAN            NAN            NAN
VMaxData          3.207100e+004  3.201600e+004  3.201400e+004  3.203500e+004
VMinData          -3.202100e+004 -3.198900e+004 -3.201300e+004 -3.202200e+004
HResolution       1.000000e-004  1.000000e-004  1.000000e-004  1.000000e-004
HOffset           -0.000000e+000 -0.000000e+000 -0.000000e+000 -0.000000e+000
HUnit             s              s              s              s
Date              2016/01/22     2016/01/22     2016/01/22     2016/01/22
Time               16:27:26       16:27:26       16:27:26       16:27:26
$Group2
...
(これがGroup8まで続く)

$PrivateInfo
RefFileNumber     0

'''

if __name__ == "__main__":
    dm = DataManager(exp_ep01or02="exp_ep01", MPorSX="MP", date="20180223")
    dm._mount()
