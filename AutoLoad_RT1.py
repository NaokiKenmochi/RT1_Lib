from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from RT1DataBrowser import DataBrowser
from pylab import *

import matplotlib.pyplot as plt
import read_wvf
import os
import time
import datetime


class ChangeHandler(FileSystemEventHandler, DataBrowser):
    def __init__(self, date, LOCALorPPL, isShotLog='False'):
        #super().__init__(date, shotNo, LOCALorPPL)
        self.date = date
        self.LOCALorPPL = LOCALorPPL
        self.isShotLog = isShotLog

    def on_created(self, event):
        filepath = event.src_path
        filename = os.path.basename(filepath)
        ctime_epoch = os.path.getctime(filepath)
        print('%sができました' % filename)
        if filename[-3:] == "wvf":
            shotNo=filename[:5].lstrip("0")
            if shotNo == "":
                shotNo = int(0)

            self.show_shotNo(time_epoch=ctime_epoch, shotNo=shotNo)
            time.sleep(15)
            db = DataBrowser(date=self.date, shotNo=int(shotNo), LOCALorPPL=self.LOCALorPPL, isShotLog=self.isShotLog)
            db.multiplot()
            stft =

    def on_modified(self, event):
        filepath = event.src_path
        filename = os.path.basename(filepath)
        print('%sを変更しました' % filename)

    def on_deleted(self, event):
        filepath = event.src_path
        filename = os.path.basename(filepath)
        print('%sを削除しました' % filename)

    def show_shotNo(self, time_epoch, shotNo):
        #plt.xkcd()
        plt.figure(figsize=(5,5))
        plt.clf()
        plt.tick_params(labelbottom="off", bottom="off")
        plt.tick_params(labelleft="off",left="off")
        box("off")
        plt.text(0.0, 1.0, "Last Shot Number:", fontsize=32, color='black')
        plt.text(0.0, 0.2, shotNo, fontsize=200, color='red', weight='bold')
        #plt.text(0, 0, time.localtime((time_epoch))[:6])
        plt.text(0, 0, "Time: " + time.ctime( time_epoch), color='blue', fontsize=16)
        plt.savefig("LastShotNumber.png")
        #plt.pause(0.1)


def main(date, isShotLog="False"):
    dm_ep01 = read_wvf.DataManager("exp_ep01", 0, date)
    print(dm_ep01.dir_path)
    target_dir = dm_ep01.dir_path
    dm_ep01._mount()
    while 1:
        event_handler = ChangeHandler(date=date, LOCALorPPL="PPL", isShotLog=isShotLog)
        observer = Observer()
        observer.schedule(event_handler, target_dir, recursive=True)
        observer.start()
        #shotNo, ctime_epoch = event_handler
        #event_handler.show_shotNo(date, event_handler,)
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()

if __name__ in '__main__':
    main(date="20180921", isShotLog="True")
#    for i in range(10):
#        show_shotNo("20180730", 1351670928.0, i)
#        time.sleep(0.5)
