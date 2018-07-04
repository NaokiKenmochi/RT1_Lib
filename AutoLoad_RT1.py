from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from RT1DataBrowser import DataBrowser

import read_wvf
import os
import time


class ChangeHandler(FileSystemEventHandler, DataBrowser):
    def __init__(self, date, LOCALorPPL):
        #super().__init__(date, shotNo, LOCALorPPL)
        self.date = date
        self.LOCALorPPL = LOCALorPPL

    def on_created(self, event):
        filepath = event.src_path
        filename = os.path.basename(filepath)
        print('%sができました' % filename)
        if filename[-3:] == "wvf":
            shotNo=filename[:5].lstrip("0")
            if shotNo == "":
                shotNo = int(0)
            db = DataBrowser(date=self.date, shotNo=int(shotNo), LOCALorPPL=self.LOCALorPPL)
            db.multiplot()


    def on_modified(self, event):
        filepath = event.src_path
        filename = os.path.basename(filepath)
        print('%sを変更しました' % filename)

    def on_deleted(self, event):
        filepath = event.src_path
        filename = os.path.basename(filepath)
        print('%sを削除しました' % filename)

def main(date):
    dm_ep01 = read_wvf.DataManager("exp_ep01", 0, date)
    print(dm_ep01.dir_path)
    target_dir = dm_ep01.dir_path
    dm_ep01._mount()
    while 1:
        event_handler = ChangeHandler(date=date, LOCALorPPL="PPL")
        observer = Observer()
        observer.schedule(event_handler, target_dir, recursive=True)
        observer.start()
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()

if __name__ in '__main__':
    main(date="20180625")
