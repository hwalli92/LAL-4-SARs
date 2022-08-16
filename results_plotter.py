import matplotlib
matplotlib.use("WXAgg")
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from matplotlib.figure import Figure

import wx
import os
import yaml
import numpy as np

class MyFrame(wx.Frame):    
    def __init__(self):
        super().__init__(parent=None, title='Lifelong Action Learning Results Plotter', size=(1250,700))

        self.split_win = wx.SplitterWindow(self)
        self.left_split = wx.Panel(self.split_win, style=wx.SUNKEN_BORDER)
        self.plt_win = PlotPanel(self.split_win)
        self.split_win.SplitVertically(self.left_split, self.plt_win, 425)

        self.sel_metric = wx.Button(self.left_split, label='Select Metric', pos=(5, 5))
        self.sel_metric.Bind(wx.EVT_BUTTON, self.select_metric)

        self.dir = "results"
        self.data = self.get_available_data()
        self.data_list = list(self.data.keys())
        self.dl = wx.ListBox(self.left_split, name="Available Data", size=(400, 200), pos=(5, 50), style=wx.LB_MULTIPLE, choices=self.data_list)

        self.plt_btn = wx.Button(self.left_split, label='Plot', pos=(5, 300))
        self.plt_btn.Bind(wx.EVT_BUTTON, self.plot)

        self.clr_btn = wx.Button(self.left_split, label='Clear', pos=(100, 300))
        self.clr_btn.Bind(wx.EVT_BUTTON, self.clear_plot)

        self.save_btn = wx.Button(self.left_split, label='Save', pos=(200, 300))
        self.save_btn.Bind(wx.EVT_BUTTON, self.save_plot)

    def select_metric(self, event):
        dlg = wx.SingleChoiceDialog(None, "Pick a Metric", "Metrics", ["TAw Acc", "TAg Acc", "TAw Forg", "TAg Forg"], wx.CHOICEDLG_STYLE)
        if dlg.ShowModal() == wx.ID_OK:
            metric = dlg.GetStringSelection()
            self.sel_metric.SetLabel(metric)
        dlg.Destroy()

    def get_available_data(self):
        #folder_list = sorted(sorted(os.listdir(self.dir)))
        with open(self.dir + "/results_config.yaml", 'r') as f:
            folder_list = yaml.load(f)
        return folder_list

    def update_classes(self, event):
        new_classes = self.classes_input.GetValue()
        print(new_classes.split())
        self.classes = [int(x) for x in new_classes.split()]
        print(self.classes)

    def load_data(self, datas, metric):
        plot_data = []

        for data in datas:
            res_dir = os.path.join(self.dir, self.data_list[data] + "/results/")
            for file in os.listdir(res_dir):
                if file.startswith("avg_accs_taw") and metric=="TAw Acc":
                    f = os.path.join(res_dir, file)
                    d = np.loadtxt(f)
                    plot_data.append(d)
                elif file.startswith("avg_accs_tag") and metric=="TAg Acc":
                    f = os.path.join(res_dir, file)
                    d = np.loadtxt(f)
                    plot_data.append(d)
                elif file.startswith("forg_taw") and metric=="TAw Forg":
                    f = os.path.join(res_dir, file)
                    d = np.mean(np.loadtxt(f), axis=1)
                    plot_data.append(d)
                elif file.startswith("forg_tag") and metric=="TAg Forg":
                    f = os.path.join(res_dir, file)
                    d = np.mean(np.loadtxt(f), axis=1)
                    plot_data.append(d)

        return plot_data

    def plot(self, event):
        s = self.dl.GetSelections()
        m = self.sel_metric.GetLabel()

        if m in ["TAw Acc", "TAg Acc", "TAw Forg", "TAg Forg"]:
            plot_data = self.load_data(s, m)

            for idx, d in enumerate(plot_data):
               self.plt_win.axes.plot(self.data[self.data_list[s[idx]]]["classes"], d*100, label=self.data[self.data_list[s[idx]]]["label"])

            self.plt_win.axes.set_xticks(np.arange(2, 21, 2))
            if m in ["TAw Acc", "TAg Acc"]:
                self.plt_win.axes.set_yticks(np.arange(45, 101, 5))
                self.plt_win.axes.set_ylabel("Accuracy %")
            else:
                self.plt_win.axes.set_yticks(np.arange(0, 50, 5))
                self.plt_win.axes.set_ylabel("Forgetness %")

            self.plt_win.axes.set_xlabel("# of classes")
            self.plt_win.axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15))
            self.plt_win.axes.grid()
            self.plt_win.canvas.draw()

    def clear_plot(self, event):
        self.plt_win.axes.clear()

    def save_plot(self, event):
        self.plt_win.figure.savefig("../{}.png".format(self.sel_metric.GetLabel())) 

class PlotPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent,-1,size=(50,50))

        self.figure = Figure(figsize=(8,6))
        self.axes = self.figure.add_subplot(111)

        self.canvas = FigureCanvas(self, -1, self.figure)



if __name__ == '__main__':
    app = wx.App()
    frame = MyFrame()
    frame.Show()
    app.MainLoop()
