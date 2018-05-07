#!/usr/bin/env python3
#
#    This file is part of ZiGo.
#    Copyright (C) 2018 ZiGo
#
# -*- coding: utf-8 -*-

#from tkinter import *
import configparser as cp
import os, sys

running = True                  #控制循环用
is_training= False

epsilon = 1e-9
#margin loss 中调节上margin和下margind的权重
lambda_val = 0.5
#上margin与下margin的参数值
m_plus = 0.9
m_minus = 0.1

# 路由更新c_ij所经过的迭代次数
iter_routing = 3
# Tensorboard 保存位置
log_path = '../log'
# 数据集路径
data_path = '../data'
save_path = '../model'
name = 'mnist'

batch_size = 8
kernel_size = (9,9)
img_size = (20,6)
filters = (256,32)
vec_lens = (8,16)
classes = 10
test_num = 40
train_sum_freq = 200
val_sum_freq = 1000
save_freq = 1000

show_pic = True

def init_cfg(config_file='config.ini'):
    global epsilon, lambda_val, m_plus, m_minus, iter_routing, test_num,show_pic
    global log_path,data_path,save_path,name, train_sum_freq, val_sum_freq, save_freq
    global batch_size, kernel_size, img_size, filters, classes, vec_lens

    cf = cp.ConfigParser()
    cwd,_ = os.path.split(sys.argv[0])
    cf.read(os.path.join(cwd, config_file), encoding="gb2312")
    epsilon = cf.getfloat('NET', 'epsilon')
    lambda_val = cf.getfloat('NET', 'lambda_val')
    m_plus = cf.getfloat('NET', 'm_plus')
    m_minus = cf.getfloat('NET', 'm_minus')
    iter_routing = cf.getint('NET', 'iter_routing')
    data_path = cf.get('NET', 'data_path')
    save_path = cf.get('NET', 'save_path')
    classes = cf.getint('NET', 'classes')
    ks = cf.get('NET', 'kernel_size')
    if ks:
        kernel_size = tuple(eval(ks))
    im = cf.get('NET', 'img_size')
    if im:
        img_size = tuple(eval(im))
    fs = cf.get('NET', 'filters')
    if fs:
        filters = tuple(eval(fs))
    vl = cf.get('NET', 'vec_lens')
    if vl:
        vec_lens = tuple(eval(vl))
    batch_size = cf.getint('NET', 'batch_size')
    train_sum_freq = cf.getint('NET', 'train_sum_freq')
    val_sum_freq = cf.getint('NET', 'val_sum_freq')
    save_freq = cf.getint('NET', 'save_freq')
    log_path = cf.get('COMMON', 'log_path')
    name = cf.get('COMMON', 'name')
    test_num = cf.getint('COMMON', 'test_num')
    show_pic = cf.getboolean('COMMON', 'show_pic')

def is_validate(content):        #如果你不加上==""的话，你就会发现删不完。总会剩下一个数字
    if content.isdigit() or (content==""):
        return True
    else:
        return False

def isfloat(s):
    m = re.match(r'^[0-9]*\.*[0-9]*$', s)
    if m or (s==""):
        return True
    return False

'''
class ConfigWindow():
    def __init__(self, wdpi, hdpi):
        self.top = Toplevel()
        self.gui(wdpi, hdpi)
        self.top.mainloop()

    def gui(self, wdpi, hdpi):
        global mtcs_width, mtcs_depth, mtcs_time, vresign
        width = int(wdpi * 7)
        height = int(hdpi * 5)
        padx=int(wdpi*0.05)
        pady=int(hdpi*0.05)
        self.top.geometry('%dx%d+%d+%d' % (width, height, wdpi* 3, hdpi*2))
        self.top.resizable(width=True, height=True)
        self.top.title("设置")
        self.vwidth=StringVar()
        self.vdepth=StringVar()
        self.vtime=StringVar()
        self.vvresign=StringVar()
        validate_fun=self.top.register(is_validate)#需要将函数包装一下，必要的
        lbl1=Label(self.top, text='蒙特卡洛搜索宽度：')
        self.txtwidth=Entry(self.top, textvariable = self.vwidth, validate='key', validatecommand=(validate_fun,'%P'))
        lbl2=Label(self.top, text='蒙特卡洛搜索深度：')
        self.txtdepth=Entry(self.top, textvariable = self.vdepth, validate='key', validatecommand=(validate_fun,'%P'))
        lbl3=Label(self.top, text='蒙特卡洛搜索时间：')
        self.txttime=Entry(self.top, textvariable = self.vtime, validate='key', validatecommand=(validate_fun,'%P'))
        lbl4=Label(self.top, text='投降的胜率阈值：')
        self.txtvresign=Entry(self.top, textvariable = self.vvresign, validate='key', validatecommand=(isfloat,'%P'))
        btnOk=Button(self.top, text='确定', command=self.save)
        btnCancel=Button(self.top, text='取消', command=self.cancel)
        lbl1.grid(row = 0, column=0, padx=padx, pady=pady)
        self.txtwidth.grid(row = 0, column=1, padx=padx, pady=pady)
        lbl2.grid(row = 1, column=0, padx=padx, pady=pady)
        self.txtdepth.grid(row = 1, column=1, padx=padx, pady=pady)
        lbl3.grid(row = 2, column=0, padx=padx, pady=pady)
        self.txttime.grid(row = 2, column=1, padx=padx, pady=pady)
        lbl4.grid(row = 3, column=0, padx=padx, pady=pady)
        self.txtvresign.grid(row = 3, column=1, padx=padx, pady=pady)
        btnOk.grid(row = 4, column = 0, padx=padx*4, pady=pady)
        btnCancel.grid(row = 4, column = 1, padx=padx*4, pady=pady)
        self.top.columnconfigure(1, weight = 1)
        self.top.rowconfigure(3, weight = 1)
        self.vwidth.set(str(mtcs_width))
        self.vdepth.set(str(mtcs_depth))
        self.vtime.set(str(mtcs_time))
        self.vvresign.set(str(vresign))

    def save(self, config_file=None):
        global mtcs_width, mtcs_depth, mtcs_time, vresign, last_config
        if config_file:
            last_config = config_file
        else:
            config_file = last_config
        config_file = 'config/%s.ini' % (config_file)
        mtcs_width = int(self.vwidth.get())
        mtcs_depth = int(self.vdepth.get())
        mtcs_time = int(self.vtime.get())
        vresign = float(self.vvresign.get())
        cf = cp.ConfigParser()
        cf.read(config_file)
        cf.set('MTCS', 'width', str(mtcs_width))
        cf.set('MTCS', 'depth', str(mtcs_depth))
        cf.set('MTCS', 'time', str(mtcs_time))
        cf.set('PLAY', 'vresign', str(vresign))
        with open(config_file,"w") as f:
            cf.write(f)
        shutile.copyfile(config_file, 'config/config.ini')
        cf0=cp.ConfigParser()
        cf0.read('config/config.ini')
        cf0.set('PLAY', 'last_config', last_config)
        with open('config/config.ini',"w") as f:
            cf0.write(f)
        self.top.destroy()

    def cancel(self):
        self.top.destroy()
'''
