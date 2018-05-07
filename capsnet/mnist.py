#!/usr/bin/env python
# -*- coding: utf-8 -*-

import struct, os, sys, random
import math
import time
import threading
from tkinter import *
from PIL import Image, ImageTk
import queue
from scrolledwindow import ScrolledWindow
import tensorflow as tf
import numpy as np
import capsnet
from tqdm import tqdm

gimgs = []

def train(msgqueue, num, batch_size = 8, 
             dataset_path = '../data/mnist', save_path = '../savedir'):
    train_sum_freq = 200
    val_sum_freq = 1000
    save_freq = 1000
    cwd, _ = os.path.split(sys.argv[0])
    data_path = os.path.join(cwd, dataset_path)
    save_path = os.path.join(cwd, save_path)
    trX, trY, max_num, valX, valY, num_val_batch = capsnet.load_mnist(True, data_path, batch_size)
    #Y = valY[:num_val_batch * batch_size].reshape((-1, 1))
    num_tr_batch = max_num
    num_val_batch = min(num, num_val_batch)
    capsNet = capsnet.CapsNet(is_training=True, batch_size = batch_size, data_path=data_path, kernel_size=3)
    print('Graph loaded')
    sv = tf.train.Supervisor(graph=capsNet.graph,
                                 logdir=save_path,
                                 save_model_secs=0)
    epoch = 1
    starttime = time.time()
    startstep = 0
    with sv.managed_session() as sess:
        for i in range(epoch):
            if sv.should_stop():
                break
            for step in tqdm(range(num_tr_batch), total=num_tr_batch, ncols=70, leave=False, unit='b'):
            #for step in range(num_tr_batch):
                global_step = i * num_tr_batch + step

                if global_step>0 and global_step % train_sum_freq == 0:
                    _, loss, train_acc = sess.run(
                        [capsNet.train_op, 
                         capsNet.total_loss, 
                         capsNet.accuracy])
                    assert not np.isnan(loss), 'Something wrong! loss is nan...'
                    #sv.summary_writer.add_summary(summary_str, global_step)

                    print('{}: loss = {:.3f}, accurate = {:.3f}, 速度={:.2f}n/s'.format(
                        global_step, loss, train_acc / batch_size, 
                        (global_step-startstep)/(time.time()-starttime)))
                    startstep = global_step
                    starttime = time.time()
                else:
                    sess.run(capsNet.train_op)
                if global_step<=0:
                    continue
                if val_sum_freq != 0 and (global_step) % val_sum_freq == 0:
                    val_acc = 0
                    for i in tqdm(range(num_val_batch), total=num_val_batch, ncols=70, leave=False, unit='b'):
                        start = i * batch_size
                        end = start + batch_size
                        decimgs, predicts, acc = sess.run(
                            [capsNet.decoded, capsNet.argmax_idx, capsNet.accuracy], 
                            {capsNet.X: valX[start:end], capsNet.labels: valY[start:end]})
                        val_acc += acc
                        imgs = (valX[start:end]*255)
                        imgs = imgs.reshape((-1, 28, 28)).astype(np.uint8)
                        decimgs = decimgs*255
                        decimgs = decimgs.reshape((-1, 28, 28)).astype(np.uint8)
                        msg=Messages(imgs, decimgs, predicts, valY[start:end], i, i<num_tr_batch-1)
                        msgqueue.put(msg)
                    val_acc = val_acc / (batch_size * num_val_batch)
                    print('validate step: {} accurate = {:.3f}'.format(global_step, val_acc))
                
                if (global_step) % save_freq == 0:
                    sv.saver.save(sess, save_path + '/model_epoch_%d_step_%d' % (i, global_step))

        global_step = sess.run(capsNet.global_step)
        sv.saver.save(sess, save_path + '/model_epoch_%d_step_%d' % (i, global_step))

def evaluate(msgqueue, num, batch_size = 32, 
             dataset_path = '../data/mnist', save_path = '../logdir'):
    cwd, _ = os.path.split(sys.argv[0])
    data_path = os.path.join(cwd, dataset_path)
    save_path = os.path.join(cwd, save_path)
    teX, teY, max_num = capsnet.load_mnist(False, data_path, batch_size)
    num_te_batch = min(max_num, num) 
    capsNet = capsnet.CapsNet(is_training=False, batch_size=batch_size, kernel_size=9)
    print('Graph loaded')
    with tf.Session(graph=capsNet.graph) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(save_path))
        print('Checkpoint restored')
        test_acc = 0
        begin = random.randint(0, max_num-num_te_batch)
        for i in tqdm(range(num_te_batch), total=num_te_batch, ncols=70, leave=False, unit='b'):
            start = (i+begin) * batch_size
            end = start + batch_size
            decimgs, predicts, acc = sess.run([capsNet.decoded, capsNet.argmax_idx, capsNet.accuracy], {capsNet.X: teX[start:end], capsNet.labels: teY[start:end]})
            test_acc += acc
            if i>0 and i%100 == 0:
                print('{}: accurate = {:.3f}%'.format(i, test_acc*100 / (batch_size * i)))
            imgs = (teX[start:end]*255)
            imgs = imgs.reshape((-1, 28, 28)).astype(np.uint8)
            decimgs = decimgs*255
            decimgs = decimgs.reshape((-1, 28, 28)).astype(np.uint8)
            msg=Messages(imgs, decimgs, predicts, teY[start:end], i, i<num_te_batch-1)
            msgqueue.put(msg)
        test_acc = test_acc*100 / (batch_size * num_te_batch)
        print('The average({} batchs) test accurate is: {:.3f}%'.format(num_te_batch, test_acc))

class myThread (threading.Thread):
    def __init__(self, name, train_num, test_num, is_training=False):
        threading.Thread.__init__(self)
        self.name = name
        self.train_num = train_num
        self.test_num = test_num
        self.running=False
        self.is_training=is_training

    def run(self):
        print ("开启线程： " + self.name)
        # 获取锁，用于线程同步
        #threadLock.acquire()
        self.running=True
        if self.is_training:
            train(msgqueue, self.test_num)
        else:
            evaluate(msgqueue, self.test_num)
        self.running=False
        print ("退出线程： " + self.name)

class Messages(object):
    def __init__(self, imgs, decodeimgs, predicts, labels, title, running=False):
        self.imgs=imgs
        self.decodeimgs=decodeimgs
        self.predicts=predicts
        self.labels=labels
        self.title=title
        self.running=running


class DrawFigure(threading.Thread):
    def __init__(self, parent, name, num, queue):
        threading.Thread.__init__(self)
        self.root=parent
        self.name = name
        self.queue=queue
        self.test_num=num
        self.running=True
        self.msgs=[]
        self.label = Label(self.root, text="空闲")
        self.label.grid(row=0, column=0, columnspan=4, sticky=EW, padx=10, pady=10)
        self.plot_frame = ScrolledWindow(self.root)
        self.plot_frame.grid(row=1, column=0, columnspan=4, sticky=NSEW, padx=10, pady=10)
        self.width = 900
        self.height = 720
        self.canvas = Canvas(self.plot_frame, bg='white')
        self.canvas.grid(row=0, column=0, sticky=W+E+N+S)

        self.btnPre = Button(self.root,text = '上一个',command = self.previous, state=DISABLED)
        self.btnPre.grid(row=2, column=0, sticky=W, padx=100, pady=20)
        self.btnTrain = Button(self.root,text = '训练',command = self.train)
        self.btnTrain.grid(row=2, column=1, padx=20, pady=20)
        self.btnTest = Button(self.root,text = '测试',command = self.test)
        self.btnTest.grid(row=2, column=2, padx=20, pady=20)
        self.btnNext = Button(self.root,text = '下一个',command = self.next, state=DISABLED)
        self.btnNext.grid(row=2, column=3, padx=100, pady=20)

        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

        self.current=-1
        self.scale = 5

    def quit(self):
        self.running = False
        sys.stdout = None
        sys.stderr = None
        self.root.quit()
        self.root.destroy()
        exit()

    def train(self):
        global gimgs
        gimgs = []
        self.label['text'] = '正在全力训练中……'
        self.btnTrain['state']=DISABLED
        train_num=600
        test_num=10
        thtrain = myThread('训练', train_num, test_num, is_training=True)
        thtrain.setDaemon(True)
        thtrain.start()

    def test(self):
        global gimgs
        gimgs = []
        self.label['text'] = '正在全力测试中……'
        self.btnTest['state']=DISABLED
        train_num=600
        test_num=10
        thtrain = myThread('计算', train_num, test_num, is_training=False)
        thtrain.setDaemon(True)
        thtrain.start()


    def previous(self):
        global gimgs
        oldindex = self.current
        if self.current>0:
            self.current -= 1
            self.show(self.current, oldindex)
            self.label['text']=gimgs[self.current].title
            if self.current<=0:
                self.btnPre['state']=DISABLED
            if self.current<len(gimgs)-1:
                self.btnNext['state']=NORMAL

    def next(self):
        global gimgs
        oldindex = self.current
        if self.current<len(gimgs)-1:
            self.current += 1
            self.show(self.current, oldindex)
            self.label['text']=gimgs[self.current].title
            if self.current>=len(gimgs)-1:
                self.btnNext['state']=DISABLED
            if self.current>0:
                self.btnPre['state']=NORMAL

    def run(self):
        print ("开启线程： " + self.name)
        # 获取锁，用于线程同步
        #threadLock.acquire()
        self.running=True
        while self.running:
            if msgqueue.qsize():
                msg=msgqueue.get(0)
                #self.draw()
                self.on_msg(msg)
            time.sleep(0.1)
        num_fig = len(gimgs)
        if num_fig>0:
            if self.current>0:
                self.btnPre['state']=NORMAL
            if self.current<num_fig-1:
                self.btnNext['state']=NORMAL
        print ("退出线程： " + self.name)
        
    def on_msg(self, msg):
        global gimgs
        if msg.title == 0:
            gimgs = []
        msgp=Messages(list(map(lambda x: arraytophoto(x, self.scale), msg.imgs)),
            list(map(lambda x: arraytophoto(x, self.scale), msg.decodeimgs)), 
            msg.predicts, msg.labels, '第{}批'.format(msg.title+1))
        gimgs.append(msgp)
        count=len(gimgs)
        self.show(count-1, self.current)
        self.current = count-1
        if not msg.running:
            self.btnPre['state']=NORMAL
            self.btnTest['state']=NORMAL

    def show(self, index, oldindex):
        global gimgs
        if oldindex==index:
            return
        self.canvas.delete(ALL)
        msg=gimgs[index]
        self.label['text'] = msg.title
        self.width = self.plot_frame.winfo_width()
        rowspace=50
        colspace=16
        colwidth=28*self.scale+colspace
        rowheight = 28*self.scale+rowspace
        pad=30
        startx=int(pad+28*self.scale/2.0)
        starty = 100
        cols=int((self.width-startx*2)/colwidth/2)*2
        pad=int((self.width-cols*colwidth)/2.0)
        texty = starty+int(28*self.scale/2)+20
        self.height= math.ceil(len(msg.imgs)*2/cols)*(rowheight)+starty+20
        self.canvas.config(width = self.width, height=self.height)

        for i,photo in enumerate(msg.imgs):
            self.canvas.create_image((colwidth*((i*2)%cols)+startx, rowheight*int(i*2/cols)+starty), image=photo) 
            self.canvas.create_image((colwidth*((i*2+1)%cols)+startx, rowheight*int((i*2+1)/cols)+starty), image=msg.decodeimgs[i]) 
            color='green'
            eq = '='
            if msg.predicts[i] != msg.labels[i]:
                eq = '!='
                color='red'
            self.canvas.create_text(colwidth*((i*2)%cols)+startx+colwidth//2, 
                                    rowheight*int(i*2/cols)+texty, 
                             text = '{}{}{}'.format(int(msg.labels[i]),eq, int(msg.predicts[i])), 
                             font = "serif 14 bold", fill=color)

def arraytophoto(a, scale=1):
    if isinstance(a, list):
        a = np.array(a)
    if a.ndim!=2:
        a = a.reshape((-1,28))
    img=Image.fromarray(a)
    x,y=img.size
    out = img.resize((x*scale, y*scale), Image.ANTIALIAS)
    return ImageTk.PhotoImage(out)


if __name__ == '__main__':
    msgqueue=queue.Queue()
    root=Tk()
    root.geometry('1080x820')
    root.resizable(width=True, height=True)
    root.title("胶囊网络手写输入测试")
    train_num=6800
    test_num=20
    thplot = DrawFigure(root, "显示", test_num, msgqueue)
    thplot.setDaemon(True)
    thplot.start()
    root.mainloop()


