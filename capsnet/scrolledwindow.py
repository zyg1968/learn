#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tkinter import *

class ScrolledWindow(Frame):
    """
    1. Master widget gets scrollbars and a canvas. Scrollbars are connected 
    to canvas scrollregion.

    2. self.scrollwindow is created and inserted into canvas

    Usage Guideline:
    Assign any widgets as children of <ScrolledWindow instance>.scrollwindow
    to get them inserted into canvas

    __init__(self, parent, canv_w = 400, canv_h = 400, *args, **kwargs)
    docstring:
    Parent = master of scrolled window
    canv_w - width of canvas
    canv_h - height of canvas

    """

    def __init__(self, master=None, **kw):
        self.frame = Frame(master, bg='green')
        self.vbar = Scrollbar(self.frame)
        self.canvas = Canvas(self.frame, bg='yellow', yscrollcommand=self.vbar.set)
        self.vbar.config(command=self.canvas.yview)
        self.canvas.pack(side=LEFT, fill=BOTH, expand=True)
        self.vbar.pack(side=RIGHT, fill=Y)

        #kw.update({'yscrollcommand': self.vbar.set})
        Frame.__init__(self, self.canvas, bg='blue', **kw)
        self.pack(side=TOP, fill=BOTH, expand=True)
        interior_id = self.canvas.create_window(0, 0, window=self,
                                           anchor=NW)
        def scrollall(event):
            self.canvas.configure(scrollregion=self.canvas.bbox('all'))
        #self.vbar['command'] = self.yview

        # Copy geometry methods of self.frame without overriding Text
        # methods -- hack!
        frame_meths = vars(Frame).keys()
        methods = vars(Pack).keys() | vars(Grid).keys() | vars(Place).keys()
        methods = methods.difference(frame_meths)

        for m in methods:
            if m[0] != '_' and m != 'config' and m != 'configure':
                setattr(self, m, getattr(self.frame, m))

        def _configure_interior(event):
            # update the scrollbars to match the size of the inner frame
            size = (self.winfo_reqwidth(), self.winfo_reqheight())
            #self.frame.update()
            self.canvas.config(scrollregion=self.canvas.bbox('all'))
            #if self.winfo_reqwidth() != self.canvas.winfo_width():, width = size[0], height=size[1]
                # update the canvas's width to fit the inner frame
            #    self.canvas.config(width=self.winfo_reqwidth())
        self.bind('<Configure>', _configure_interior)

        def _configure_canvas(event):
            if self.winfo_reqwidth() != self.canvas.winfo_width():
                # update the inner frame's width to fill the canvas
                self.canvas.itemconfigure(interior_id, width=self.canvas.winfo_width())
#                                          height = self.canvas.winfo_height())
        self.canvas.bind('<Configure>', _configure_canvas)


'''
    def __init__(self, parent, height = 400, width = 400, *args, **kwargs):
        """Parent = master of scrolled window
        canv_w - width of canvas
        canv_h - height of canvas

       """
        self.x=0
        self.y=0
        self.parent = Frame(master=parent, height=height, width=width, bg='yellow')
        # creating a canvas
        self.canv = Canvas(self.parent, height=height-21, width=width-21, bg='gray')
        # placing a canvas into frame

        super().__init__(self.canv, bg='red', *args, **kwargs)

        # creating a scrollbars
        self.xscrlbr = Scrollbar(self.parent, orient = 'horizontal')
        self.yscrlbr = Scrollbar(self.parent)
        # accociating scrollbar comands to canvas scroling
        self.xscrlbr.config(command = self.canv.xview)
        self.yscrlbr.config(command = self.canv.yview)

        # creating a frame to inserto to canvas
        self.interior_id = self.canv.create_window(0, 0, window = self, anchor = 'nw')

        self.canv.config(xscrollcommand = self.xscrlbr.set,
                         yscrollcommand = self.yscrlbr.set,
                         scrollregion = (0, 0, 100, 100))

        self.yscrlbr.lift(self)        
        self.xscrlbr.lift(self)
        self.update_layout()
        self.canv.bind('<Configure>', self.canv_configure)
        #self.bind('<Configure>', self.self_configure)
        self.parent.bind('<Enter>', self._bound_to_mousewheel)
        self.parent.bind('<Leave>', self._unbound_to_mousewheel)

        return

    def _bound_to_mousewheel(self, event):
        self.canv.bind_all("<MouseWheel>", self._on_mousewheel)   

    def _unbound_to_mousewheel(self, event):
        self.canv.unbind_all("<MouseWheel>") 

    def _on_mousewheel(self, event):
        self.canv.yview_scroll(int(-1*(event.delta/120)), "units")  

    def update_layout(self):
        self.update_idletasks()
        self.canv.configure(scrollregion=self.canv.bbox('all'))
        self.canv.yview('moveto','0.0')
        self.parent.size = self.grid_size()

    def self_configure(self, event):
        if (self.x!=event.x or self.y!=event.y):
            self.x,self.y=event.x,event.y
            return
        w,h = event.width, event.height
        reqw = self.winfo_reqwidth()
        reqh = self.winfo_reqheight()
        maxwidth = max(w, reqw)
        maxheight = max(h, reqh)
        xshow=haschild(self.parent, self.xscrlbr)
        yshow=haschild(self.parent, self.yscrlbr)
        changex=False
        changey=False
        if not xshow and maxwidth>self.parent.winfo_width():
            self.xscrlbr.grid(row=1, column=0, columnspan=2, sticky=EW)     
            changex=True
        elif xshow:
            self.xscrlbr.grid_forget()
            changex=True
        if not yshow and maxheight>self.parent.winfo_height():
            self.yscrlbr.grid(row=0, column=1, sticky=NS)         
            changey=True
        elif yshow:
            self.yscrlbr.grid_forget()
            changey=True
        if changey:
            self.canv['width']=self.parent.winfo_width()- (self.yscrlbr.winfo_width() if not yshow else 0)
        if changex:
            self.canv['height']=self.parent.winfo_height()-(self.xscrlbr.winfo_height() if not xshow else 0)

    def pack(self, cnf = {}, **kw):
        self.parent.pack_propagate(0)
        rt = self.parent.pack(cnf, **kw)
        self.xscrlbr.pack(side=BOTTOM, fill=X)         
        self.yscrlbr.pack(side=RIGHT, fill=Y)         
        self.canv.pack_propagate(1)
        self.canv.pack(side=LEFT, fill=BOTH, anchor=NW)
        self.pack_propagate(1)
        super().pack(fill=BOTH)
        return rt

    def grid(self, cnf = {}, **kw):
        self.parent.grid_propagate(0)
        rt = self.parent.grid(cnf, **kw)
        self.xscrlbr.grid(row=1, column=0, columnspan=2, sticky=EW)         
        self.yscrlbr.grid(row=0, column=1, sticky=NS)         
        self.canv.grid(row=0, column=0, sticky=NSEW)
        #self.parent.columnconfigure(0, weight=1)
        super().grid_columnconfigure(0, weight=1)
        super().grid_rowconfigure(0, weight=1)
        #self.parent.rowconfigure(0, weight=1)
        #super().grid_columnconfigure(0, weight=1)
        #super().grid(sticky=NSEW)
        return rt

def haschild(control, child):
    for item in control.grid_slaves():
        if item==child:
            return True
    return False
'''