from attrdict import AttrDict

class OnchipMem():
    def __init__(self):
        self.ip_act = AttrDict({'layer_idx': -1,'h': -1, 'end_h': -1, 'w': -1, 'end_w': -1, 'c': -1, 'end_c': -1})
        self.wgt = AttrDict({'layer_idx': -1,'h': -1, 'end_h': -1, 'w': -1, 'end_w': -1, 'c': -1, 'end_c': -1, 'f': -1, 'end_f': -1})

    def check_if_ip_act_exists(self,layer_idx, h,end_h,w,end_w,c,end_c):
        if self.ip_act.layer_idx == layer_idx and self.ip_act.h <= h and self.ip_act.end_h >= end_h \
                and self.ip_act.w <= w and self.ip_act.end_w >= end_w \
                and self.ip_act.c <= c and self.ip_act.end_c >= end_c:
            return True
        else:
            return False

    def check_if_wgt_exists(self, layer_idx, h,end_h,w,end_w,c,end_c, f, end_f):
        if self.wgt.layer_idx == layer_idx and self.wgt.h <= h and self.wgt.end_h >= end_h \
                and self.wgt.w <= w and self.wgt.end_w >= end_w \
                and self.wgt.c <= c and self.wgt.end_c >= end_c and self.wgt.f <= f \
                and self.wgt.end_f >= end_f:
            return True
        else:
            return False

    def clear(self):
        self.ip_act = AttrDict({'layer_idx':-1, 'h': -1, 'end_h': -1, 'w': -1, 'end_w': -1, 'c': -1, 'end_c': -1})
        self.wgt = AttrDict({'layer_idx':-1,'h': -1, 'end_h': -1, 'w': -1, 'end_w': -1, 'c': -1, 'end_c': -1, 'f': -1, 'end_f': -1})


    def insert_ip_act(self, layer_idx, h,end_h,w,end_w,c,end_c):
        self.ip_act = AttrDict({'layer_idx':layer_idx, 'h': h, 'end_h': end_h, 'w': w, 'end_w': end_w, 'c': c, 'end_c': end_c})

    def insert_wgt(self,layer_idx, h,end_h,w,end_w,c,end_c, f, end_f):
        self.wgt = AttrDict({'layer_idx':layer_idx,'h': h, 'end_h': end_h, 'w': w, 'end_w': end_w, 'c': c,
                                'end_c': end_c, 'f': f, 'end_f': end_f })
