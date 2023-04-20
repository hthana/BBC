
import tensorflow as tf

import sonnet as snt
import dsnt

import sonnet as snt

class DSNT(snt.AbstractModule):

    def __init__(self, name='DSNT', method='softmax'):

        super(DSNT, self).__init__(name=name)
        self._method = 'softmax'

    def _build(self, inputs):
        norm_heatmap, coords = dsnt.dsnt(inputs, self._method)
        return norm_heatmap, coords
    
