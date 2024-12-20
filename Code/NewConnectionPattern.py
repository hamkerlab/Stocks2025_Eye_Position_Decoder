import math
import time

from ANNarchy import CSR
from ANNarchy import *  #mach das niemals ordentlich
import Modelparams
import numpy as np
global loadedParam
loadedParam = Modelparams.defParams

# width = geometry[0], height = geometry[1]

saveConnections = False
if saveConnections:
    # get directory for saving
    import globalParams
    import os
    saveDir = globalParams.saveDir + "connections/"
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

MIN_CONNECTION_VALUE = 0.001

# connect two maps with a gaussian receptive field 1d to 2d
# 1d is supposed to be vertical
def connect_gaussian1dTo2d_v(pre, post, mv, radius, mgd):

    time0 = time.time()

    prW = pre.width
    poW = post.width
    poH = post.height

    #print("gaussian1dTo2d_v:", mv, radius, mgd, "(connecting", pre.name, "to", post.name, ")")

    # Normalization along width of sigma values on afferent map
    if prW == 21 or prW == 41:
        sigma = radius * (prW-1)
    else:
        sigma = radius * prW

    synapse = CSR()

    # for speedup
    m_exp = math.exp

    if saveConnections:
        fn = pre.name + "_" + post.name + "_" + "gaussian1dTo2d_v"
        strToWrite = "connect " + pre.name + " to " + post.name + " with gaussian1dTo2d_v" + "\n\n"

    for w_post in range(poW):
        for h_post in range(poH):

            post_rank = post.rank_from_coordinates((w_post, h_post))

            values = []
            pre_ranks = []

            for h_pre in range(prW):

                saveVal = 0

                if pre != post or h_post != h_pre:
                    dist = (h_post-h_pre)**2
                    if ((mgd == 0) or (dist < mgd*mgd)):
                        val = mv * m_exp(-dist/sigma/sigma)
                        if val > MIN_CONNECTION_VALUE:
                            # connect
                            values.append(val)
                            pre_ranks.append(h_pre)

                            saveVal = val

                if saveConnections:
                    strToWrite += "(" + str(h_pre) + ") -> (" + str(w_post) + "," + str(h_post) + ") with " + str(saveVal) + "\n"

            synapse.add(post_rank, pre_ranks, values, [0])

    if saveConnections:
        f = open(saveDir+fn+'.txt', 'w')
        f.write(strToWrite)
        f.close()

    time1 = time.time()
    #print('created', synapse.nb_synapses, 'synapses out of ', poW*prW*poH, ' in', time1-time0, 's')
    return synapse

def connect_gaussian1dTo2d_v_variable_delays(pre, post, mv, radius, mgd):

    delay_dist = Normal(loadedParam["delay_mu"],loadedParam['delay_sigma'],loadedParam['delay_min'],loadedParam['delay_max'])   # ERROR: wir glauben das die Delays nicht mit dem anlegen des Delay-F Vektors aus einer Verteilung klarkommen
    time0 = time.time()

    prW = pre.width
    poW = post.width
    poH = post.height

    #print("gaussian1dTo2d_v:", mv, radius, mgd, "(connecting", pre.name, "to", post.name, ")")

    # Normalization along width of sigma values on afferent map
    if prW == 21 or prW == 41:
        sigma = radius * (prW-1)
    else:
        sigma = radius * prW

    synapse = CSR()

    # for speedup
    m_exp = math.exp

    if saveConnections:
        fn = pre.name + "_" + post.name + "_" + "gaussian1dTo2d_v"
        strToWrite = "connect " + pre.name + " to " + post.name + " with gaussian1dTo2d_v" + "\n\n"

    for w_post in range(poW):
        for h_post in range(poH):

            post_rank = post.rank_from_coordinates((w_post, h_post))

            values = []
            pre_ranks = []
            

            for h_pre in range(prW):

                saveVal = 0

                if pre != post or h_post != h_pre:
                    dist = (h_post-h_pre)**2
                    if ((mgd == 0) or (dist < mgd*mgd)):
                        val = mv * m_exp(-dist/sigma/sigma)
                        if val > MIN_CONNECTION_VALUE:
                            # connect
                            values.append(val)
                            pre_ranks.append(h_pre)

                            saveVal = val

                if saveConnections:
                    strToWrite += "(" + str(h_pre) + ") -> (" + str(w_post) + "," + str(h_post) + ") with " + str(saveVal) + "\n"
            
            # one delay for all in this dendrite
            my_delays = [delay_dist.get_values((1,))]*len(pre_ranks)
            #my_delays = [150.0]*len(pre_ranks)

            synapse.add(post_rank, pre_ranks, values, my_delays)

    if saveConnections:
        f = open(saveDir+fn+'.txt', 'w')
        f.write(strToWrite)
        f.close()

    time1 = time.time()
    #print('created', synapse.nb_synapses, 'synapses out of ', poW*prW*poH, ' in', time1-time0, 's')
    return synapse


# connect two maps with a gaussian receptive field 1d to 2d
# 1d is supposed to be horizontal
def connect_gaussian1dTo2d_h(pre, post, mv, radius, mgd):

    time0 = time.time()

    prW = pre.width
    poW = post.width
    poH = post.height

    #print("gaussian1dTo2d_h:", mv, radius, mgd, "(connecting", pre.name, "to", post.name, ")")

    # Normalization along width of sigma values on afferent map
    if prW == 21 or prW == 41:
        sigma = radius * (prW-1)
    else:
        sigma = radius * prW

    synapse = CSR()

    # for speedup
    m_exp = math.exp

    if saveConnections:
        fn = pre.name + "_" + post.name + "_" + "gaussian1dTo2d_h"
        strToWrite = "connect " + pre.name + " to " + post.name + " with gaussian1dTo2d_h" + "\n\n"

    for w_post in range(poW):
        for h_post in range(poH):

            post_rank = post.rank_from_coordinates((w_post, h_post))

            values = []
            pre_ranks = []

            for w_pre in range(prW):

                saveVal = 0

                if pre != post or w_post != w_pre:
                    dist = (w_post - w_pre)**2
                    if ((mgd == 0) or (dist < mgd*mgd)):
                        val = mv * m_exp(-dist/sigma/sigma)
                        if val > MIN_CONNECTION_VALUE:
                            # connect
                            values.append(val)
                            pre_ranks.append(w_pre)

                            saveVal = val

                if saveConnections:
                    strToWrite += "(" + str(w_pre) + ") -> (" + str(w_post) + "," + str(h_post) + ") with " + str(saveVal) + "\n"

            synapse.add(post_rank, pre_ranks, values, [0])

    if saveConnections:
        f = open(saveDir+fn+'.txt', 'w')
        f.write(strToWrite)
        f.close()

    time1 = time.time()
    #print('created', synapse.nb_synapses, 'synapses out of ', poW*prW*poH, ' in', time1-time0, 's')
    return synapse

def connect_gaussian1dTo2d_h_noisy_weights(pre, post, mv, radius, mgd):

    time0 = time.time()

    prW = pre.width
    poW = post.width
    poH = post.height

    #print("gaussian1dTo2d_h:", mv, radius, mgd, "(connecting", pre.name, "to", post.name, ")")
	
    # Normalization along width of sigma values on afferent map
    if prW == 21 or prW == 41:
        sigma = radius * (prW-1)
    else:
        sigma = radius * prW

    synapse = CSR()

    # for speedup
    m_exp = math.exp

    if saveConnections:
        fn = pre.name + "_" + post.name + "_" + "gaussian1dTo2d_h"
        strToWrite = "connect " + pre.name + " to " + post.name + " with gaussian1dTo2d_h" + "\n\n"

    for w_post in range(poW):
        for h_post in range(poH):

            post_rank = post.rank_from_coordinates((w_post, h_post))

            values = []
            pre_ranks = []

            for w_pre in range(prW):

                saveVal = 0
                temp_mv = mv * np.random.normal(1,0.3,1)
                if pre != post or w_post != w_pre:
                    dist = (w_post - w_pre)**2
                    if ((mgd == 0) or (dist < mgd*mgd)):
                        val = temp_mv * m_exp(-dist/sigma/sigma)
                        if val > MIN_CONNECTION_VALUE:
                            # connect
                            values.append(val)
                            pre_ranks.append(w_pre)

                            saveVal = val

                if saveConnections:
                    strToWrite += "(" + str(w_pre) + ") -> (" + str(w_post) + "," + str(h_post) + ") with " + str(saveVal) + "\n"

            synapse.add(post_rank, pre_ranks, values, [0])

    if saveConnections:
        f = open(saveDir+fn+'.txt', 'w')
        f.write(strToWrite)
        f.close()

    time1 = time.time()
    #print('created', synapse.nb_synapses, 'synapses out of ', poW*prW*poH, ' in', time1-time0, 's')
    return synapse





# connect two maps with a gaussian receptive field 2d to 1d
# 1d is supposed to be horizontal
def connect_gaussian2dTo1d_h(pre, post, mv, radius, mgd):

    time0 = time.time()

    prW = pre.width
    prH = pre.height
    poW = post.width

    #print("gaussian2dTo1d_h:", mv, radius, mgd, "(connecting", pre.name, "to", post.name, ")")

    # Normalization along width of sigma values on afferent map
    if prW == 21 or prW == 41:
        sigma = radius * (prW-1)
    else:
        sigma = radius * prW

    synapse = CSR()

    # for speedup
    m_exp = math.exp

    if saveConnections:
        fn = pre.name + "_" + post.name + "_" + "gaussian2dTo1d_h"
        strToWrite = "connect " + pre.name + " to " + post.name + " with gaussian2dTo1d_h" + "\n\n"

    for w_post in range(poW):

        values = []
        pre_ranks = []

        for w_pre in range(prW):

            dist = (w_post - w_pre)**2
            if ((mgd == 0) or (dist < mgd*mgd)):
                val = mv * m_exp(-dist/sigma/sigma)
                if val > MIN_CONNECTION_VALUE:

                    for h_pre in range(prH):

                        # connect
                        pre_rank = pre.rank_from_coordinates((w_pre, h_pre))
                        values.append(val)
                        pre_ranks.append(pre_rank)

                        if saveConnections:
                            strToWrite += "(" + str(w_pre) + "," + str(h_pre) + ") -> (" + str(w_post) + ") with " + str(val) + "\n"

        synapse.add(w_post, pre_ranks, values, [0])

    if saveConnections:
        f = open(saveDir+fn+'.txt', 'w')
        f.write(strToWrite)
        f.close()

    time1 = time.time()
    #print('created', synapse.nb_synapses, 'synapses out of ', poW*prW*prH, ' in', time1-time0, 's')

    return synapse

# connect two maps with a gaussian receptive field 2d to 1d diagonally
# 1d is supposed to be just width, 2d is supposed to be a square
# intended to connect the central RBF maps to Xh
def connect_gaussian2dTo1d_diag(pre, post, mv, radius, mgd):

    time0 = time.time()

    prW = pre.width
    prH = pre.height
    poW = post.width

    #print("gaussian2dTo1d_diag:", mv, radius, mgd, "(connecting", pre.name, "to", post.name, ")")

    # Normalization along width of sigma values on afferent map
    if prW == 21 or prW == 41:
        sigma = radius * (prW-1) #<-- not very consistent (properly better to normalize along diagonal)
        offset = (prW-1)/2.0
    else:
        sigma = radius * prW
        offset = prW/2.0

    synapse = CSR()

    # for speedup
    m_exp = math.exp

    if saveConnections:
        fn = pre.name + "_" + post.name + "_" + "gaussian2dTo1d_diag"
        strToWrite = "connect " + pre.name + " to " + post.name + " with gaussian2dTo1d_diag" + "\n\n"
    
    for w_post in range(poW):

        values = []
        pre_ranks = []

        for w_pre in range(prW):
            for h_pre in range(prH):

                saveVal = 0

                if pre != post or w_post != w_pre:
                    dist = (w_post - (w_pre+h_pre) + offset)**2
                    if ((mgd == 0) or (dist < mgd*mgd)):
                        val = mv * m_exp(-dist/sigma/sigma)
                        if val > MIN_CONNECTION_VALUE:
                            # connect
                            pre_rank = pre.rank_from_coordinates((w_pre, h_pre))
                            values.append(val)
                            pre_ranks.append(pre_rank)

                            saveVal = val

                if saveConnections:
                    strToWrite += "(" + str(w_pre) + "," + str(h_pre) + ") -> (" + str(w_post) + ") with " + str(saveVal) + "\n"

        synapse.add(w_post, pre_ranks, values, [0])

    if saveConnections:
        f = open(saveDir+fn+'.txt', 'w')
        f.write(strToWrite)
        f.close()

    time1 = time.time()
    #print('created', synapse.nb_synapses, 'synapses out of ', poW*prW*prH, ' in', time1-time0, 's')
    return synapse

# connect two maps with a gaussian receptive field 1d to 2d diagonally
# 1d is supposed to be just width, 2d is supposed to be a square
def connect_gaussian1dTo2d_diag(pre, post, mv, radius, mgd):

    time0 = time.time()

    prW = pre.width
    poW = post.width
    poH = post.height

    #print("gaussian1dTo2d_diag:", mv, radius, mgd, "(connecting", pre.name, "to", post.name, ")")

    synapse = CSR()

    # Normalization along width of sigma values on afferent map
    if prW == 21 or prW == 41:
        sigma = radius * (prW-1)
        offset = (prW-1)/2.0
    else:
        sigma = radius * prW
        offset = prW/2.0

    # for speedup
    m_exp = math.exp

    if saveConnections:
        fn = pre.name + "_" + post.name + "_" + "gaussian1dTo2d_diag"
        strToWrite = "connect " + pre.name + " to " + post.name + " with gaussian1dTo2d_diag" + "\n\n"

    for w_post in range(poW):
        for h_post in range(poH):

            post_rank = post.rank_from_coordinates((w_post, h_post))

            values = []
            pre_ranks = []

            for w_pre in range(prW):

                saveVal = 0

                if pre != post or w_post != w_pre:
                    dist = (w_post + h_post - w_pre - offset)**2
                    if ((mgd == 0) or (dist < mgd*mgd)):
                        val = mv * m_exp(-dist/sigma/sigma)
                        if val > MIN_CONNECTION_VALUE:
                            # connect
                            values.append(val)
                            pre_ranks.append(w_pre)

                            saveVal = val

                if saveConnections:
                    strToWrite += "(" + str(w_pre) + ") -> (" + str(w_post) + "," + str(h_post) + ") with " + str(saveVal) + "\n"

            synapse.add(post_rank, pre_ranks, values, [0])

    if saveConnections:
        f = open(saveDir+fn+'.txt', 'w')
        f.write(strToWrite)
        f.close()

    time1 = time.time()
    #print('created', synapse.nb_synapses, 'synapses out of ', poW*prW*poH, ' in', time1-time0, 's')
    return synapse

# connect two maps with a gaussian receptive field 2d to 2d
# it is intended to connect a 2d Xe_CD (that is retinotopic CD with eye position gain field, Xe2r) read out diagonally to the CD input side of Xb_CD
def connect_gaussian2d_diagTo2d_v(pre, post, mv, radius, mgd):

    time0 = time.time()

    prW = pre.width
    prH = pre.height
    poW = post.width
    poH = post.height

    #print("gaussian2d_diagTo2d_v:", mv, radius, mgd, "(connecting", pre.name, "to", post.name, ")")

    # Normalization along width of sigma values on afferent map
    if prW == 21 or prW == 41:
        sigma = radius * (prW-1) #<-- not very consistent (properly better to normalize along diagonal)
        offset = (prW-1)/2.0
    else:
        sigma = radius * prW
        offset = prW/2.0

    synapse = CSR()

    # for speedup
    m_exp = math.exp

    if saveConnections:
        fn = pre.name + "_" + post.name + "_" + "gaussian2d_diagTo2d_v"
        strToWrite = "connect " + pre.name + " to " + post.name + " with gaussian2d_diagTo2d_v" + "\n\n"

    for w_post in range(poW):
        for h_post in range(poH):

            post_rank = post.rank_from_coordinates((w_post, h_post))

            values = []
            pre_ranks = []

            for w_pre in range(prW):
                for h_pre in range(prH):

                    saveVal = 0

                    if pre != post or h_post != h_pre:
                        dist = (h_post - (w_pre+h_pre) + offset)**2
                        if ((mgd == 0) or (dist < mgd*mgd)):
                            val = mv * m_exp(-dist/sigma/sigma)
                            if val > MIN_CONNECTION_VALUE:
                                # connect
                                pre_rank = pre.rank_from_coordinates((w_pre, h_pre))

                                values.append(val)
                                pre_ranks.append(pre_rank)

                                saveVal = val


                    if saveConnections:
                        strToWrite += "(" + str(w_pre) + "," + str(h_pre) + ") -> (" + str(w_post) + "," + str(h_post) + ") with " + str(saveVal) + "\n"

            synapse.add(post_rank, pre_ranks, values, [0])

    if saveConnections:
        f = open(saveDir+fn+'.txt', 'w')
        f.write(strToWrite)
        f.close()

    time1 = time.time()
   # print('created', synapse.nb_synapses, 'synapses out of ', poW*prW*poH*prH, ' in', time1-time0, 's')
    return synapse

# connecting two maps (normally these maps are equal) with gaussian field depending on distance
# maps are 1d
def connect_all2all_exp1d(pre, post, factor, radius, mgd):

    time0 = time.time()

    prW = pre.width
    poW = post.width

    #print("all2all_exp1d:", factor, radius, mgd, "(connecting", pre.name, "to", post.name, ")")

    # Normalization along width of sigma values on afferent map
    if prW == 21 or prW == 41:
        sigma = radius * (prW-1)
        mgd = mgd * (prW-1)
    else:
        sigma = radius * prW
        mgd = mgd * prW

    synapse = CSR()

    # for speedup
    m_exp = math.exp

    if saveConnections:
        fn = pre.name + "_" + post.name + "_" + "all2all_exp1d"
        strToWrite = "connect " + pre.name + " to " + post.name + " with all2all_exp1d" + "\n\n"

    for w_post in range(poW):

        values = []
        pre_ranks = []

        for w_pre in range(prW):

            saveVal = 0

            # distance between 2 neurons
            dist = (w_post-w_pre)**2

            if ((mgd == 0) or (dist < mgd*mgd)):
                val = factor * m_exp(-dist/sigma/sigma)
                # connect
                values.append(val)
                pre_ranks.append(w_pre)

                saveVal = val

            if saveConnections:
                strToWrite += "(" + str(w_pre) + ") -> (" + str(w_post) + ") with " + str(saveVal) + "\n"

        synapse.add(w_post, pre_ranks, values, [0])

    if saveConnections:
        f = open(saveDir+fn+'.txt', 'w')
        f.write(strToWrite)
        f.close()

    time1 = time.time()
    #print('created', synapse.nb_synapses, 'synapses out of ', poW*prW, ' in', time1-time0, 's')
    return synapse


# connecting two maps (normally these maps are equal) with gaussian field depending on distance
# maps are 2d
def connect_all2all_exp2d(pre, post, factor, radius, mgd):

    time0 = time.time()

    prW = pre.width
    prH = pre.height
    poW = post.width
    poH = post.height

    #print("all2all_exp2d:", factor, radius, mgd, "(connecting", pre.name, "to", post.name, ")")

    # Normalization along width of sigma values on afferent map
    if prW == 21 or prW == 41:
        sigma = radius * (prW-1)
        mgd = mgd * (prW-1)
    else:
        sigma = radius * prW
        mgd = mgd * prW

    synapse = CSR()

    # for speedup
    m_exp = math.exp

    if saveConnections:
        fn = pre.name + "_" + post.name + "_" + "all2all_exp2d"
        strToWrite = "connect " + pre.name + " to " + post.name + " with all2all_exp2d" + "\n\n"

    for w_post in range(poW):
        for h_post in range(poH):

            post_rank = post.rank_from_coordinates((w_post, h_post))

            values = []
            pre_ranks = []

            for w_pre in range(prW):
                for h_pre in range(prH):

                    saveVal = 0

                    # distance between 2 neurons
                    dist_w = (w_post-w_pre)**2
                    dist_h = (h_post-h_pre)**2
                    if ((mgd == 0) or ((dist_w < mgd*mgd) and (dist_h < mgd*mgd))):
                        val = factor * m_exp(-((dist_w+dist_h)/sigma/sigma))
                        # connect
                        pre_rank = pre.rank_from_coordinates((w_pre, h_pre))
                        values.append(val)
                        pre_ranks.append(pre_rank)

                        saveVal = val

                    if saveConnections:
                        strToWrite += "(" + str(w_pre) + "," + str(h_pre) + ") -> (" + str(w_post) + "," + str(h_post) + ") with " + str(saveVal) + "\n"

            synapse.add(post_rank, pre_ranks, values, [0])

    if saveConnections:
        f = open(saveDir+fn+'.txt', 'w')
        f.write(strToWrite)
        f.close()

    time1 = time.time()
    #print('created', synapse.nb_synapses, 'synapses out of ', poW*prW*poH*prH, ' in', time1-time0, 's')
    return synapse
