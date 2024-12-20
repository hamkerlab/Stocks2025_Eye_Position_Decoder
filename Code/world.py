import numpy
import math
import os

import sys
NO_STIM = sys.float_info.max



####################################
#### defining the input signals ####
####################################
#### calculate signals ####
def init_inputsignals(precalcParam,loadedParam,stim_pos,current_run,count='', subfolder=''):
    

    # load parameters from currently used parameter file. Good idea to make multiple worlds or add a conditional to which param 
    # file to load.




    duration = loadedParam['t_end']
    size = loadedParam['layer_size'] # =layer_width_ from Arnold
    # do we need a copy of Xe_PC for the projection to Xe_FEF? (signal not suppressed)
    needCopylayer = loadedParam['split_Xe']

    # initialize maps
    # is a saccade in progress?
    sac_in_progress = numpy.ones(duration)
    # suppression factor during saccades
    suppressionMap = numpy.ones(duration)
    # real eye position over time, eye focused at 0 degrees at beginning
    epMap = numpy.zeros(duration)
    # stimuli positions over time
    spMap = {}

    global xe_sig, xe2_sig, xh_sig, xr_sig, xo_sig
    # internal eye position at 0 degrees too
    xe_sig = numpy.ones((duration, 1)) * esig(size, 0, loadedParam['sigma_xe'], loadedParam['ce'])
    if needCopylayer:
        global xe_forFEF_sig
        xe_forFEF_sig = numpy.ones((duration, 1)) * esig(size, 0, loadedParam['sigma_xe'],loadedParam['ce'])
    xe2_sig = numpy.zeros((duration, size))     # corollary discharge signal
    xh_sig = numpy.zeros((duration, size))     # attention (on Xh) signal
    xr_sig = numpy.zeros((duration, size))     # retina signal
    xo_sig = numpy.zeros((duration, size))     # oculomotor signal

    ## pass one: gather events, generate internal eye position signal,
    #            oculomotor signal and suppression map ##
    for i in precalcParam['EVENTS']['order']:
        currentEvent = precalcParam['EVENTS'][i]
        t = currentEvent['time'] - loadedParam['t_begin']

        # event eyepos
        if currentEvent['type'] == 'EVENT_EYEPOS':
            eyepos = currentEvent['value']
            for j in range(t, duration):
                epMap[j] = eyepos
                xe_sig[j] = esig(size, eyepos, loadedParam['sigma_xe'], loadedParam['ce'])
                if needCopylayer:
                    xe_forFEF_sig[j] = esig(size, eyepos, loadedParam['sigma_xe'], loadedParam['ce'])

        # event saccade
        if currentEvent['type'] == 'EVENT_SACCADE':

            # 1. generate eyeposition signal Xe
            eye0 = epMap[t]                 # source position
            eye1 = currentEvent['value']    # target position


            # do we use a linear eye trajectory, or the WetterOpstal model?
            if loadedParam['better_eye_trajectory'] == 1:
                # we use the extended model from WetterOpstal
                m0= loadedParam['sac_curvature']
                if 'better_et_speed' in loadedParam:
                    vpk = loadedParam['better_et_speed']
                else:
                    vpk = 525.0/1000   # speed in deg / ms
                T = eye1-eye0           # saccade amplitude in deg
                # get direction of saccade
                if T < 0:
                    T = -T
                    direction = -1
                elif T == 0:
                    direction = 0
                else:
                    direction = 1

                use_this_eye_speed_for_sac_end = loadedParam['use_this_eye_speed_for_sac_end']
                A = 1.0 / (1.0 - math.exp(-T/m0))
                E = eye0
                saccade_in_progress = True
                for j in range(duration-t):
                    E_previous = E
                    if direction == 0:
                        E = eye0 #+   0   *(m0 * log((A * exp(vpk * j / m0)) / (1 + A * exp((vpk * j - T)/m0))))
                    else:
                        E = eye0 + direction*(m0 * math.log((A * math.exp(vpk * j / m0)) / (1.0 + A * math.exp((vpk * j - T)/m0))))
                    # detect saccade end
                    sac_has_ended = False
                    if use_this_eye_speed_for_sac_end != 0:
                        # saccade end is calculated by eye speed
                        # one time step is one ms, so now "current_eye_speed" is in deg/sec
                        current_eye_speed = math.fabs(E - E_previous)*1000
                        if ((current_eye_speed < use_this_eye_speed_for_sac_end) and (j > 0)):
                            sac_has_ended = True
                    else:
                        # saccade end is calculated by position
                        if math.fabs(E - eye1) < loadedParam['sac_offset_threshold']:
                            sac_has_ended = True
                    if sac_has_ended:
                        # saccade has ended
                        if 'better_et_simulte_entire_saccade' in loadedParam and loadedParam['better_et_simulte_entire_saccade']:
                            epMap[j+t] = E
                        else:
                            epMap[j+t] = eye1
                        if saccade_in_progress:
                            dur = j
                            saccade_in_progress = False
                    else:
                        # saccade still ongoing
                        sac_in_progress[j+t] = True
                        epMap[j+t] = E
            else:
                # we use the standard, linear model
                amplitude = eye1-eye0
                if 'sac_dur_slope' in loadedParam:
                    slope = loadedParam['sac_dur_slope']
                else:
                    slope = 5.0
                if 'sac_dur_intercept' in loadedParam:
                    intercept = loadedParam['sac_dur_intercept']
                else:
                    intercept = 10.0
                dur = intercept + amplitude*slope
                # generate trajectory in EP-map
                for j in range(t, min(t+dur, duration)):
                    sac_in_progress[j] = True
                    epMap[j] = eye0 + (j-t)*(eye1-eye0)/dur
                for j in range(t+dur, duration):
                    epMap[j] = eye1

            # save saccade duration
            saccade_duration = dur
            print("Saccade_Dur: ", saccade_duration)

            # generate Eye Position signal (EP-Signal, formerly "Xe tonic component")
            t_EP_off = t + loadedParam['EP_off']
            if 'EP_off_lock_to_offset' in loadedParam and loadedParam['EP_off_lock_to_offset']:
                t_EP_off += dur
            # bounded between 0 and duration
            t_EP_off = min(max(0, t_EP_off), duration)

            t_EP_on = t + loadedParam['EP_on']
            if 'EP_on_lock_to_offset' in loadedParam and loadedParam['EP_on_lock_to_offset']:
                t_EP_on += dur
            # bounded between 0 and duration
            t_EP_on = min(max(0, t_EP_on), duration)

            t_EP_supp_off = t + loadedParam['EP_supp_off']
            if 'EP_supp_off_lock_to_offset' in loadedParam and loadedParam['EP_supp_off_lock_to_offset']:
                t_EP_supp_off += dur
            # bounded between 0 and duration
            t_EP_supp_off = min(max(0, t_EP_supp_off), duration)

            t_EP_supp_on = t + loadedParam['EP_supp_on']
            if 'EP_supp_on_lock_to_offset' in loadedParam and loadedParam['EP_supp_on_lock_to_offset']:
                t_EP_supp_on += dur
            # bounded between 0 and duration
            t_EP_supp_on = min(max(0, t_EP_supp_on), duration)

            if loadedParam['Xe_noSuppression']:
                supp_strength = 1.0
            else:
                supp_strength = loadedParam['xe_supp_strength']

            if 'EP_signals_interact_by_max' in loadedParam:
                interact_by_max = loadedParam['EP_signals_interact_by_max']
            else:
                interact_by_max = 0
            threshold = loadedParam['xe_threshold_activity']
            ce = loadedParam['ce']

            xe0 = esig(size, eye0, loadedParam['sigma_xe'], ce)
            xe1 = esig(size, eye1 + loadedParam['EP_pos_post_offset'], loadedParam['sigma_xe'], ce)
            # remove previously existing EP-signal, also that part which has to be suppressed, if any
            for j in range(min(t_EP_off, t_EP_supp_on), duration):
                xe_sig[j] = numpy.zeros(size)
                if needCopylayer:
                    xe_forFEF_sig[j] = numpy.zeros(size)
            # recreate previously existing EP-signal with proper suppression, if applicable
            for j in range(t_EP_supp_on, t_EP_off):
                if (j > t_EP_supp_on) and (j < t_EP_supp_off):
                    supp = supp_strength
                else:
                    supp = 1.0

                if supp * ce > threshold:
                    xe_sig[j] += supp * xe0
                if needCopylayer and (ce > threshold):
                    xe_forFEF_sig[j] += xe0
            # add new EP-signal
            for j in range(t_EP_on, duration):
                if (j > t_EP_supp_on) and (j < t_EP_supp_off):
                    supp = supp_strength
                else:
                    supp = 1.0

                if supp * ce > threshold:
                    xe_sig[j] += supp * xe1
                if needCopylayer and (ce > threshold):
                    xe_forFEF_sig[j] += xe1
            # add Xe-decay
            if 'ep_off_decay_gaussian' in loadedParam:
                # we have a gaussian decay
                for j in range(t_EP_off+1, duration):
                    if(j > t_EP_supp_on) and (j < t_EP_supp_off):
                        supp = supp_strength
                    else:
                        supp = 1.0
                    factor = math.exp(-(pow(j-t_EP_off, 2.0))/(2.0*(pow(loadedParam['ep_off_decay_gaussian'], 2.0))))
                    xe_sig_new = xe0*factor
                    if interact_by_max:
                        if supp * ce * factor > threshold:
                            xe_sig[j] = numpy.maximum(xe_sig[j], supp * xe_sig_new)
                        if needCopylayer and (ce * factor > threshold):
                            xe_forFEF_sig[j] = numpy.maximum(xe_forFEF_sig[j], xe_sig_new)
                    else:
                        if supp * ce * factor > threshold:
                            xe_sig[j] += supp * xe_sig_new
                        if needCopylayer and (ce * factor > threshold):
                            xe_forFEF_sig[j] += xe_sig_new
            else:
                # we have a linear decay
                for j in range(t_EP_off, min(t_EP_off+loadedParam['ep_off_decay'], duration)):
                    if (j > t_EP_supp_on) and (j < t_EP_supp_off):
                        supp = supp_strength
                    else:
                        supp = 1.0
                    factor = (loadedParam['ep_off_decay']-(j-t_EP_off))/float(loadedParam['ep_off_decay'])
                    xe_sig_new = xe0*factor
                    if interact_by_max:
                        if supp * ce * factor > threshold:
                            xe_sig[j] = numpy.maximum(xe_sig[j], supp * xe_sig_new)
                        if needCopylayer and (ce * factor > threshold):
                            xe_forFEF_sig[j] = numpy.maximum(xe_forFEF_sig[j], xe_sig_new)
                    else:
                        if supp * ce * factor > threshold:
                            xe_sig[j] += supp * xe_sig_new
                        if needCopylayer and (ce * factor > threshold):
                            xe_forFEF_sig[j] += xe_sig_new
            # add Xe gaussian buildup
            if 'ep_on_buildup_gaussian' in loadedParam:
                for j in range(t_EP_on-1, 0, -1):
                    if (j > t_EP_supp_on) and (j < t_EP_supp_off):
                        supp = supp_strength
                    else:
                        supp = 1.0
                    factor = math.exp(-(pow(t_EP_on-j, 2.0))/(2.0*(pow(loadedParam['ep_on_buildup_gaussian'], 2.0))))
                    xe_sig_new = xe1*factor
                    if interact_by_max:
                        if supp * ce * factor > threshold:
                            xe_sig[j] = numpy.maximum(xe_sig[j], supp * xe_sig_new)
                        if needCopylayer and (ce * factor > threshold):
                            xe_forFEF_sig[j] = numpy.maximum(xe_forFEF_sig[j], xe_sig_new)
                    else:
                        if supp * ce * factor > threshold:
                            xe_sig[j] += supp * xe_sig_new
                        if needCopylayer and (ce * factor > threshold):
                            xe_forFEF_sig[j] += xe_sig_new
            # add Xe linear buildup
            if 'ep_on_buildup_linear' in loadedParam:
                if 'ep_on_buildup_linear_duration' in loadedParam:
                    dur2 = loadedParam['ep_on_buildup_linear_duration']
                else:
                    dur2 = loadedParam['ep_on_buildup_linear']
                for j in range(t_EP_on-1, max(0, t_EP_on-dur2), -1):
                    if (j > t_EP_supp_on) and (j < t_EP_supp_off):
                        supp = supp_strength
                    else:
                        supp = 1.0
                    factor = 1.0-(t_EP_on - j)/float(loadedParam['ep_on_buildup_linear'])
                    xe_sig_new = xe1*factor
                    if interact_by_max:
                        if supp * ce * factor > threshold:
                            xe_sig[j] = numpy.maximum(xe_sig[j], supp * xe_sig_new)
                        if needCopylayer and (ce * factor > threshold):
                            xe_forFEF_sig[j] = numpy.maximum(xe_forFEF_sig[j], xe_sig_new)
                    else:
                        if supp * ce * factor > threshold:
                            xe_sig[j] += supp * xe_sig_new
                        if needCopylayer and (ce * factor > threshold):
                            xe_forFEF_sig[j] += xe_sig_new
            

        #event stimulus
        if currentEvent['type'] == 'EVENT_STIMULUS':
            nameOfEvent = currentEvent['name']
            if nameOfEvent not in spMap:
                spMap[nameOfEvent] = numpy.ones(duration)*NO_STIM # stimulus off until now
            for j in range(t, duration):
                #print(currentEvent['value'] )
                spMap[nameOfEvent][j] =currentEvent['value'] # stimulus on


    ## pass two: generate Corollary Discharge Signal (CD-Signal, formerly 'phasic xe signal') ##
    if loadedParam['xe_phasic_exists']:
        for i in precalcParam['EVENTS']['order']:
            currentEvent = precalcParam['EVENTS'][i]
            t = currentEvent['time'] - loadedParam['t_begin']

            # event saccade
            if currentEvent['type'] == 'EVENT_SACCADE':
                eye0 = epMap[t]                 # source position
                eye1 = currentEvent['value']    # target position

                # add a user-supplied offset to the spatial position of CD signal
                eye1 += loadedParam['xe_phasic_pos_offset']
                # generate CD-Signal
                t_xe_phasic = t + loadedParam['xe_phasic_time']
                for j in range(t_xe_phasic-loadedParam['xe_phasic_range'],
                               t_xe_phasic+loadedParam['xe_phasic_range']):
                    if ((j >= 0) and (j < duration)):
                        # CD-Signal is retinotopic
                        xe = osig(size, eye1-eye0, j - t_xe_phasic, loadedParam['xe_phasic_sigma'],
                                  loadedParam['xe_phasic_alpha'], loadedParam['xe_phasic_beta'], loadedParam['xe_phasic_strength'])
                        # are we above the (optional) threshold?
                        # otherwise no CD-Signal will be generated...
                        under_threshold = numpy.all(xe <= loadedParam['xe_phasic_threshold_activity']*numpy.ones(size))
                        if not under_threshold:
                            xe2_sig[j] += xe


    ## pass three: generate attention signal ##
    if loadedParam['attention_on_Xh']:
        # generate attention
        if loadedParam['num_of_attention'] == 1:
            # one attention spot
            for j in range(loadedParam['attention_start'], loadedParam['attention_end']):
                if (j >= 0) and (j < duration):
                    xh_sig[j] = esig(size, loadedParam['attention_position'],
                                     loadedParam['attention_sigma'],
                                     loadedParam['attention_strength'])
        else:
            # more than one attention spot
            for i in range(loadedParam['num_of_attention']):
                for j in range(loadedParam['attention_start'][i], loadedParam['attention_end'][i]):
                    if (j >= 0) and (j < duration):
                        xh_sig[j] += esig(size, loadedParam['attention_position'][i],
                                          loadedParam['attention_sigma'],
                                          loadedParam['attention_strength'])







    ## pass four: calculation of retina signal from real world data ##
    cr = float(loadedParam['cr'])
    told = 0
    stimpos_old = []
    eyepos_old = 0
    signal_old = numpy.zeros(size)
    for tnew in range(duration-1):
        if told > duration-1:
            break
        stimpos = []
        strength = []
        for it in spMap:
            a_stimpos = spMap[it][tnew]
            if a_stimpos != NO_STIM:
                stimpos.append(a_stimpos)
                strength.append(cr)
        eyepos = epMap[tnew]
        signal = rsig(size, eyepos, stimpos, strength)
        if ((sorted(stimpos) != sorted(stimpos_old)) or (eyepos != eyepos_old) or (tnew == duration-2)):  #this is the original version
        #if ((sorted(stimpos) != sorted(stimpos_old)) or (tnew == duration-2)):    
            # use latency
            xr_start = told + loadedParam['xr_latency']
            if xr_start >= duration-1:
                break
            xr_end = min(tnew+loadedParam['xr_latency'], duration-1)
            # 1. generate the actual stimulus (using suppression and depression)
            for t in range(xr_start, xr_end):
                # use depression
                depr = xr_depr(t-xr_start, loadedParam['xr_tau'], loadedParam['xr_d'])
                xr_sig[t] = numpy.maximum(xr_sig[t], signal_old * suppressionMap[t] * depr)
            # 2. generate the Xr signal after stimulus release (again using suppression and depression)
            if xr_end < duration-1:

                suppression = 1.0
                decay = min(duration-1-xr_end, loadedParam['xr_decay'])
                for t in range(xr_end, xr_end+decay):
                    # use depression
                    depr = xr_depr(t-xr_start, loadedParam['xr_tau'], loadedParam['xr_d'])
                    # if the end of a suppression phase is within the decay,
                    # prevent the activity from rising after the suppression phase by extending
                    # the suppression until decay is over
                    suppression = min(suppression, suppressionMap[t])
                    xr_sig[t] = numpy.maximum(xr_sig[t],
                                              signal_old*suppression*depr*(1-loadedParam['xr_decay_rate']*(t-xr_end)))
            told = tnew
        signal_old = signal
        stimpos_old = stimpos
        eyepos_old = eyepos

    ## finished ##

    # save inputs if wanted
    if 'save_inputs' in loadedParam and loadedParam['save_inputs']:
        # save the new generated inputs as an txt-file
        #xr_sig, xe_sig, (xe_forFEF_sig,) xe2_sig, xh_sig
        #print 'save inputs','for',loadedParam['sigma_xe'],precalcParam['sigma_xe'],count,subfolder
        saveDirRates = "../data/Input_Signals/"
        save_inputs(xr_sig, xr_sig.shape, saveDirRates + '_xr_input.txt')
        save_inputs(xe_sig, xe_sig.shape, saveDirRates + '_xe_input.txt')
        if needCopylayer:
            save_inputs(xe_forFEF_sig, xe_forFEF_sig.shape, saveDirRates + '_xe_forFEF_input.txt')
        save_inputs(xe2_sig, xe2_sig.shape, saveDirRates + '_xe2_input.txt')
        save_inputs(xh_sig, xh_sig.shape, saveDirRates + '_xh_input.txt')
       

    if 'save_eyePosition' in loadedParam and loadedParam['save_eyePosition']:
        #print 'save eye position'
        epMap[0] = loadedParam['Fixation']
        
        saveDirEP = "../data/Eye_Pos/"
        print("AKTUELLES SAKKADEN SAFEDIR IST: ", saveDirEP)
        filename = saveDirEP + str(stim_pos)+"_" + str(current_run) + '_eyepos.txt'
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        strToWrite = ''
        for t in range(duration):
            strToWrite += str(t) + ' ' + str(epMap[t]) + '\n'
        f = open(filename, 'w')
        f.write(strToWrite)
        f.close()


    # return
    # summarize inputs in dictionary
    signals = {'xe_sig': xe_sig, 'xe2_sig': xe2_sig, 'xh_sig': xh_sig, 'xr_sig': xr_sig}
    if needCopylayer:
        signals['xe_forFEF_sig'] = xe_forFEF_sig
    
    numpy.save("../data/Input_Signals/signals.npy", signals)
    # return
    return signals


#############################
#### auxiliary functions ####
#############################
# Returns an internal eye position signal given the head-centered eye position in degrees.
# also used for top-down attention
def esig(width, ep, sigma, strength):
    # Eq 5: r^Xe_PC,in = strength_PC * exp(...)
    xe = []
   
    for i in range(width):
        xe.append(strength*math.exp(-(pow(math.fabs(ep-idx_to_deg(i, width)), 2)) / (2.0*(pow(sigma, 2)))))

    return numpy.array(xe)

# Returns an internal corollary discharge signal given the eye-centered saccade target in degrees.
# also used for oculomotor signal
def osig(width, st, t, sigma, alpha, beta, strength):
    # Eq 14: r^Xe_CD,in = strength_CD * exp(...) * S_CD(t)
    xo = []


    for i in range(width):
        # gauss-Version
        if t <= 0:
            # S_CD rises
            sig_S = alpha
        else:
            # S_CD decays
            sig_S = beta
        S = math.exp(-(pow(t, 2.0))/(2.0*(pow(sig_S, 2))))
        xo.append(strength * math.exp(-(pow(math.fabs(st-idx_to_deg(i, width)), 2))/(2.0*(pow(sigma, 2)))) * S)

    return numpy.array(xo)

# Returns a retina signal given the head-centered eye position in degrees and the head-centered stimulus positions in degrees
def rsig(width, eyepos, hc_stimpos, strengthPerStimpos):
    # see Eq 1: r^Xr,in = (S^Xr) * strength_r * exp(...)
    xr = numpy.zeros(width)

    numOfStims = len(hc_stimpos)
#    for it in hc_stimpos:
    for s in range(numOfStims):
        it = hc_stimpos[s]
        strength = strengthPerStimpos[s]
        ec_stimpos = -eyepos + it
        for i in range(width):
            xr[i] += strength * math.exp(-(pow(math.fabs(ec_stimpos-idx_to_deg(i, width)), 2)) / (2.0*(pow(sigma_xr(ec_stimpos), 2))))

    return xr


# Maps a cell index i to a position in visual space
def idx_to_deg(i, size):
    if size == 21 or size == 41:
        size -= 1
    return 80*(float(i)/size-0.5)

# "width" or receptive field
def sigma_xr(pos):
    #return (0.0875*math.fabs(pos) + 6.3500)  /2 
    return (0.0875*math.fabs(0) + 6.3500)  /2     #fixed RF Size
    
    
# "width" of xo signal
def sigma_xo(sac_amplitude):
    return (-0.265 * math.fabs(sac_amplitude) + 9.9987) /2

# Returns the depression factor for Xr where the differential equation is statically solved
#  s.t. xr_depr(0) = 1
def xr_depr(t, tau, d):
    I = 1.0
    a = tau * math.log((I + 1.0)/I)
    return 1.0 - d * (math.exp(-t/tau) + I - I*math.exp((a-t)/tau))


# save the inputs
def save_inputs(signal, size, filename):
    saveDirInputs = os.path.dirname(filename)
    if not os.path.exists(saveDirInputs):
        os.makedirs(saveDirInputs)

    strToWrite = ''
    for i in range(size[0]):
        for j in range(size[1]):
            strToWrite += "{0},{1}: {2}\n".format(i, j, signal[i][j])

    f = open(filename, 'w')
    f.write(strToWrite)
    f.close()


#######################################
#### Definition of the environment ####
#######################################
def set_input(t, signals, populations):
    # set the input for each population for the current timestep t
    # t = act_time - 't_begin'
    for pop in populations:
        if pop.name == "Xr":
            pop.input = signals['xr_sig'][t]
            
        if pop.name == "Xe_PC":
            pop.input = signals['xe_sig'][t]
        
        if pop.name == "Xe_PC_forFEF":
            pop.input = signals['xe_forFEF_sig'][t]
            
        if pop.name == "Xe_CD":
            pop.input = signals['xe2_sig'][t]
            
        if pop.name == "Xh":
            pop.baseline = signals['xh_sig'][t]
