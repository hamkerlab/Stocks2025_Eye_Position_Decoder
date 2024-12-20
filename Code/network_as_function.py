from ANNarchy import *
# get own-defined connection pattern
from NewConnectionPattern import connect_gaussian1dTo2d_h, connect_gaussian1dTo2d_v,\
                                 connect_gaussian1dTo2d_diag, connect_gaussian2dTo1d_h,\
                                 connect_gaussian2dTo1d_diag, connect_gaussian2d_diagTo2d_v,\
                                 connect_all2all_exp1d, connect_all2all_exp2d, connect_gaussian1dTo2d_v_variable_delays, connect_gaussian1dTo2d_h_noisy_weights

# load parameters 

def network_setup(loadedParam):
    
    clear()


    Xr_Neurons = Neuron(
        name='Xr',
        parameters="""
            A = 'A_Xr' : population
            tau = 'tau_Xr' : population
            num_neurons = 'layer_size' : population
            K = 'K_Xr_Att' : population
            Baseline = 'Xr_Baseline_Activity' :population
            input = 0.0
        """,
        equations="""
            att = (input+Baseline) * (1 + sum(FF))   
            tau * dm/dt + m = att - m * K * mean(att)*num_neurons
            r = (m-0.05) : min=0.0
        """,
        extra_values=loadedParam
    )



    # Xe = Xe_PC
    Xe_Neurons = Neuron(
        name='Xe_PC',
        parameters="""
            tau = 'tau_Xe' : population
            input = 0.0
        """,
        equations="""
            tau * dr/dt + r = pos(input) : min=0.0, max=1.0
        """,
        extra_values=loadedParam
    )
    
    Xe2_Neurons = Neuron(
        name='Xe_CD',
        parameters="""
            tau = 'tau_Xe2' : population
            input = 0.0
        """,
        equations="""
            tau * dr/dt + r = pos(input) : min = 0.0, max = 1.0
        """,
        extra_values=loadedParam
    )

    Xe2r_Neurons = Neuron(
        name='Xe_FEF',
        parameters="""
            A = 'A_Xe2r' : population
            tau = 'tau_Xe2r' : population
        """,
        equations="""
            tau * dr/dt + r = sum(FF1)*sum(FF2) - r*sum(inh) : min = 0.0, max = 1.0
        """,
        extra_values=loadedParam
    )



    # Xb = Xb_PC    
    Xb_Neurons = Neuron(
        name='Xb_PC',
        parameters="""
            A = 'A_Xb' : population
            D = 'D_Xb' : population
            tau = 'tau_Xb' : population
            LIP_Noise = 'Noise_on_LIP' : population
            Noise_LIP_Low = 'Noise_LIP_Low' : population
            Noise_LIP_High = 'Noise_LIP_High' : population
            Xr_Faktor = 'XR_in_LIP_constant' : population
        """,
        equations="""
           	tau * dr/dt + r = if LIP_Noise == True: (sum(FF1)) * (Xr_Faktor +pos(A - max(r))*sum(FF2)) + sum(FB)*sum(FF2) + sum(exc) - (r + D) * sum(inh) + Normal(Noise_LIP_Low, Noise_LIP_High): min = 0.0, max = 2.0
            else:  (sum(FF1)) * (Xr_Faktor+pos(A - max(r))*sum(FF2)) + sum(FB)*sum(FF2) + sum(exc) - (r + D) * sum(inh) : min = 0.0, max = 2.0
        """,
        extra_values=loadedParam
    )
    
    
        # Xb2 = Xb_CD
    Xb2_Neurons = Neuron(
        name='Xb_CD',
        parameters="""
            A = 'A_Xb2' : population
            D = 'D_Xb2' : population
            tau = 'tau_Xb2' : population
            LIP_Noise = 'Noise_on_LIP' : population
            Noise_LIP_Low = 'Noise_LIP_Low' : population
            Noise_LIP_High = 'Noise_LIP_High' : population
            Xr_Faktor = 'XR_in_LIP_constant' : population
        """,
        equations="""
        	
           	tau * dr/dt + r = if LIP_Noise == True: (sum(FF1)) * (Xr_Faktor +pos(A - max(r))*sum(FF2)) + sum(FB)*sum(FF2) + sum(exc) - (r + D) * sum(inh) + Normal(Noise_LIP_Low, Noise_LIP_High): min = 0.0, max = 2.0
            else:  (sum(FF1)) * (Xr_Faktor+pos(A - max(r))*sum(FF2)) + sum(FB)*sum(FF2) + sum(exc) - (r + D) * sum(inh) : min = 0.0, max = 2.0
        """,
        extra_values=loadedParam
    )

    # Xh = Xh
    Xh_Neurons = Neuron(
        name='Xh',
        parameters="""
            D = 'D_Xh' : population
            tau = 'tau_Xh' : population
            dt_dep = 'dt_dep_Xh' : population
            tau_dep = 'tau_dep_Xh' : population
            d_dep = 'd_dep_Xh' : population
            baseline = 0.0
        """,
        equations="""
            input = sum(FF1) + sum(FF2) + baseline
            s = s + (input - s)*dt_dep/tau_dep
            S2 = 1-d_dep*s : min = 0.0, max = 1.0
            tau * dr/dt + r = input * S2 + sum(exc) - (r + D) * sum(inh) : min = 0.0, max = 1.0
        """,
        extra_values=loadedParam
    )


    ###############################
    #### Defining the synapses ####
    ###############################

    #don't need them because we do not learn the weights

    ##################################
    #### Creating the populations ####
    ##################################
    num_neurons = loadedParam['layer_size']

    # Xr = Xr
    Xr_Pop = Population(name='Xr', geometry=(num_neurons), neuron=Xr_Neurons)


    # Xe = Xe_PC
    Xe_Pop = Population(name='Xe_PC', geometry=(num_neurons), neuron=Xe_Neurons)
    if loadedParam['split_Xe']:
        # need extra layer for the projection to Xe_FEF ('copy' of Xe_PC)
        Xe_forFEF_Pop = Population(name='Xe_PC_forFEF', geometry=(num_neurons), neuron=Xe_Neurons)
    # Xe2 = Xe_CD
    Xe2_Pop = Population(name='Xe_CD', geometry=(num_neurons), neuron=Xe2_Neurons)
    # Xe2r = Xb_FEF
    Xe2r_Pop = Population(name='Xe_FEF', geometry=(num_neurons, num_neurons), neuron=Xe2r_Neurons)

    # Xb = Xb_PC
    Xb_Pop = Population(name='Xb_PC', geometry=(num_neurons, num_neurons), neuron=Xb_Neurons)
    # Xb2 = Xb_CD
    Xb2_Pop = Population(name='Xb_CD', geometry=(num_neurons, num_neurons), neuron=Xb2_Neurons)

    # Xh = Xh
    Xh_Pop = Population(name='Xh', geometry=(num_neurons), neuron=Xh_Neurons)

    ##################################
    #### Creating the projections ####
    ######################populations###########
    
    v = float(loadedParam['v']) # visual field
    max_gauss_distance = loadedParam['max_gauss_distance']/v


    ## to Xe2r = Xe_FEF ##
    # - FF (from Xe2)
    Xe2_Xe2r = Projection(
        pre=Xe2_Pop,
        post=Xe2r_Pop,
        target='FF1'
    ).connect_with_func(method=connect_gaussian1dTo2d_h, mv=loadedParam['Ke2e2r'],
                        radius=loadedParam['sigma_e2e2r']/v, mgd=max_gauss_distance)
    if loadedParam['split_Xe']:
        preLayer = Xe_forFEF_Pop
    else:
        preLayer = Xe_Pop
    Xe_Xe2r = Projection(
        pre=preLayer,
        post=Xe2r_Pop,
        target='FF2'
    ).connect_with_func(method=connect_gaussian1dTo2d_v_variable_delays, mv=loadedParam['Kee2r'],
                        radius=loadedParam['sigma_ee2r']/v, mgd=max_gauss_distance)
    # - inh (from Xe2r)
    Xe2r_inh = Projection(
        pre=Xe2r_Pop,
        post=Xe2r_Pop,
        target='inh'
    ).connect_all_to_all(weights=loadedParam['w_inh_Xe2r'])

    ## to Xb = Xb_PC ##
    # - FF (from Xr)
    Xr_Xb = Projection(
        pre=Xr_Pop,
        post=Xb_Pop,
        target='FF1'
    ).connect_with_func(method=connect_gaussian1dTo2d_h, mv=loadedParam['Krb'],
                        radius=loadedParam['sigma_rb']/v, mgd=max_gauss_distance)
    # - FF (from Xe)
    Xe_Xb = Projection(
        pre=Xe_Pop,
        post=Xb_Pop,
        target='FF2',
        name= "PC_to_LIP_PC"
    ).connect_with_func(method=connect_gaussian1dTo2d_v_variable_delays, mv=loadedParam['Keb'],
                        radius=loadedParam['sigma_eb']/v, mgd=max_gauss_distance)
    



    # - FB (from Xh)
    if loadedParam['Feedback_to_LIP_PC']:
        Xh_Xb = Projection(
            pre=Xh_Pop,
            post=Xb_Pop,
            target='FB'
        ).connect_with_func(method=connect_gaussian1dTo2d_diag, mv=loadedParam['Khb'],
                            radius=loadedParam['sigma_hb']/v, mgd=max_gauss_distance)
    # - exc (from Xb)
    Xb_exc = Projection(
        pre=Xb_Pop,
        post=Xb_Pop,
        target='exc'
    ).connect_with_func(method=connect_all2all_exp2d, factor=loadedParam['w_exc_Xb'],
                        radius=loadedParam['sigma_exc']/v, mgd=max_gauss_distance)
    # - inh (from Xb)
    Xb_inh = Projection(
        pre=Xb_Pop,
        post=Xb_Pop,
        target='inh'
    ).connect_all_to_all(weights=loadedParam['w_inh_Xb'])

    ## to Xb2 = Xb_CD ##
    # - FF (from Xr)
    Xr_Xb2 = Projection(
        pre=Xr_Pop,
        post=Xb2_Pop,
        target='FF1'
    ).connect_with_func(method=connect_gaussian1dTo2d_h, mv=loadedParam['Krb2'],
                        radius=loadedParam['sigma_rb2']/v, mgd=max_gauss_distance)
    # - FF (from Xe2r)

    Xe2r_Xb2 = Projection(
        pre=Xe2r_Pop,
        post=Xb2_Pop,
        target='FF2',
        name= "FEF_to_LIP_CD"
    ).connect_with_func(method=connect_gaussian2d_diagTo2d_v, mv=loadedParam['Ke2rb2'],
                        radius=loadedParam['sigma_e2rb2']/v, mgd=max_gauss_distance)

    # - FB (from Xh)
    Xh_Xb2 = Projection(
        pre=Xh_Pop,
        post=Xb2_Pop,
        target='FB'
    ).connect_with_func(method=connect_gaussian1dTo2d_diag, mv=loadedParam['Khb2'],
                        radius=loadedParam['sigma_hb2']/v, mgd=max_gauss_distance)
    # - inh (from Xb2)
    Xb2_inh = Projection(
        pre=Xb2_Pop,
        post=Xb2_Pop,
        target='inh'
    ).connect_all_to_all(weights=loadedParam['w_inh_Xb2'])

    ## to Xh = Xh ##
    # - FF (from Xb)
    Xb_Xh = Projection(
        pre=Xb_Pop,
        post=Xh_Pop,
        target='FF1'
    ).connect_with_func(method=connect_gaussian2dTo1d_diag, mv=loadedParam['Kbh'],
                        radius=loadedParam['sigma_bh']/v, mgd=max_gauss_distance)

    # - FB_ADD (from Xb2)
    Xb2_Xh = Projection(
        pre=Xb2_Pop,
        post=Xh_Pop,
        target='FF2'
    ).connect_with_func(method=connect_gaussian2dTo1d_diag, mv=loadedParam['Kb2h'],
                        radius=loadedParam['sigma_b2h']/v, mgd=max_gauss_distance)
    # - exc (from Xh)
    Xh_exc = Projection(
        pre=Xh_Pop,
        post=Xh_Pop,
        target='exc'
    ).connect_with_func(method=connect_all2all_exp1d, factor=loadedParam['w_exc_Xh'],
                        radius=loadedParam['sigma_exc']/v, mgd=max_gauss_distance)
    # - inh (from Xh)
    Xh_inh = Projection(
        pre=Xh_Pop,
        post=Xh_Pop,
        target='inh'
    ).connect_all_to_all(weights=loadedParam['w_inh_Xh'])


