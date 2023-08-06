#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 19:32:03 2022

@author: arhamze
"""
from obspy import UTCDateTime, Stream, read, Trace, read_inventory
from obspy.signal import rotate
from obspy.signal.invsim import estimate_wood_anderson_amplitude_using_response,estimate_wood_anderson_amplitude
import numpy as np
import os, glob, subprocess, sys
from pathlib import PurePath, Path
import matplotlib.pyplot as plt
from collections import defaultdict
import mtspec
import scipy
from tqdm import tqdm 


print('''
Python code to for rotating seismogram 
Developed by Arham Zakki Edelo

Before you run this program, make sure you have changed all the path correctly      
      ''')

######################## PARAMETERIZATION #######################
WATER_LEVEL=30                      #water level 
PRE_FILTER=[0.001, 0.002, 250, 255] #these values need to be customized for a specific bandwidth
DENSITY = 2700.0 # Rock density in km/m^3.
#Vp = 4566.92 # Velocities in m/s.
#Vs = Vp / 1.73
# How many seconds before and after the pick to choose for calculating the
# spectra.
TIME_BEFORE_PICK = 0.1
NOISE_PADDING=0.2

## for static 
TIME_AFTER_PICK_P = 0.45
TIME_AFTER_PICK_S=1.2

## for dinamic 
# P windowed time is 75% of S-P delay time
# S windowed time is 1.75# of s-P delay time

## nor noise spectra
NOISE_TIME=1.2

# Fixed quality factor. Very unstable inversion for it. Has almost no influence
# on the final seismic moment estimations but has some influence on the corner
# frequency estimation and therefore on the source radius estimation.
QUALITY_FACTOR = 100

## Velocity Model SEML###
layer_top=[ [-2,-1],[-1,-0.5],[-0.5,0],[0, 0.5],[0.5,1], [1,1.5], [1.5, 2], [2, 2.5],[2.5, 3],[3,5],[5,10], [10,15], [15,9999] ]
velocity_Vp=[3.64, 3.67, 3.86, 3.87, 3.91, 4.25, 4.35, 4.4, 4.41, 4.47, 4.5, 6.12, 7.92]
velocity_Vs=[2.11, 2.12, 2.15, 2.25, 2.44, 2.59, 2.65, 2.67, 2.69, 2.72, 2.74, 3.71, 4.81]

###### Fitting spectrum function ######
#### using levenberg-marquardt algorithm #####
def fit_spectrum(spectrum, frequencies, traveltime, initial_omega_0,
    initial_f_c):
    """
    Fit a theoretical source spectrum to a measured source spectrum.
    Uses a Levenburg-Marquardt algorithm.
    :param spectrum: The measured source spectrum.
    :param frequencies: The corresponding frequencies.
    :para traveltime: Event traveltime in [s].
    :param initial_omega_0: Initial guess for Omega_0.
    :param initial_f_c: Initial guess for the corner frequency.
    :param initial_q: initial quality factor
    :returns: Best fits and standard deviations.
        (Omega_0, f_c, Omega_0_std, f_c_std)
        Returns None, if the fit failed.
    """
    def f(frequencies, omega_0, f_c):
        return calculate_source_spectrum(frequencies, omega_0, f_c,
                QUALITY_FACTOR, traveltime)
    popt, pcov = scipy.optimize.curve_fit(f, frequencies, spectrum, \
        p0=list([initial_omega_0, initial_f_c]), maxfev=100000)        ### maxfev is the maximum number of function calls allowed during the optimization
    # p0 is the initial guest that will be optimized by the fit method
    # popt is the optimezed parameters and the pcov is the covariance matrix
    
    x_fit=frequencies
    y_fit= f(x_fit, *popt)
    
    if popt is None:
        return None
    return popt[0], popt[1], pcov[0, 0], pcov[1, 1], x_fit,y_fit


### function for calculating the source spectrum (spectrum model) ###
def calculate_source_spectrum(frequencies, omega_0, corner_frequency, Q,
    traveltime):

    """
    After Abercrombie (1995) and Boatwright (1980).
    Abercrombie, R. E. (1995). Earthquake locations using single-station deep
    borehole recordings: Implications for microseismicity on the San Andreas
    fault in southern California. Journal of Geophysical Research, 100,
    24003–24013.
    Boatwright, J. (1980). A spectral theory for circular seismic sources,
    simple estimates of source dimension, dynamic stress drop, and radiated
    energy. Bulletin of the Seismological Society of America, 70(1).
    The used formula is:
        Omega(f) = (Omege(0) * e^(-pi * f * T / Q)) / (1 + (f/f_c)^4) ^ 0.5
    :param frequencies: Input array to perform the calculation on.
    :param omega_0: Low frequency amplitude in [meter x second].
    :param corner_frequency: Corner frequency in [Hz].
    :param Q: Quality factor.
    :param traveltime: Traveltime in [s].
    """
    num = omega_0 * np.exp(-np.pi * frequencies * traveltime / Q)
    denom = (1 + (frequencies / corner_frequency) ** 4) ** 0.5
    return num / denom


#### function for calculating the moment magnitude ####
def calculate_moment_magnitudes(pick_cat,st,output_file_detail,output_file_collective,ID):
    """
    :param cat: obspy.core.event.Catalog object.
    """
    ## for plotting (Uncomment the code below to create image)
    fig, axs= plt.subplots((len(pick_cat)*6),2, figsize=(20,140) )
    plt.subplots_adjust(hspace=0.5) 
    axs[0,0].set_title("P Phase Spectra Profile", fontsize='20')
    axs[0,1].set_title("S Phase Spectra Profile", fontsize='20')
    
    ### parameterization for radiation pattern and K
    r_pattern_P=0.52;k_P=0.32
    r_pattern_S=0.63;k_S=0.21
    #velocity_P=Vp;velocity_S=Vs
    
    ## Holder value for average moments, source radius, and corner_frequencies from several stations
    moments_P = []
    moments_S = []
    
    source_radii_P = []
    source_radii_S = []
    
    corner_frequencies_P = []
    corner_frequencies_S = []
    
    ## holder value for omegas and corner_freq from 3 component of each station
    omegas_P=[]
    omegas_S=[]
    
    corner_freqs_P=[]
    corner_freqs_S=[]
    
    counter=0
    for tr in st:
        
        status=tr.stats
        station=status.station
        component=status.component

        #the dict format for referrence:
        #picking_holder={"STA_1":'P phase pick', 'S phase pick', 'origin time', 'source distance', 'depth'}
        
        try:
            
            ### Spectra Calculation for Data Window
            ## dinamic windowing 
            S_P_time=float(UTCDateTime(pick_cat[station][1]) - UTCDateTime(pick_cat[station][0]))
            TIME_AFTER_PICK_P = 0.80 * S_P_time
            TIME_AFTER_PICK_S = 1.75 * S_P_time
            
            ## determine the data index for windowing purpose P phases
            p_phase_first_index=int(round( (UTCDateTime(pick_cat[station][0]) - tr.stats.starttime )/ tr.stats.delta,4)) - \
            int(round(TIME_BEFORE_PICK / tr.stats.delta,4))
            
            p_phase_finish_index=int(round((UTCDateTime(pick_cat[station][0]) - tr.stats.starttime )/ tr.stats.delta,4))+ \
            int(round(TIME_AFTER_PICK_P / tr.stats.delta,4)) ## pick_cat[station][1] can be change with TIME_AFTER_PICK
            
            ## determine the data index for windowing purpose S phases
            s_phase_first_index=int(round( (UTCDateTime(pick_cat[station][1]) - tr.stats.starttime )/ tr.stats.delta,4))- \
            int(round(TIME_BEFORE_PICK / tr.stats.delta,4))
            
            s_phase_finish_index=int(round((UTCDateTime(pick_cat[station][1]) - tr.stats.starttime )/ tr.stats.delta,4))+ \
            int(round(TIME_AFTER_PICK_S / tr.stats.delta,4)) ## pick_cat[station][1] can be change with TIME_AFTER_PICK
            
            ## windowing the data for fitting spectrum and modelling
            P_windowed_data=tr.data[p_phase_first_index :p_phase_finish_index +1]
            S_windowed_data=tr.data[s_phase_first_index :s_phase_finish_index +1]

            ## calculate the spectrum
            spec_P, freq_P = mtspec.mtspec(P_windowed_data, tr.stats.delta, 2)
            spec_S, freq_S = mtspec.mtspec(S_windowed_data, tr.stats.delta, 2)
            
            ### Spectra Calculation for Noise Window
            ## noise window
            noise_first_index=int(round( (UTCDateTime(pick_cat[station][0]) - tr.stats.starttime )/ tr.stats.delta,4)) - \
            int(round( NOISE_TIME / tr.stats.delta,4))
            noise_finish_index=int(round( (UTCDateTime(pick_cat[station][0]) - tr.stats.starttime )/ tr.stats.delta,4)) - \
            int(round( NOISE_PADDING / tr.stats.delta,4))
            
            # noise spectrum, noise spectrum is set as long as S-P time before the P phase window
            noise_data = tr.data[noise_first_index:noise_finish_index+1]
            
            # calculate the noise spectrum
            spec_N, freq_N = mtspec.mtspec(noise_data, tr.stats.delta, 2)
            
            
        except Exception as e:
            print("Test mantepppp:",e )
            pass
        
        ## model the spectrum
        try:
            fit_P = fit_spectrum(spec_P, freq_P, UTCDateTime(pick_cat[station][0]) - UTCDateTime(pick_cat[station][2]),spec_P.max(), 10.0)  ### 10 is a initial corner frequency
            fit_S = fit_spectrum(spec_S, freq_S, UTCDateTime(pick_cat[station][1]) - UTCDateTime(pick_cat[station][2]),spec_S.max(), 10.0)  ### 10 is a initial corner frequency

            
            #print("Joss Gandoss:",fit_P,fit_S )
        except Exception as e:
            print("Test3 :",e)
            continue
        
        if fit_P is None and fit_S is None:
            continue
        else:
            pass
        
        ## fitting spectrum output
        Omega_0_P, f_c_P, err_P, _P , x_fit_P, y_fit_P= fit_P
        Omega_0_S, f_c_S, err_S, _S , x_fit_S, y_fit_S= fit_S
        
        ## append the fitting spectrum output to the holder list
        Omega_0_P = np.sqrt(Omega_0_P)
        Omega_0_S = np.sqrt(Omega_0_S)        
        
        omegas_P.append(Omega_0_P)
        omegas_S.append(Omega_0_S)
        
        corner_freqs_P.append(f_c_P)
        corner_freqs_S.append(f_c_S)
        
        ##### Uncomment to create plot display of spectra!!!!
        try:
            ### windowing trace data to be displayed
            tr_d=tr.copy()
            start_time=tr_d.stats.starttime
            before=(UTCDateTime(pick_cat[station][0]) - start_time) - 2.0
            after=(UTCDateTime(pick_cat[station][1])  - start_time) + 6.0
            tr_d.trim(start_time+before, start_time+after)
            start_time2=tr_d.stats.starttime
            station_plot=tr_d.stats.station
            component_plot=tr_d.stats.component

            ## plot
            # for P phase
            axs[counter][0].plot(tr_d.times(), tr_d.data, 'k')
            axs[counter][0].axvline( x= (UTCDateTime(pick_cat[station][0]) - start_time2 ), color='r', linestyle='-', label='P arrival')
            axs[counter][0].axvline( x= (UTCDateTime(pick_cat[station][1]) - start_time2 ), color='b', linestyle='-', label='S arrival')
            axs[counter][0].axvline( x= (UTCDateTime(pick_cat[station][0]) - TIME_BEFORE_PICK -  start_time2), color='g', linestyle='--')
            axs[counter][0].axvline( x= (UTCDateTime(pick_cat[station][0]) + TIME_AFTER_PICK_P - start_time2), color='g', linestyle='--', label='P phase window')
            axs[counter][0].axvline( x= (UTCDateTime(pick_cat[station][0]) - NOISE_TIME -  start_time2), color='gray', linestyle='--')
            axs[counter][0].axvline( x= (UTCDateTime(pick_cat[station][0]) - NOISE_PADDING  - start_time2), color='gray', linestyle='--', label='Noise window')
            axs[counter][0].set_title("{}_BH{}".format(station_plot, component_plot), loc="right",va='center')
            axs[counter][0].legend()
            axs[counter][0].set_xlabel("Relative Time (s)")
            axs[counter][0].set_ylabel("Amp (m)")
            
            # for s phase
            axs[counter][1].plot(tr_d.times(), tr_d.data, 'k')
            axs[counter][1].axvline( x= (UTCDateTime(pick_cat[station][0]) - start_time2 ), color='r', linestyle='-', label='P arrival')
            axs[counter][1].axvline( x= (UTCDateTime(pick_cat[station][1]) - start_time2), color='b', linestyle='-', label='S arrival')
            axs[counter][1].axvline( x= (UTCDateTime(pick_cat[station][1]) - TIME_BEFORE_PICK -  start_time2  ), color='g', linestyle='--')
            axs[counter][1].axvline( x= (UTCDateTime(pick_cat[station][1]) + TIME_AFTER_PICK_S - start_time2 ), color='g', linestyle='--', label='S phase window')
            axs[counter][1].axvline( x= (UTCDateTime(pick_cat[station][0]) - NOISE_TIME -  start_time2), color='gray', linestyle='--')
            axs[counter][1].axvline( x= (UTCDateTime(pick_cat[station][0]) - NOISE_PADDING  - start_time2), color='gray', linestyle='--', label='Noise window')
            axs[counter][1].set_title("{}_BH{}".format(station_plot, component_plot), loc="right",va='center')
            axs[counter][1].legend()
            axs[counter][1].set_xlabel("Relative Time (s)")
            axs[counter][1].set_ylabel("Amp (m)")
            
            ## Plot the spectra (P, S dan Noise spectra)
            counter+=1
            
            axs[counter][0].loglog(freq_P, spec_P, color='black', label='P spectra')
            axs[counter][0].loglog(freq_N, spec_N, color='gray', label='Noise spectra')
            axs[counter][0].loglog(x_fit_P, y_fit_P, 'b-', label='Fitted P Spectra')
            axs[counter][0].set_title("{}_BH{}".format(station_plot, component_plot), loc="right",va='center')
            axs[counter][0].legend()
            axs[counter][0].set_xlabel("Frequencies (Hz)")
            axs[counter][0].set_ylabel("Amp (m/Hz)")
            
            axs[counter][1].loglog(freq_S, spec_S, color='black', label='S spectra')
            axs[counter][1].loglog(freq_N, spec_N, color='gray', label='Noise spectra')
            axs[counter][1].loglog(x_fit_S, y_fit_S, 'b-', label='Fitted S Spectra')
            axs[counter][1].set_title("{}_BH{}".format(station_plot, component_plot), loc="right",va='center')
            axs[counter][1].legend()
            axs[counter][1].set_xlabel("Frequencies (Hz)")
            axs[counter][1].set_ylabel("Amp (m/Hz)")
            
            counter +=1
            
        except Exception as e:
            print(e)
            pass
         
        if component == "T":
            
            try:
                ### Find the correct velocity value for the spesific layer depth

                for i in range(len(layer_top)):
                    top_layer_limit=layer_top[i][0]
                    bottom_layer_limit=layer_top[i][1]
                    if top_layer_limit   <= pick_cat[station][4] <= bottom_layer_limit:
                        velocity_P = velocity_Vp[i]*1000  ### velocity in m/s
                        velocity_S = velocity_Vs[i]*1000  ### velocity in m/s
                        print("Alamak janggggg:", velocity_P,velocity_S  )
                    else:
                        pass
            except Exception as e:
                print("Testt joss gandoss XXX:", e)
                pass
            
            try:

                ## calculate seismic moment
                M_0_P = 4.0 * np.pi * DENSITY * velocity_P ** 3 * (pick_cat[station][3]*1000)* \
                        np.sqrt(omegas_P[0] ** 2 + omegas_P[1] ** 2 + omegas_P[2] ** 2) / \
                        (r_pattern_P * 2.0)                                                   ### should it be multipled by 2 ??
                        
                M_0_S = 4.0 * np.pi * DENSITY * velocity_S ** 3 * (pick_cat[station][3]*1000)* \
                        np.sqrt(omegas_S[0] ** 2 + omegas_S[1] ** 2 + omegas_S[2] ** 2) / \
                        (r_pattern_S * 2.0)                                                    ### should it be multipled by 2 ??
                
                ## calculate source radius
                r_P = 3 * k_P * velocity_P / sum(corner_freqs_P) ## times 3 becouse it is a three components
                r_S = 3 * k_S * velocity_S / sum(corner_freqs_S) ## times 3 becouse it is a three components
                
                ## appends for Mw calculations
                moments_P.append(M_0_P)
                moments_S.append(M_0_S)
                

                source_radii_P.append(r_P)
                source_radii_S.append(r_S)
                
                corner_frequencies_P.extend(corner_freqs_P)
                corner_frequencies_S.extend(corner_freqs_S)
                
                #### Write the output
                #### Format Event ID, Station, P_M0, S_M0,Src_rad_P, Src_rad_S
                output_file_detail.write("{},{},{:e},{:e}, {}, {}\n".format(int(ID),station,float(M_0_P),float(M_0_S),float(r_P),float(r_S)))
                
                
                ## clean the holder for 3 component spectral
                ## holder value for omegas and corner_freq from 3 component of each station
                omegas_P.clear()
                omegas_S.clear()
                
                corner_freqs_P.clear()
                corner_freqs_S.clear()
            except Exception as e:
                print("Test 4:",e)
                continue
        #print("jossss palalo gandosssss:",moments_P, moments_S)
            
        if not len (moments_P) or not len(moments_S):
            print("Cannot calculate Moment magnitude for this event!!!")
            continue
    
    # Calculate the seismic moment via basic statistics.
    moments_P = np.array(moments_P)
    moments_S = np.array(moments_S)
    
    moment_P = moments_P.mean()
    moment_S = moments_S.mean()
    
    moment_std_P = moments_P.std()
    moment_std_S = moments_S.std()
    
    ## calculate the corner frequencies via basic statistics.
    corner_frequencies_P = np.array(corner_frequencies_P)
    corner_frequencies_S = np.array(corner_frequencies_S)
    
    corner_frequency_P = corner_frequencies_P.mean()
    corner_frequency_S = corner_frequencies_S.mean()
    
    corner_frequency_std_P = corner_frequencies_P.std()
    corner_frequency_std_S = corner_frequencies_S.std()

    # Calculate the source radius.
    source_radii_P = np.array(source_radii_P)
    source_radii_S = np.array(source_radii_S)
    
    source_radius_P = source_radii_P.mean()
    source_radius_S = source_radii_S.mean()

    source_radius_std_P = source_radii_P.std()
    source_radius_std_S = source_radii_S.std()
    
    # Calculate the stress drop of the event based on the average moment and
    # source radii.
    stress_drop_P = (7 * moment_P) / (16 * source_radius_P ** 3)
    stress_drop_S = (7 * moment_S) / (16 * source_radius_S ** 3)
    
    stress_drop_std_P = np.sqrt((stress_drop_P ** 2) * (((moment_std_P ** 2) / (moment_P ** 2)) + \
    (9 * source_radius_P * source_radius_std_P ** 2)))
    
    stress_drop_std_S = np.sqrt((stress_drop_S ** 2) * (((moment_std_S ** 2) / (moment_S ** 2)) + \
    (9 * source_radius_S * source_radius_std_S ** 2)))
    
    
    if source_radius_P > 0 and source_radius_std_P < source_radius_P:
        print ("Source radius:", source_radius_P, " Std:", source_radius_std_P)
        print ("Stress drop:", stress_drop_P / 1E5, " Std:", stress_drop_std_P / 1E5)
    elif source_radius_S > 0 and source_radius_std_S < source_radius_S:
        print ("Source radius:", source_radius_S, " Std:", source_radius_std_S)
        print ("Stress drop:", stress_drop_S / 1E5, " Std:", stress_drop_std_S / 1E5)
    else:
        pass
        

    ## Calculate the moment magnitude
    
    Mw_p = ((2.0 / 3.0) * np.log10(moment_P)) - 6.07
    Mw_s = ((2.0 / 3.0) * np.log10(moment_S)) - 6.07
    Avg_Mw=(Mw_p + Mw_s)/2  ## the averaged Mw (Mw P and Mw S)
    
    Mw_std_P = 2.0 / 3.0 * moment_std_P / (moment_P * np.log(10))
    Mw_std_S = 2.0 / 3.0 * moment_std_S / (moment_S * np.log(10))
    
    ## Write the Average value on output file
    output_file_detail.write("Average:,,{:e},{:e},{:5.3f},{:5.3f}, Mw_P:,{:5.3f}, Mw_S:, {:5.3f}, Avg Mw:, {:5.3f} \n".\
                      format(float(moment_P), float(moment_S), float(source_radius_P), float(source_radius_S), \
                      float(Mw_p), float(Mw_s), float(Avg_Mw)   ))
    
    output_file_collective.write("{},{:5.3f}\n".format(int(ID), float(Avg_Mw)) )

    ##### Uncomment to save the spectra imgae
    fig.suptitle("Event {} Spectral Fitting Profile".format(ID), fontsize='24', fontweight='bold')
    #plt.title("Event {} Spectral Fitting Profile".format(ID), fontsize='20')
    plt.savefig("event_{}.png".format(ID))
    
    return None



###########       THE END OF THE FUNCIONS
########### HERE YOU RUN YOUR CODE ############
if __name__=="__main__":
    prompt=str(input('Please type yes/no if you had changed the path :'))
    if prompt != 'yes':
        sys.exit("Ok, please correct the path first!")
    else:
        print("Process the program ....\n\n")
        pass

    ########################################################
    
    # Initialize Input and save path
    rotated_waveform=Path(r"/run/media/arhamze/BGJX/SEML/DATA ROTATED/2023/2023 06/") ## rotated wave location
    Input2=Path(r"/run/media/arhamze/BGJX/SEML/DATA PROCESSING/MEQ MISCELANEOUS PROCESSING/NonLinLoc Results/Hasil NLL Revised/HASIL NLL 27 06 2023/UPDATE 27 06 2023/2023_06/")                     # NLL output data .hyp
    mw_output_path=Path(r"/run/media/arhamze/BGJX/SEML/DATA PROCESSING/MEQ MISCELANEOUS PROCESSING/MW CALCULATIONS/MW Magnitude/Mw 2023/2023_06/")
    inventory_path=Path(r"/run/media/arhamze/BGJX/SEML/DATA PROCESSING/MEQ MISCELANEOUS PROCESSING/Program Penting/SEISMOMETER INSTRUMENT CORRECTION/CALIBRATION/")
   
    ###########################################
    listOfStation=['M01','M02','M03','M04','M05','M06','M07','M08','M09','M10','M11','M12','M13','M14','M15'] ## Please pay attention to the 'L' words normally placed in station code
    
    #### READING THE RAW FIL
    ### Global search for all name file 
    FilesName=glob.glob(os.path.join(Input2, "*.grid0.loc.hyp"), recursive=True)
    FilesName.sort()
    
    ## ID start file
    ID_mw_starts= 2903
    
    ## reading data
    waves_name=glob.glob(os.path.join(rotated_waveform, "*.mseed"), recursive=True)
    waves_name.sort()
    
    with open(mw_output_path.joinpath(r"mw_2023_06_detail.csv"),'w') as mw_write_detail,open(mw_output_path.joinpath(r"mw_2023_06_collective.csv"),'w') as mw_write_collective :
        mw_write_detail.write("Event ID, Station, P_M0, S_M0,Src_rad_P, Src_rad_S\n")
        mw_write_collective.write("Event ID, Avg_Mw\n")
        
        # initiate progress bar 
        total_iterations = len(FilesName)
        progress_bar = tqdm(total=total_iterations, unit="iteration",bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
        for xx in range(len(FilesName)):
        
            ## Update the event ID
            ID_mw_starts+=1
            
            progress_bar.set_postfix(custom_postfix=xx+1)
            progress_bar.update(1)
            
            # dict to hold the pick values
            # the dict format:
            # picking_holder={"STA_1":'P phase pick', 'S phase pick', 'origin time', 'source distance','Depth','ID'}
            picking_holder=defaultdict(list)
            
            # for wav file name purpose and id name
            files_name=Path(FilesName[xx]).stem 
            wave_name="{:8d}_{:04d}_{:02d}".format(int(files_name[5:13]),int(files_name[14:18]),int(files_name[18:20]))
            
            ## Open the NLL result file
            with open (FilesName[xx], 'r') as pick_read:
                for line in pick_read:
                    line=line.split()
                    ## time format 2023-06-08T12:34:56 to convert to UTCDateTime
                    try:
                        if line[0] =='GEOGRAPHIC':
                            origine_time="{}-{:02d}-{:02d}T{:02d}:{:02d}:{:07.5f}". \
                            format(int(line[2]), int(line[3]), int(line[4]),int(line[5]), int(line[6]),float(line[7]))
                            depth=float(line[-1])
    
                        elif line[0] in listOfStation and line[4] == 'P':
                            line[0]=line[0][0]+'L'+line[0][1:]
                            picking_holder[line[0]].append("{}-{:02d}-{:02d}T{:02d}:{:02d}:{:07.5f}". \
                            format(int(line[6][:4]),int(line[6][4:6]),int(line[6][6:]),int(line[7][:2]),int(line[7][2:]), float(line[8])))
                            
                        elif line[0] in listOfStation and line[4] == 'S':
                            line[0]=line[0][0]+'L'+line[0][1:]
                            picking_holder[line[0]].append("{}-{:02d}-{:02d}T{:02d}:{:02d}:{:07.5f}". \
                            format(int(line[6][:4]),int(line[6][4:6]),int(line[6][6:]),int(line[7][:2]),int(line[7][2:]), float(line[8])))
                            picking_holder[line[0]].append(origine_time)
                            picking_holder[line[0]].append(float(line[22]))
                            picking_holder[line[0]].append(depth)
                            picking_holder[line[0]].append(ID_mw_starts)
    
                        else:
                            pass
                    except Exception as e:
                        pass

                pick_read.close()

            try:
                if wave_name==Path(waves_name[xx]).stem:
                    st=read(waves_name[xx])
                    sst=st.copy()
                    calculate=calculate_moment_magnitudes(picking_holder, sst, mw_write_detail,mw_write_collective, ID_mw_starts)
                else:
                    pass
                    
            except Exception as e:
                print("Test 5:",e)
                pass
                
            # Update the progress bar and add a custom postfix
        progress_bar.close()
    print('''
     _____  ____  ____   ____ _____ __ __    ___  ___        __  __  __ 
    |     ||    ||    \ |    / ___/|  |  |  /  _]|   \      |  ||  ||  |
    |   __| |  | |  _  | |  (   \_ |  |  | /  [_ |    \     |  ||  ||  |
    |  |_   |  | |  |  | |  |\__  ||  _  ||    _]|  D  |    |__||__||__|
    |   _]  |  | |  |  | |  |/  \ ||  |  ||   [_ |     |     __  __  __ 
    |  |    |  | |  |  | |  |\    ||  |  ||     ||     |    |  ||  ||  |
    |__|   |____||__|__||____|\___||__|__||_____||_____|    |__||__||__|
    '''
    )
                    