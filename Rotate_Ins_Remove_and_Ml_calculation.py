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
#import mtspec

print('''
Python code to for rotating seismogram 
Developed by Arham Zakki Edelo

Before you run this program, make sure you have changed all the path correctly      
      ''')
      
######################## PARAMETERIZATION #######################
WATER_LEVEL=20                      #water level 
PRE_FILTER=[0.1, 0.25, 247.25, 250] #these values need to be customized for a specific bandwidth
DENSITY = 2700.0 # Rock density in km/m^3.
Vp = 4566.92 # Velocities in m/s.
Vs = Vp / 1.73
# How many seconds before and after the pick to choose for calculating the
# spectra.
TIME_BEFORE_PICK = 0.2
TIME_AFTER_PICK = 0.8
PADDING = 20
# Fixed quality factor. Very unstable inversion for it. Has almost no influence
# on the final seismic moment estimations but has some influence on the corner
# frequency estimation and therefore on the source radius estimation.
QUALITY_FACTOR = 1000

##################   LIST OF FUNCTIONS   ########################
################## Wave Preparation Functions ###################
#### rotate functions
def Rotate(st, back_azim_holder, station_holder): ## rotation function
    try:
        key_list=list(back_azim_holder.keys())
        key_list.sort()
        sorted_holder={v:back_azim_holder[v] for v in key_list}
    except Exception:
        pass
    sst_rotated=Stream()
    try:
        for tr in st:
            tr_component=tr.stats
            #tr.detrend("demean")
            if tr_component.station in station_holder: # it will make sure that only the respected recorded station will be used
                if tr_component.component == 'E':
                    tr_E=tr
                    tr_component.component="R"
                    tr_component_R=tr_component
                elif tr_component.component == 'N':
                    tr_N=tr
                    tr_component.component="T"
                    tr_component_T=tr_component
                elif (tr_component.station + "_BHZ") in sorted_holder.keys() and tr_component.component=='Z':
                    tr_component=tr_component.station + "_BHZ"
                    b_azim=sorted_holder[tr_component]
                    sst_rotated+=tr
                    tr_R,tr_T=rotate.rotate_ne_rt(tr_N.data,tr_E.data,b_azim)
                    tr_R=Trace(tr_R);tr_T=Trace(tr_T)
                    tr_R.stats.update(tr_component_R);tr_T.stats.update(tr_component_T)
                    sst_rotated+=tr_R
                    sst_rotated+=tr_T
                else:
                    pass
            else:
                pass
    except Exception as e:
        print(e)
        pass
    return sst_rotated

### intrument response remove function
def instrument_remove (st,inventory_path, ID):
    st_removed=Stream()
    for tr in st:
        try:
            tr_status=tr.stats;station=tr_status.station; component=tr_status.component
            ## so first we need to remove the mean
            inv_path=inventory_path.joinpath("RESP.ML.{}..BH{}".format(station,component))
            inv=read_inventory(inv_path, format='RESP')
            
            # pre tapering and detrending 
            tr.detrend("demean")
            tr.detrend("linear")
            tr.taper(0.05,type='cosine',max_length=None, side='both') 
            
            ## remove response
            ## be cautious with the water level parameter
            rtr=tr.remove_response(inventory=inv, pre_filt=PRE_FILTER,water_level=WATER_LEVEL,output='DISP',zero_mean=False, taper=False, plot="fig_{}_BH{}".format(station, component)) # pre_filt=PRE_FILTER optional plot=False or "fig_{}_BH{}".format(station, component)
            
            ## re-detrending
            #rtr.detrend("linear")
            
            st_removed+=rtr
            #rtr.plot(outfile="fig_{}_BH{}".format(station, component))
        except Exception as e:
            print(e)
            pass
    return st_removed
        
### write the signal function
def WriteSeis(SaveDir,st2,FilesName):  ## write the waveform function
    s=st2[0].stats
    for tr in st2:
        tr.data=np.require(tr.data, dtype=np.float32)  ### this code will decrease the size of the written data
    st2.write(os.path.join(SaveDir,FilesName),format='mseed')
    return s

######################## Magnitude and Spectrum Calculation ############################

#### Estimate the Local Magnitude Function ######
def calculate_peak2peak_and_period (tr_raw, time_pick_P, time_pick_S):
    time_before_windowing=0.0
    time_after_windowing=1.75
    time_padding_af_first_max_amp=0.0
    dl_time_af_first_max_amp=0.075 ### the maximum time delay after the first max amplitude occurance (to find the end index) 
    dl_time_bf_first_max_amp=0.075
    try:
        ## pre windowing (peak only pick in S wave phases)
        tr=tr_raw.copy()
        start_time=tr.stats.starttime
        #p_pick_second=(time_pick_P - start_time) - time_before_windowing
        s_pick=(time_pick_S - start_time) - time_before_windowing
        s_pick_plus_padding=(time_pick_S - start_time) + time_after_windowing
        tr.trim(start_time+s_pick, start_time+s_pick_plus_padding )
        
        # Start calculating timespan and peak amplitude
        # Locate first Max amplitude
        first_max_amplitude=tr.max() ## find the first max amplitude
        first_max_amplitude_index=tr.data.tolist().index(first_max_amplitude) ## find the index where the max amplitude happened
        first_max_amplitude_raw_time=tr.times()[first_max_amplitude_index] ## find the time where the max amplitude happened, the times() only return the non original timestamp
        start_time=tr.stats.starttime; 
        
        # find start index for first peak and end index for second peak for windowing purposes
        first_peak_index = int(round((first_max_amplitude_raw_time + time_padding_af_first_max_amp)  / tr.stats.delta,4))
        index_after_first_peak=int(round((first_max_amplitude_raw_time + dl_time_af_first_max_amp)/tr.stats.delta,4))
        index_before_first_peak=int(round((first_max_amplitude_raw_time - dl_time_bf_first_max_amp)/tr.stats.delta,4))
        
        # windowing the data
        # windowing after first peak
        pre_windowed_af_data= tr.data[first_peak_index:index_after_first_peak+1]
        data_after_first_max_amplitude=Trace(data=pre_windowed_af_data,header=tr.stats)
        windowed_start_time_after = UTCDateTime(tr.stats.starttime) + round(first_peak_index/ tr.stats.sampling_rate,4)  ### index divided by sampling rate will return second
        data_after_first_max_amplitude.stats.starttime=windowed_start_time_after
        
        # windowing before first peak
        pre_windowed_bf_data= tr.data[index_before_first_peak:first_peak_index+1]
        data_before_first_max_amplitude=Trace(data=pre_windowed_bf_data,header=tr.stats)
        windowed_start_time_before = UTCDateTime(tr.stats.starttime) + round(index_before_first_peak / tr.stats.sampling_rate,4)  ### index divided by sampling rate will return second
        data_before_first_max_amplitude.stats.starttime=windowed_start_time_before 
        

        ## Locate the second Peak
        if first_max_amplitude > 0:
            # raw_data_in_list=data_after_first_max_amplitude.data.tolist()
            
            # min_amp = raw_data_in_list[0]
            # min_amp_index = None

            # for i, amp in enumerate(raw_data_in_list):
                # if amp < min_amp:
                    # min_amp = amp
                    # min_amp_index = i
                # elif amp > min_amp:
                    # break  # Break the loop after finding the first maximum value
            # second_max_amplitude_raw_time=data_after_first_max_amplitude.times()[min_amp_index]
            # second_max_amplitude=min_amp
            try:
                scn_mx_amp_after_raw_time=data_after_first_max_amplitude.times()[data_after_first_max_amplitude.data.argmin()]
                scn_mx_amp_after=data_after_first_max_amplitude[data_after_first_max_amplitude.data.argmin()]
                
                scn_mx_amp_before_raw_time=data_before_first_max_amplitude.times()[data_before_first_max_amplitude.data.argmin()]
                scn_mx_amp_before=data_before_first_max_amplitude[data_before_first_max_amplitude.data.argmin()]
                
                if scn_mx_amp_after < scn_mx_amp_before:
                    second_max_amplitude_raw_time=scn_mx_amp_after_raw_time
                    second_max_amplitude=scn_mx_amp_after
                    pick_value_2=UTCDateTime(data_after_first_max_amplitude.stats.starttime) + second_max_amplitude_raw_time  ## pick value 2
                    
                else:
                    second_max_amplitude_raw_time=scn_mx_amp_before_raw_time
                    second_max_amplitude=scn_mx_amp_before
                    pick_value_2=UTCDateTime(data_before_first_max_amplitude.stats.starttime) + second_max_amplitude_raw_time  ## pick value 2
            except Exception as e:
                pass
                    
                    
                
        elif first_max_amplitude < 0:
        
            # raw_data_in_list=data_after_first_max_amplitude.data.tolist()
        
            # max_amp = raw_data_in_list[0]
            # max_amp_index = None

            # for i, amp in enumerate(raw_data_in_list):
                # if amp > max_amp:
                    # max_amp = amp
                    # max_amp_index = i
                # elif amp < max_amp:
                    # break  # Break the loop after finding the first maximum value
            # second_max_amplitude_raw_time=data_after_first_max_amplitude.times()[max_amp_index]
            # second_max_amplitude=max_amp

            try:
                scn_mx_amp_after_raw_time=data_after_first_max_amplitude.times()[data_after_first_max_amplitude.data.argmax()]
                scn_mx_amp_after=data_after_first_max_amplitude[data_after_first_max_amplitude.data.argmax()]
                
                scn_mx_amp_before_raw_time=data_before_first_max_amplitude.times()[data_before_first_max_amplitude.data.argmax()]
                scn_mx_amp_before=data_before_first_max_amplitude[data_before_first_max_amplitude.data.argmax()]
                
                if scn_mx_amp_after > scn_mx_amp_before:
                    second_max_amplitude_raw_time=scn_mx_amp_after_raw_time
                    second_max_amplitude=scn_mx_amp_after
                    pick_value_2=UTCDateTime(data_after_first_max_amplitude.stats.starttime) + second_max_amplitude_raw_time  ## pick value 2
                else:
                    second_max_amplitude_raw_time=scn_mx_amp_before_raw_time
                    second_max_amplitude=scn_mx_amp_before
                    pick_value_2=UTCDateTime(data_before_first_max_amplitude.stats.starttime) + second_max_amplitude_raw_time  ## pick value 2
            except Exception as e:
                pass
        else:
            pass
         
        ### for plotting purpose only
        #pick_value_1=windowed_start_time ### pick value 1
        pick_value_1=UTCDateTime(start_time) + first_max_amplitude_raw_time
        
            
        ## calculating Peak to Peak and Timespan
        if pick_value_2 - pick_value_1 > 0:
            timespan = pick_value_2 - pick_value_1
        else :
            timespan=pick_value_1 - pick_value_2
        peak_to_peak=abs(first_max_amplitude) + abs(second_max_amplitude) ## absolute value of peak_to_peak
    except Exception as e:
        pass
    #print("Test fixx:",pick_value_1,pick_value_2)
    return timespan, peak_to_peak,pick_value_1,pick_value_2

#### Funcion to calculate Ml magnitude ######
def calculate_ml_magnitude (inventory_path, st,pick_and_distance, ml_write_files_detail,ml_write_files_collective, ID):
    
    #params_qd=[1.11, 0.00189, -1.99] ### distance correction parameters
    params_qd=[1.11, 0.00189, -2.09] ### distance correction parameters
    datetime = UTCDateTime(2020, 1, 1, 1, 0, 0)
    fig, axs=plt.subplots(len(pick_and_distance.keys()),1, figsize=(26,20)) ### for plotting purposes
    plt.subplots_adjust(hspace=0.3)                                ### for plotting purposes
    count=0
    
    ml_magnitudes=[]
    
    for tr in st:
        
        try:
            tr_status=tr.stats; component=tr_status.component
            if component=='Z':
                station=tr_status.station
                station_name=station + "_BHZ"
                src_distance=pick_and_distance[station_name][-1]
                inv_path=inventory_path.joinpath("RESP.ML.{}..BH{}".format(station,component))
                inv=read_inventory(inv_path, format="RESP")
                instrument_response=inv[0][0][0]
                
                ## Get PAZ from response file
                
                #paz_print=instrument_response.response.get_paz()
                #print(paz_print)
                #paz=instrument_response.response.get_paz()
                #paz = {'poles': [-8.8857-8.8857j, -8.8857+8.8857j],
                #          'zeros': [0+0j, 0+0j],
                #            'gain': 1.0, 'sensitivity': 1579070000}
                #print(paz)
                
                ### get response file from inventroy
                #response=inv.get_response("ML.{}..BH{}".format(station,component),datetime )
                
                ## Get the first P phase pick time and S phase pick time for time windowing
                P_pick=UTCDateTime(pick_and_distance[station_name][0])
                S_pick=UTCDateTime(pick_and_distance[station_name][1])
                
                
                #calculate timespan and peak to peak by calling the calculate peak to peak function
                timespan, peak_to_peak, pick_value_1, pick_value_2 = calculate_peak2peak_and_period(tr, P_pick, S_pick)

                
                #w_andersoon_amplitude=estimate_wood_anderson_amplitude(paz, peak_to_peak,timespan) ## estimate the wood andrs amplitude
                w_andersoon_amplitude=estimate_wood_anderson_amplitude_using_response(instrument_response.response, peak_to_peak,timespan)

                #ml_magnitude=np.log10(w_andersoon_amplitude*1000) + (params_qd[0] * np.log10(src_distance)) + (params_qd[1] * src_distance) - ((1.16 * (2.71828))**(-0.2 * src_distance)) + params_qd[2]
                ml_magnitude = np.log10(w_andersoon_amplitude) + np.log10(src_distance / 100.0) + 0.00301 * (src_distance - 100.0) + 3.0
                ml_magnitudes.append(ml_magnitude)
                
                ## writting the excel file
                ml_write_files_detail.write("{},{},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n".format(int(ID),station,float(peak_to_peak),
                                                                                        float(pick_value_1),float(pick_value_2),float(timespan),
                                                                                        float(w_andersoon_amplitude),(ml_magnitude)))
                
                
                ##### For Plotting Purposes
                axs[count].plot(tr.times(), tr.data, 'k')
                
                pick_value_1index=int((pick_value_1-tr.stats.starttime)*tr.stats.sampling_rate)
                pick_value_1y=tr.data[pick_value_1index]
                pick_value_2index=int((pick_value_2-tr.stats.starttime)*tr.stats.sampling_rate)
                pick_value_2y=tr.data[pick_value_2index]
                
                axs[count].plot((pick_value_1 - tr.stats.starttime),pick_value_1y,'r*',markersize=14)
                axs[count].annotate("Peak 1",((pick_value_1 - tr.stats.starttime),pick_value_1y),textcoords="offset points", xytext=(0,10), ha='center',fontsize=18)
                
                axs[count].plot((pick_value_2 - tr.stats.starttime),pick_value_2y,'r*',markersize=14)
                axs[count].annotate("Peak 2",((pick_value_2 - tr.stats.starttime),pick_value_2y),textcoords="offset points", xytext=(0,10), ha='center',fontsize=18)
                
                axs[count].plot([(pick_value_1 - tr.stats.starttime),(pick_value_2 - tr.stats.starttime)],[pick_value_1y,pick_value_1y],'r--', markersize=16, label='timespan')
                #axs[count].annotate("Timespan",((pick_value_2 - tr.stats.starttime),pick_value_1y),textcoords="offset points",xytext=(50,0), ha='right',fontsize=18)
                
                ## add vertical line as a sign of S pick position
                s_pick_relative_time=S_pick-tr.stats.starttime
                axs[count].axvline(x=s_pick_relative_time, color='b', linestyle='--', label='S Arrival')
                axs[count].legend(fontsize=18)
                
                
                axs[count].set_xlim((pick_value_1 - tr.stats.starttime-0.800),(pick_value_2 - tr.stats.starttime + 0.400) )
                
                axs[count].set_title("{}".format(station), loc="right",va='center', fontsize=18)
                count+=1
            else:
                pass
        
        except Exception as e:
            pass
    # Set common labels for the x-axis and y-axis
    fig.text(0.5, 0.04, 'Relative time (s)', ha='center', fontsize=24)
    fig.text(0.04, 0.5, 'Amplitude (Counts)', va='center', rotation='vertical', fontsize=24)
    fig.suptitle('Peak to Peak and Timespan Event ID_{}'.format(ID), fontsize=30, fontweight='bold')
    #plt.rcParams.update({'font.size': 24})
    plt.savefig(ml_result.joinpath("event_{}.png".format(ID)))
    ml_write_files_detail.write("\n")
    
    ### calculate the ml magnitude as average of all ml magnitude calculated every station
    ml_magnitudes_array=np.array(ml_magnitudes)
    average_ml=np.mean(ml_magnitudes_array)
    ml_write_files_collective.write("{},{:.3f}\n".format(int(ID), average_ml))
    
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
    Input1=Path(r"E:\SEML\DATA TRIMMING\EVENT DATA TRIM\2023\2023 05")            # trimmed .mseed file
    Input2=Path(r"E:\SEML\DATA PROCESSING\MEQ MISCELANEOUS PROCESSING\NonLinLoc Results\Hasil NLL Revised\HASIL NLL 27 06 2023\UPDATE 27 06 2023\2023_05")     # NLL output data .hyp
    Save=Path(r"E:\SEML\DATA ROTATED\2023\2023 05")      # Rotated data output
    ml_result=Path(r"E:\SEML\DATA PROCESSING\MEQ MISCELANEOUS PROCESSING\MW CALCULATIONS\ML 2023\2023_05")
    inventory_path=Path(r"E:\SEML\DATA PROCESSING\MEQ MISCELANEOUS PROCESSING\Program Penting\SEISMOMETER INSTRUMENT CORRECTION\CALIBRATION")
    ID_starts_event=2000
    ###########################################
    listOfStation=['M01','M02','M03','M04','M05','M06','M07','M08','M09','M10','M11','M12','M13','M14','M15'] ## Please pay attention to the 'L' words normally placed in station code
    #### READING THE RAW FILE
    ### Global search for all name file 
    FilesName=glob.glob(os.path.join(Input2, "*.grid0.loc.hyp"), recursive=True)
    wave_names=glob.glob(os.path.join(Input1,"*.mseed"), recursive=True)
    wave_names.sort()
    FilesName.sort()
    with open (ml_result.joinpath("ml_magnitude_2023.csv"), 'w') as ml_output_detail, open (ml_result.joinpath("ml_magnitude_collect_2023_05.csv"), 'w') as ml_output_collective :
        ml_output_detail.write("Event_ID, Station, P2P Amp(Counts), 1st Peak(s), 2nd Peak(s), Timespan(s),WAA (mm),Ml\n")
        ml_output_collective.write("Event_ID, Avg_Ml\n")
        for x in range(len(FilesName)):
            ID_starts_event+=1
            
            ## create dict or list holder 
            back_azim_holder={};# dict to hold the station code and the value of the back azimuth
            
            picking_and_distance=defaultdict(list) ## dict to hold picking time and source distance (for rotation and ml calculation puposes)
            ## with format dict={"Keydict":[picking P, Picing S, Source distance]}
            
            
            #ource_distance={}; ## dict to hold the hypocentral distance from source to the station 
            
            station_holder=[] # a list to hold all stations that record the event
            
            ## for wave naming purpose
            files_name=Path(FilesName[x]).stem # for wav file name purpose and id name
            wave_name="{:8d}_{:04d}_{:02d}".format(int(files_name[5:13]),int(files_name[14:18]),int(files_name[18:20]))
            
            ## Create the file holder 
            with open (FilesName[x], 'r') as NLL_input:
                for line in NLL_input:
                    if line[:3] in listOfStation:
                        vv=line.split()
                        vv[0]=vv[0][0]+'L'+vv[0][1:3]
                        if vv[2]=='BHZ':
                            key_dict="{}_{}".format(vv[0],vv[2]) ## naming format of back azimuth object holder 
                            back_azim_holder[key_dict]=float(vv[23])
                            #source_distance[key_dict]=float(vv[22])
                            picking_and_distance[key_dict]=[0,1,2]
                            ## register the source distance value to the dict
                            picking_and_distance[key_dict][2]=float(vv[22]) 
                            picking_and_distance[key_dict][0]="{}-{:02d}-{:02d}T{:02d}:{:02d}:{:07.5f}". \
                                                              format(int(vv[6][:4]),int(vv[6][4:6]),int(vv[6][6:]),int(vv[7][:2]),int(vv[7][2:]), float(vv[8]))
                            
                            ## station holder
                            station_holder.append(vv[0])
                        elif vv[2] == 'BHN' or vv[2] == 'BHE':
                            picking_and_distance[key_dict][1]="{}-{:02d}-{:02d}T{:02d}:{:02d}:{:07.5f}". \
                                                              format(int(vv[6][:4]),int(vv[6][4:6]),int(vv[6][6:]),int(vv[7][:2]),int(vv[7][2:]), float(vv[8]))
                        else:
                            pass
                    else:
                        pass
                NLL_input.close()
            
            ## start processing the waveform
            try:
                if wave_name == Path(wave_names[x]).stem:
                    st=read(wave_names[x])
                    sst=st.copy()
                    
                    ## calculate the local magnitude
                    ml_magnitude=calculate_ml_magnitude (inventory_path,sst,picking_and_distance, ml_output_detail, ml_output_collective, ID_starts_event)
                    
                    ## remove instrument response
                    #rrt=instrument_remove(sst, inventory_path, ID_starts_event)  ## uncomment to use this line!!!!

                    ## rotate the seismogram
                    #rtt=Rotate(rrt,back_azim_holder,station_holder)
                    
                    #NewSeis=WriteSeis(Save,rtt,wave_name + '.mseed')             ## uncomment to use this line!!!!
                else:
                    pass
            except Exception as e:
                print(e)
                pass
        ml_output_detail.close()
        ml_output_collective.close()
        
        
    print("Rotating procedure has been done succesfully ")
    print('''
     _____  ____  ____   ____ _____ __ __    ___  ___        __  __  __ 
    |     ||    ||    \ |    / ___/|  |  |  /  _]|   \      |  ||  ||  |
    |   __| |  | |  _  | |  (   \_ |  |  | /  [_ |    \     |  ||  ||  |
    |  |_   |  | |  |  | |  |\__  ||  _  ||    _]|  D  |    |__||__||__|
    |   _]  |  | |  |  | |  |/  \ ||  |  ||   [_ |     |     __  __  __ 
    |  |    |  | |  |  | |  |\    ||  |  ||     ||     |    |  ||  ||  |
    |__|   |____||__|__||____|\___||__|__||_____||_____|    |__||__||__|
                                                                        
    ''')
