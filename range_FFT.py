import numpy as np 
import sys
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import os
import cv2
from collections import defaultdict
import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
matplotlib.use('TkAgg') 
from matplotlib.animation import FuncAnimation
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import itertools
import struct

class FrameConfiguration:
    def __init__(self):
        self.numtx=3
        self.numrx=4
        self.chirps=182
        self.range_bins=256
        self.iq=2
        self.frameSize=3*4*182*256
        self.enable_static_removal=False


class RawDataReader:
    def __init__(self, path):
        self.path = path
        self.ADCBinFile = open(path, 'rb')

    def getNextFrame(self, frameconfig):
        timestamp = self.ADCBinFile.read(8)
        timestamp = struct.unpack('d', timestamp)[0]

        frame = np.frombuffer(self.ADCBinFile.read(frameconfig.frameSize * 4), dtype=np.int16)

        r_omega = self.ADCBinFile.read(8)
        r_omega = struct.unpack('d', r_omega)[0]

        l_omega = self.ADCBinFile.read(8)
        l_omega = struct.unpack('d', l_omega)[0]

        angle = self.ADCBinFile.read(8)
        angle = struct.unpack('d', angle)[0]

        return timestamp, frame, r_omega, l_omega, angle

    def close(self):
        self.ADCBinFile.close()

def i_qvalues(frame):#returns the frma in iq value
    np_frame = np.zeros(shape=(len(frame) // 2), dtype=np.complex_)
    np_frame[0::2] = frame[0::4] + 1j * frame[2::4]
    np_frame[1::2] = frame[1::4] + 1j * frame[3::4]
    return np_frame

def reshape_frame(frame):
    #the size of the frame has to be made to 3*4*182*256
     frame=np.reshape(frame,(182,3,4,-1))
     return frame.transpose(1,2,0,3)
     

def range_FFT(reshape_frame,window=-1):
    rangeFFT=None
    if window==-1:#by default rectangular window
        rangeFFT=np.fft.fft(reshape_frame)
        return rangeFFT
    if window=='hamming':
        window = np.hamming(256)
        windowed_frame=reshape_frame*window
        return np.fft.fft(windowed_frame,axis=-1)
    
def dopplerFFT(range_result,frameconfig):
    #rangeresult ka shape hai 3*4*128*256
    #frameConfig.numLoopsPerFrame hai 128
    windowedBins2D = range_result * np.reshape(np.ones(frameconfig.chirps), (1, 1, -1, 1))
    #taking a hamming window for FFT.
    dopplerFFTResult = np.fft.fft(windowedBins2D, axis=2) #array of size 3*4*128*256
    dopplerFFTResult = np.fft.fftshift(dopplerFFTResult, axes=2)#shift the zero frequency to the center of the array
    return dopplerFFTResult

def clutter_removal(input_val, axis=0):  #axis =2 passed from main function
    # Reorder the axes
    #input val is rangeFFT of dimension 3*4*128*256
    reordering = np.arange(len(input_val.shape))
    #input_val.shape ka length 4 hai as there are 4 dimensions
    #so reordering is the array [0,1,2,3]
    reordering[0] = axis
    #reordering=[2,1,2,3]
    reordering[axis] = 0
    #reordering=[2,1,0,3]
    input_val = input_val.transpose(reordering)
    #abhi input value ka shape hai 182*4*3*256

    # Apply static clutter removal
    mean = input_val.mean(0) #caluclate mean across the first axis(across the 128 wala axis)
    #mean ka shape is 4*3*256
    output_val = input_val - mean
    """
    This operation essentially removes the static background or "clutter" from 
    the radar data, leaving behind only the dynamic components (like moving objects). 
    This is a common preprocessing step in radar processing to enhance the detection 
    capability of moving targets.
    """
    #basically this will give a black line in the doppler range heatmap at velocity =zero. This code removes the zero velocity onject
    return output_val.transpose(reordering)

def get_coordinates(dopplerResult):
    #First 30cm make it very negative so the first 3 bins
    cfar_result=np.zeros(dopplerResult.shape,bool)
    top_128=128
    energy_threshold = np.partition(dopplerResult.ravel(), 182 * 256 - top_128 - 1)[182 * 256 - top_128 - 1]
        #So energy Thre128 is the 128th most energetic point
    # print(energy_threshold)
    cfar_result[dopplerResult>energy_threshold]=True
    det_peaks_indices = np.argwhere(cfar_result == True)
    # print(det_peaks_indices.shape)
    object_energy_coordinates=np.zeros((top_128,3))
    object_energy_coordinates[:,0]=det_peaks_indices[:,0]
    object_energy_coordinates[:,1]=det_peaks_indices[:,1]
    for i in range(top_128):
        x_cor=object_energy_coordinates[i][0]
        y_cor=object_energy_coordinates[i][1]
        object_energy_coordinates[i][2]=dopplerResult[int(x_cor)][int(y_cor)]
    
    return object_energy_coordinates,cfar_result
        
def get_azimuthal_angle(dopplerResult,cfar_result):
    az_angle_map={}
    for i in range(cfar_result.shape[0]):
        for j in range(cfar_result.shape[1]):
            if cfar_result[i][j]==True:
                key=(i,j)
                az_angle_map[key]=dopplerResult[:,:,i,j].reshape(12,-1).flatten()[0:8]
    for key,value in az_angle_map.items():
        azimuth_fft_padded=np.zeros(64,dtype=np.complex_)
        azimuth_fft_padded[0:8]=az_angle_map[key]
        azimuth_fft_padded=np.fft.fft(azimuth_fft_padded)
        azimuth_fft_padded = np.fft.fftshift(azimuth_fft_padded)
        az_angle_map[key]=np.abs(azimuth_fft_padded)
    
    #Now we have a dictionary of x,y corrdinates and the corresponding angle FFTs
    
    # print(cfar_result.shape)
    # for i in range(cfar_result.shape[0]):
    #     for j in range(cfar_result.shape[1]):
    #         if cfar_result[i][j]==False:
    #             dopplerResult[:,:,i,j]=0
    
    # dopplerResult = dopplerResult.reshape(12,128,256)[:8,:,:]
    
    # angleFFT = np.fft.ftt(dopplerResult, axis=0)
    
    return az_angle_map


    # input_angle_FFT= dopplerResult[:, :, cfar_result == True]#input_angle_FFT is an array of 3*4*128 We wish to change its to shape (12,128)
    # input_angle_FFT=input_angle_FFT.reshape(12,-1)
    # #Now it is a 12*128 array here each column corresponds to an object
    # azimuth_angle_FFT=input_angle_FFT[0:8,:]#only the first 8 rows are use din azimuthal angle estimation
    # #If we do a 8 point DFT the results wont be good so we do a 64 point FFT by padding with zeros
    # #Obviously this wont increase the resolution but will give us better interpretability
    # azimuth_angle_FFT_padded=np.zeros((64,128),dtype=np.complex_)
    # azimuth_angle_FFT_padded[:8,]=azimuth_angle_FFT
    # azimuth_fft=np.fft.fft(azimuth_angle_FFT_padded,axis=0)


# def get_scores_and_labels(combinations, X):
#     scores = []
#     all_labels_list = []

#     for i, (eps, num_samples) in enumerate(combinations):
#         dbscan_cluster_model = DBSCAN(eps=eps, min_samples=num_samples).fit(X)
#         labels = dbscan_cluster_model.labels_
#         labels_set = set(labels)
#         num_clusters = len(labels_set)
#         if -1 in labels_set:
#             num_clusters -= 1
    
#         if (num_clusters < 2) or (num_clusters > 50):
#             scores.append(-10)
#             all_labels_list.append('bad')
            # c = (eps, num_samples)
    #         print(f"Combination {c} on iteration {i+1} of {N} has {num_clusters} clusters. Moving on")
    #         continue
    
    #     scores.append(ss(X, labels))
    #     all_labels_list.append(labels)
    #     print(f"Index: {i}, Score: {scores[-1]}, Labels: {all_labels_list[-1]}, NumClusters: {num_clusters}")

    # best_index = np.argmax(scores)
    # best_parameters = combinations[best_index]
    # best_labels = all_labels_list[best_index]
    # best_score = scores[best_index]

    # return {'best_epsilon': best_parameters[0],
    #         'best_min_samples': best_parameters[1], 
    #         'best_labels': best_labels,
    #         'best_score': best_score}


#--------------------------------------------------------------------------------------

#Total frame numbers
total_frame_number=0
file_path=sys.argv[1]
total_frame_number=int(sys.argv[2])
frame_to_show = int(sys.argv[3])
count=1

# Uncomment the following lines if you want to store the .png files for animation
# -------------------------------------------------
# range_path='range_angle'
# doppler_path='doppler_angle'
# os.makedirs(range_path, exist_ok=True)
# os.makedirs(doppler_path, exist_ok=True)
# -------------------------------------------------

frameconfig=FrameConfiguration()
bin_reader = RawDataReader(file_path)
for frame_no in range(total_frame_number):
    timestamp, np_frame, l_omega, r_omega, angle = bin_reader.getNextFrame(frameconfig)
    # print(timestamp, l_omega, r_omega, angle)

    np_frame=i_qvalues(np_frame)
    reshaped_np_frame=reshape_frame(np_frame)
    range_result=range_FFT(reshaped_np_frame)
    # range_result=clutter_removal(range_result,axis=2)
    dopplerResult=dopplerFFT(range_result,frameconfig)
    dopplerResultabs=np.absolute(dopplerResult)
    dopplerResultabs=np.sum(dopplerResultabs,axis=(0,1))
    
    # Uncomment the following two lines to show the doppler result for each frame
    # sns.heatmap(dopplerResultabs)
    # plt.show()

    energy_coordinates,cfar_result=get_coordinates(dopplerResultabs)
    energy_coordinates=energy_coordinates[energy_coordinates[:,2].argsort()[::-1]]
        
    #     # groups = defaultdict(list)
    #     # for x,y,energy in energy_coordinates:
    #     #     groups[y].append(((x,y),energy))
    #     # result = []
    #     # for y, items in groups.items():
    #     #     max_energy = max(item[1] for item in items)
    #     #     layer=[]
    #     #     for (x,y), energy in items:
    #     #        if energy == max_energy:
    #     #            layer.append([x,y,energy])
    #     #     result.append(layer)
        
    #     print(energy_coordinates)

    #az_angle_map is a dict keyed by (range, doppler) coordinates of cfar points (top 128 energetic points) 
    #and values as corresponding 64-point angle FFT
    az_angle_map=get_azimuthal_angle(dopplerResult,cfar_result)
    
    # print(len(az_angle_map))  #Length is 128 : Top 128 points are considered in the CFAR 
    # print(az_angle_map)
        
    #Now we have the coordinates of the objects in a single frame and their x and y coordinates where x coordinate
    #refers to the velocity and y is the range(range-doppler heatmap)
        # for i in range(cfar_result.shape[0]):
        #     for j in range(cfar_result.shape[1]):
        #         cfar_result[i][j]=False
        # for ele in result:
        #     cfar_result[int(ele[0][0])][int(ele[0][1])]=True
        # az_angle_map=get_azimuthal_angle(dopplerResult,cfar_result)
        # print(az_angle_map)
        # range_angle_dict={}
        # for key,value in az_angle_map.items():
        #     if key[1] not in range_angle_dict.keys():
        #         range_angle_dict[key[1]]=[]
        #         range_angle_dict[key[1]].append(value)
        #         range_angle_dict[key[1]].append(dopplerResultabs[key[0]][key[1]])
        #     else:
        #         range_angle_dict[key[1]].append(value)
        #         range_angle_dict[key[1]].append(dopplerResultabs[key[0]][key[1]])
        
        # for key,value in range_angle_dict.items():
        #     print(key,value)

    # Generate range_angle_heatmap for a specific frame number

    count+=1

    if (count == frame_to_show):           #count == frame_number
        range_angle=np.zeros((256,64),dtype=np.complex_)
        for key,value in az_angle_map.items():      #key = (vel, range)
            range_angle[key[1]]+=np.abs(value)           #unique range
        
        sns.heatmap(np.abs(range_angle))
        plt.show()

        # print(range_angle)

        # range_angle_abs = np.abs(range_angle)

        # #Following code generates the mask
        # max_val = np.max(range_angle_abs)
        # range_angle_abs /= max_val
        # mask = range_angle_abs > 0.5

        #Uncomment the following code for object detection using OpenCV contour detection
        # ------------------------------------------------------------------------------------------
        # original = np.copy(range_angle_abs)

        # contours, _ = cv2.findContours(mask.astype(np.uint8), 
        #                        cv2.RETR_EXTERNAL, 
        #                        cv2.CHAIN_APPROX_SIMPLE)

        # print(contours)

        # centers = []

        # for cnt in contours:
        #     center = np.average(cnt, axis=0)
        #     centers.append(center)
        #     radius = np.max(np.linalg.norm((cnt - center)[:, 0], axis=1))
        #     radius = max(radius, 10.0)
        #     cv2.circle(original, center[0].astype(np.int32), int(radius), (0, 0, 255), 2)
        #     cv2.circle(mask, center, 5, (0, 0, 255), -1)

        # cv2.imshow('Mask with centers', mask) #For this to work, need to convert mask to an image

        # print(centers)  #Gives list of r, theta for objects
        # sns.heatmap(mask)
        # plt.show()

        # -----------------------------------------------------------------------------------------------

        """
        KMEans clustering
        1. Store in a df
        2. Normalize range and angle features using MinMaxScaler
        3. fit and predict using model
        4. Include clusters in the df
        """

        # -----------------------------------------------------------------------------------------------------------
        # #Initialize empty dataframe
        # range_angle_df = pd.DataFrame(columns = ['Range', 'Angle'])

        # #Iterate over points in range angle mask and store True in dataframe
        # for i in range(256):
        #     for j in range(64):
        #         if(mask[i][j]):
        #             #Append to df
        #             new_entry = {'Range': i, 'Angle': j}
        #             range_angle_df = pd.concat([range_angle_df, pd.DataFrame([new_entry])], ignore_index=True)

        # #Scatter plot before clustering
        # # plt.scatter(range_angle_df.Angle, range_angle_df.Range)
        # # plt.xlabel('Angle')
        # # plt.ylabel('Range')
        # # plt.show()

        # #Scaling features
        # scaler = MinMaxScaler()
        # scaler.fit(range_angle_df[['Angle']])
        # range_angle_df['Angle'] = scaler.transform(range_angle_df[['Angle']])
        # scaler.fit(range_angle_df[['Range']])
        # range_angle_df['Range'] = scaler.transform(range_angle_df[['Range']])

        # # plt.scatter(range_angle_df.Angle, range_angle_df.Range)
        # # plt.xlabel('Angle')
        # # plt.ylabel('Range')
        # # plt.show()

        # #Find best value of n_clusters using elbow plot
        # sse = []
        # k_rng = range(1,10)
        # for k in k_rng:
        #     km = KMeans(n_clusters=k)
        #     km.fit(range_angle_df[['Angle','Range']])
        #     sse.append(km.inertia_)

        # #Find the elbow point 
        # bf_ratio=np.zeros( (len(sse),) )
        # for i in range(len(sse)):
        #     if i==0 and i==len(sse)-1:
        #         continue
        #     bf_ratio[i] = (sse[i-1] - sse[i])/(sse[i] - sse[i+1])

        # n_clusters_elbow = np.argmax(bf_ratio)+1

        # #Uncomment the following lines of code to plot the elbow point
        # # plt.xlabel('K')
        # # plt.ylabel('Sum of squared error')
        # # plt.plot(k_rng,sse)
        # # plt.show()

        # #Fit model and predict clusters
        # km = KMeans(n_clusters=n_clusters_elbow)
        # y_predicted = km.fit_predict(range_angle_df[['Range','Angle']])
        # y_predicted

        # range_angle_df['cluster']=y_predicted

        # centers = km.cluster_centers_

        #Uncomment the following lines of code to plot the clusters (currently 3 clusters)
        # df1 = range_angle_df[range_angle_df.cluster==0]
        # df2 = range_angle_df[range_angle_df.cluster==1]
        # df3 = range_angle_df[range_angle_df.cluster==2]
        # plt.scatter(df1['Angle'],df1['Range'],color='green')
        # plt.scatter(df2['Angle'],df2['Range'],color='red')
        # plt.scatter(df3['Angle'],df3['Range'],color='black')
        # plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
        # plt.legend()
        # plt.show()# #Scatter plot before clustering
        # # plt.scatter(range_angle_df.Angle, range_angle_df.Range)
        # # plt.xlabel('Angle')
        # # plt.ylabel('Range')
        # # plt.show()

        # #Scaling features
        # scaler = MinMaxScaler()
        # scaler.fit(range_angle_df[['Angle']])
        # range_angle_df['Angle'] = scaler.transform(range_angle_df[['Angle']])
        # scaler.fit(range_angle_df[['Range']])
        # range_angle_df['Range'] = scaler.transform(range_angle_df[['Range']])

        # # plt.scatter(range_angle_df.Angle, range_angle_df.Range)
        # # plt.xlabel('Angle')
        # # plt.ylabel('Range')
        # # plt.show()

        # #Find best value of n_clusters using elbow plot
        # sse = []
        # k_rng = range(1,10)
        # for k in k_rng:
        #     km = KMeans(n_clusters=k)
        #     km.fit(range_angle_df[['Angle','Range']])
        #     sse.append(km.inertia_)

        # #Find the elbow point 
        # bf_ratio=np.zeros( (len(sse),) )
        # for i in range(len(sse)):
        #     if i==0 and i==len(sse)-1:
        #         continue
        #     bf_ratio[i] = (sse[i-1] - sse[i])/(sse[i] - sse[i+1])

        # n_clusters_elbow = np.argmax(bf_ratio)+1

        # #Uncomment the following lines of code to plot the elbow point
        # # plt.xlabel('K')
        # # plt.ylabel('Sum of squared error')
        # # plt.plot(k_rng,sse)
        # # plt.show()

        # #Fit model and predict clusters
        # km = KMeans(n_clusters=n_clusters_elbow)
        # y_predicted = km.fit_predict(range_angle_df[['Range','Angle']])
        # y_predicted

        # range_angle_df['cluster']=y_predicted

        # centers = km.cluster_centers_

        #Uncomment the following lines of code to plot the clusters (currently 3 clusters)
        # df1 = range_angle_df[range_angle_df.cluster==0]
        # df2 = range_angle_df[range_angle_df.cluster==1]
        # df3 = range_angle_df[range_angle_df.cluster==2]
        # plt.scatter(df1['Angle'],df1['Range'],color='green')
        # plt.scatter(df2['Angle'],df2['Range'],color='red')
        # plt.scatter(df3['Angle'],df3['Range'],color='black')
        # plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
        # plt.legend()
        # plt.show()

        # -----------------------------------------------------------------------------------------------------------

        """
        DBSCAN
            1. Store in a df
            2. Normalize range and angle features using MinMaxScaler
            3. Perform grid search to find best epsilon and min_samples
            4. Fit and train model
            5. Include model in df 
        """
        # -----------------------------------------------------------------------------------------------------------
        
        # # #Initialize empty dataframe
        # range_angle_df = pd.DataFrame(columns = ['Range', 'Angle'])

        # #Iterate over points in range angle mask and store True in dataframe
        # for i in range(256):
        #     for j in range(64):
        #         if(mask[i][j]):
        #             #Append to df
        #             new_entry = {'Range': i, 'Angle': j}
        #             range_angle_df = pd.concat([range_angle_df, pd.DataFrame([new_entry])], ignore_index=True)


        # X = range_angle_df[['Angle', 'Range']].to_numpy()
        # range_list, angle_list = range_angle_df.Range, range_angle_df.Angle

        # plt.scatter(angle_list, range_list)
        # plt.show()

        # #Try tweaking the following list of parameters for getting a good clustering
        # epsilons = np.linspace(0.01, 1, num=15)
        # min_samples = np.arange(2, 20, step=3)

        # combinations = list(itertools.product(epsilons, min_samples))

        # best_dict = get_scores_and_labels(combinations, X)
        # range_angle_df['cluster'] = best_dict.best_labels



    
    # Uncomment following code to save the range_angle plots as .png files for animation

    # file_path_save=os.path.join(range_path,f'range_angle_{count}.png')
    # plt.savefig(file_path_save)
    # plt.close()
        
        
    #Uncomment the following code if DOPPLER ANGLE Plot needs to be generated 

    # doppler_angle=np.zeros((128,64),dtype=np.complex_)
    # for key,value in az_angle_map.items():
    #     if key[0]==64:
    #             continue
    #     doppler_angle[key[0]]+=value #superimpose
             
    # sns.heatmap(np.abs(doppler_angle))

    #Uncomment the following code to save DOPPLER ANGLE plots as .png files for animation

    # file_path_save=os.path.join(doppler_path,f'doppler_angle_{count}.png')
    # plt.savefig(file_path_save)
    # plt.close()
        
    


    




     

    
     

    
    



