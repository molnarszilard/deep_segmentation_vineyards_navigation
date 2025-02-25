#!/usr/bin/env python3

# Copyright 2021 Simone Cerrato and Diego Aghi. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import tensorflow as tf

import numpy as np
import cv2

# ROS libraries
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import time


# Global variables
depth_flag = 0
depth_frame = Image()
bridge = CvBridge()
pub = 0
buffer_counter=0
output_buffer =0
#Input network shape
r,c=224,224

#Init variables for control function
init_state=True
previous_command=int(r/2)

'''
#uncomment to use GPU

if not tf.config.list_physical_devices('XLA_GPU'):
	print("No GPU was detected.")

gpus = tf.config.experimental.list_physical_devices('XLA_GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'XLA_GPU')
#tf.config.experimental.set_memory_growth(gpus[1], True)
'''

model_file = 'model_mobile_seg_fp32.tflite'

#Define functions to use the tflite model

interpreter = tf.lite.Interpreter(model_file)
interpreter.allocate_tensors()

def setInput(interpreter, data):
    """Copies data to input tensor."""
    input_tensor(interpreter)[:, :] = data
    
def output_tensor(interpreter):
    """Returns dequantized output tensor."""
    output_details = interpreter.get_output_details()[0]
    output_data = interpreter.tensor(output_details['index'])()
    return output_data
    
def input_tensor(interpreter):
    """Returns input tensor view as numpy array of shape (height, width, 3)."""
    tensor_index = interpreter.get_input_details()[0]['index']
    return interpreter.tensor(tensor_index)()[0]

#Normalize the frame
def normalizeImage(frame):
	frame = (frame/255.)
	return frame

#Filter out the noise
def get_rid_of_noise(frame):
	'''Taking into account the values from the index of maximum to the bottom of the image, 
	the first time the value of the noise function goes below a certain threshold,
	in this case 3%, everything beyond that index it is considered to be noise'''

	#compute the sum of the 1 per rows 
	noise=np.zeros(frame.shape[0])

	noise = np.sum(frame,axis=1)

	#find the index of the max value
	max_ind= np.argmax(noise)  

	#threshold
	sigma_=3/100   
	cut_noise_index=0

	t_counter=1

	while (cut_noise_index ==0):
		sigma_ *=t_counter
		cut_noise_index= np.argmin(noise[max_ind:]>(int(noise[max_ind]*sigma_)))
		t_counter+=1

	return (max_ind+cut_noise_index)

#Get pixel value for the controller functions
def get_x_value(histo):
	global previous_command, init_state

	#Command initialization
	command_x = int(r/2)
	
	ERROR_FLAG=0

	# Look for column indexes where histo==0 
	zero_ind=np.where(histo==0)[0]
		
	cluster_list=[]
	#starting index of the last cluster
	last_index = 0 
	
	for index in range(len(zero_ind)-1):

		#If the difference is greater than 1 then it is a different cluster
		if not (zero_ind[index+1]-zero_ind[index])==1:

			#Filtering from small clusters (noise)
			if len(zero_ind[last_index:index+1])>3:
				cluster_list.append(zero_ind[last_index:index+1].tolist())
			last_index=index+1 #to append the last cluster

	if len(zero_ind[last_index:])>3:
		cluster_list.append(zero_ind[last_index:].tolist())
	
	zero_ind = np.array(cluster_list).flatten()

	#If only one cluster is detected
	if len(cluster_list)==1:
		command_x=int((zero_ind[-1]-zero_ind[0])/2)+zero_ind[0]
		previous_command=command_x
		init_state=False
		print(command_x,"command_x")
	
	else:
		if init_state:
			#Removing clusters in the sides
			for cluster in cluster_list:
				cluster_removal=False
				for ind in cluster:
					if ind > (0.8*r) or ind < (0.2*r):
						cluster_removal=True
						break
				if cluster_removal:
					cluster_list.remove(cluster)

			if len(cluster_list)>0:

				#If there is more than one cluster take the largest
				final_cluster=max(cluster_list, key=len )
				command_x=int((final_cluster[-1]-final_cluster[0])/2)+final_cluster[0]
				previous_command=command_x
				print(command_x,"command_x")
				init_state=False

			else:
				print("ERROR: init state with, no clusters detected")
				ERROR_FLAG=1

		else: 
			#check for previous commands
			if previous_command in zero_ind:
				#p_c_index : index in which the previous cluster is
				for index in range(len(cluster_list)):
					if previous_command in cluster_list[index]:
						p_c_index=index
						break

				#command_x in new cluster

				command_x=int((cluster_list[p_c_index][-1]-cluster_list[p_c_index][0])/2)+cluster_list[p_c_index][0]
				previous_command=command_x
				print(command_x,"command_x")

			else: 
				#if previous command is not in one of the clusters
				#check if it is near 3%

				neig=np.arange( (previous_command- int(0.02*r)),(previous_command+1 + int(0.02*r)))

				#new previous command
				new_p_c=[i for i in neig if i in zero_ind]
				
				if len(new_p_c)==0:
					#retake frame
					print("ERROR: anomaly detected and no previous_command can be found in the current clusters")
					ERROR_FLAG=1

				else:
					#check for previous commands

					for index in range(len(cluster_list)):
						if new_p_c[0] in cluster_list[index]:
							p_c_index=index
							break
					command_x=int((cluster_list[p_c_index][-1]-cluster_list[p_c_index][0])/2)+cluster_list[p_c_index][0]
					previous_command=command_x

	return command_x,ERROR_FLAG

#Controller function
def controller_velocities(delta,w,max_Ang_Vel,max_Lin_Vel):
	#parabolic control 
	if delta > 0:                  #  LEFT SIDE
	  ang_vel_command = float( - max_Ang_Vel*delta*delta)/((w/2)*(w/2))

	else:                       # RIGTH SIDE
	  ang_vel_command = float(max_Ang_Vel*delta*delta)/((w/2)*(w/2))


	lin_vel_command = float(max_Lin_Vel*(1-((delta*delta)/((w/2)*(w/2)))))


	return (lin_vel_command,ang_vel_command)

def depth_callback(data):
	global depth_flag,depth_frame
	if not depth_flag:
		depth_frame = data
	 
def process_depth(frame):
	global bridge

	#Conversion from ROS message to opencv frame format
	frame_depth = bridge.imgmsg_to_cv2(frame)

	#Resizing and numpy array format
	frame_depth_resized = cv2.resize(frame_depth, (r,c), interpolation = cv2.INTER_AREA)
	depth_array = np.asarray(frame_depth_resized)

	#Max value and normalization
	max_value = np.amax(depth_array)
	depth_array_normalized = depth_array/float(max_value)

	#Selection of indexes greater thanh threshold
	perc_threshold = 0.3
	over_threshold_indexes = np.argwhere(depth_array_normalized > perc_threshold)
	
	return over_threshold_indexes


def image_callback(data):
    
	global buffer_counter, output_buffer,bridge,pub,depth_flag,depth_frame,t0
	#Conversion from ROS message to opencv frame format
	frame = bridge.imgmsg_to_cv2(data, "bgr8")

	# Resizing and normalization
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = cv2.resize(frame.astype('uint8'), (r,c), interpolation = cv2.INTER_AREA)
	frame = normalizeImage(frame)
	
	#model input
	setInput(interpreter, frame[None,...])
	# invoke interpreter
	interpreter.invoke()
	y_pred = output_tensor(interpreter)[0]
	y_pred = (y_pred > 0.85)
	output_process=np.copy(y_pred).astype(np.uint8)

	# Remove noise at the bottom
	threshold_noise=get_rid_of_noise(frame=output_process)
	indexes = range(threshold_noise,r)
	output_process[indexes,:]=0
	
	if buffer_counter==0:
		output_buffer=np.copy(output_process)
		buffer_counter+=1

	elif (buffer_counter>=3):
		depth_flag = 1
		indexes_depth = process_depth(depth_frame)
		output_buffer[indexes_depth[:,0],indexes_depth[:,1]] = 0
		depth_flag = 0
		buffer_counter=0
		
		histo=np.zeros(c).astype(np.uint8)  #astype must be kept
		
		histo = np.sum(output_buffer,axis=0,dtype=np.uint8)
		
	else:
		output_buffer=output_process+output_buffer-(output_process*output_buffer)  #union
		buffer_counter+=1
	
	
	if (buffer_counter==0): #it means 3 frames are passed and the controller can be applied
		command_x, ERROR_FLAG = get_x_value(histo=histo)
		message = Twist()
		if not ERROR_FLAG:
			#control functions based on command_x can be done  (or send to other ROS NODE)
			lin_vel,ang_vel=controller_velocities((command_x- int(c/2)),c,1.0,1.0)
			message.linear.x = 0.5*lin_vel
			message.angular.z = ang_vel

		#If ERROR_FLAG=1 it publishes 0 command velocities
		pub.publish(message)

if __name__=="__main__":
	rospy.init_node("ML_algorithm")
	rospy.Subscriber("/camera/color/image_raw",Image,image_callback)
	rospy.Subscriber("/camera/depth/image_rect_raw",Image,depth_callback)
	pub = rospy.Publisher("/cmd_vel",Twist,queue_size = 2)
	rospy.spin()
	


