import numpy as np
import math
import os
import tensorflow as tf
import glob
import random
from compact_bilinear_pooling import compact_bilinear_pooling_layer
    
print(tf.__version__)


#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
 

raw_points_init = tf.placeholder(tf.float32, shape=[ None, 3], name="raw_points")

centered_points = tf.subtract(raw_points_init, tf.reduce_mean(raw_points_init, axis = 0, keepdims = True))

centered_points_expanded = tf.expand_dims(centered_points, 0,
		                                     name="cn_caps1_output_expanded")

adjoint_mat = tf.matmul(tf.transpose(centered_points_expanded, [0,2,1]), centered_points_expanded)

e,ev = tf.self_adjoint_eig(adjoint_mat, name="eigendata")



normal_vec = ev[:,:,0]
normalized_normal_vec = tf.nn.l2_normalize(normal_vec, axis = 1)


rot_theta = tf.acos(tf.matmul(normalized_normal_vec, tf.transpose(tf.constant([[0.0,0.0,1.0]]),[1,0])))

b_vec = tf.nn.l2_normalize(tf.cross(tf.constant([[0.0,0.0,1.0]]), normalized_normal_vec), axis = 1)

q0 = tf.cos(rot_theta/2.0)
q1 = tf.sin(rot_theta/2.0) * b_vec[0,0]
q2 = tf.sin(rot_theta/2.0) * b_vec[0,1]
q3 = tf.sin(rot_theta/2.0) * b_vec[0,2]

el_0_0 = tf.square(q0) + tf.square(q1) - tf.square(q2) - tf.square(q3)
el_0_1 = 2*(q1*q2-q0*q3)
el_0_2 = 2*(q1*q3+q0*q2)
el_1_0 = 2*(q1*q2+q0*q3)
el_1_1 = tf.square(q0) - tf.square(q1) + tf.square(q2) - tf.square(q3)
el_1_2 = 2*(q2*q3+q0*q1)
el_2_0 = 2*(q1*q3-q0*q2)
el_2_1 = 2*(q2*q3+q0*q1)
el_2_2 = tf.square(q0) - tf.square(q1) - tf.square(q2) + tf.square(q3)

Q = tf.concat([tf.concat([el_0_0,el_0_1,el_0_2], axis = 1), tf.concat([el_1_0,el_1_1,el_1_2], axis = 1), tf.concat([el_2_0,el_2_1,el_2_2], axis = 1)], axis=0)

u_ = tf.matmul(Q,tf.transpose(tf.constant([[1.0,0.0,0.0]]), [1,0]))
v_ = tf.matmul(Q,tf.transpose(tf.constant([[0.0,1.0,0.0]]), [1,0]))
w_ = tf.matmul(Q,tf.transpose(tf.constant([[0.0,0.0,1.0]]), [1,0]))

transform_mat = tf.concat([u_,v_,w_], axis = 1)
    
#transformed_coordinates_ = tf.matmul(centered_points,transform_mat)  
transformed_coordinates_ = centered_points

transformed_coordinates__  = tf.subtract(transformed_coordinates_, tf.stack([tf.reduce_mean(transformed_coordinates_[:,0], keep_dims = True), tf.reduce_mean(transformed_coordinates_[:,1], keep_dims = True), tf.constant([0.0]) ], axis = 1))


a = tf.matmul(tf.transpose(tf.slice(transformed_coordinates__,[0,0], [-1,1]), [1,0]),tf.slice(transformed_coordinates__, [0,1],[-1,1]  )) / tf.matmul(tf.transpose(tf.slice(transformed_coordinates__, [0,0], [-1,1]), [1,0]),tf.slice(transformed_coordinates__, [0,0], [-1,1]))

angle = tf.atan2(a,1)

print("adasdas")
print(angle)


rot_z = tf.concat([[[tf.cos(angle[0,0]), tf.sin(angle[0,0]), 0.0]], [[-tf.sin(angle[0,0]), tf.cos(angle[0,0]), 0.0]], [[0.0, 0.0, 1.0]]], axis = 0)


#transformed_coordinates___ = tf.matmul(transformed_coordinates__, rot_z)
transformed_coordinates___ = centered_points
b = tf.add(tf.slice(transformed_coordinates___, [0, 0], [-1, 1]) * 100000, tf.slice(transformed_coordinates___, [0, 1], [-1, 1]))

reordered = tf.gather(transformed_coordinates___, tf.nn.top_k(b[:, 0], k=tf.shape(transformed_coordinates___)[0], sorted=True).indices)
transformed_coordinates = tf.reverse(reordered, axis=[0])
#transformed_coordinates = centered_points

print(rot_z)

#transformed_coordinates = transformed_coordinates_ 





mask = tf.greater(transformed_coordinates[:,2],0)

points_from_side_one = tf.boolean_mask(transformed_coordinates, mask) 

mask2 = tf.less(transformed_coordinates[:,2],0)

points_from_side_two = tf.boolean_mask(transformed_coordinates, mask2) 


indices_one_x = tf.nn.top_k(points_from_side_one[:,0], k=tf.shape(points_from_side_one)[0]).indices
reordered_points_one_x = tf.gather(points_from_side_one, indices_one_x, axis=0)

indices_two_x = tf.nn.top_k(points_from_side_two[:, 0], k=tf.shape(points_from_side_two)[0]).indices
reordered_points_two_x = tf.gather(points_from_side_two, indices_two_x, axis=0)


indices_one_y = tf.nn.top_k(points_from_side_one[:,1], k=tf.shape(points_from_side_one)[0]).indices
reordered_points_one_y = tf.gather(points_from_side_one, indices_one_y, axis=0)

indices_two_y = tf.nn.top_k(points_from_side_two[:, 1], k=tf.shape(points_from_side_two)[0]).indices
reordered_points_two_y = tf.gather(points_from_side_two, indices_two_y, axis=0)


#b = tf.add(tf.slice(transformed_coordinates, [0, 0], [-1, 1]) * 100000, tf.slice(transformed_coordinates, [0, 1], [-1, 1]))

#reordered = tf.gather(transformed_coordinates, tf.nn.top_k(b[:, 0], k=tf.shape(transformed_coordinates)[0], sorted=True).indices)
#reordered_s = tf.reverse(reordered, axis=[0])


input1_1_x = tf.expand_dims([reordered_points_one_x[:,2]],2)


filter1_1_x = tf.get_variable("a_1", [6, 1, 10], initializer=tf.random_normal_initializer(seed=0.1))#, trainable=False)

output1_1_x = tf.nn.conv1d(input1_1_x, filter1_1_x, stride=2, padding="SAME")

filter2_1_x = tf.get_variable("a_2", [3, 10, 20], initializer=tf.random_normal_initializer(seed=0.1))#,trainable=False)

output2_1_x_temp = tf.nn.conv1d(output1_1_x, filter2_1_x, stride=2, padding="SAME")

output2_1_x = tf.cond(tf.shape(output2_1_x_temp)[1] >= 100, lambda: tf.slice(output2_1_x_temp, [0,0,0], [-1,100,-1]), lambda: tf.concat([output2_1_x_temp, tf.zeros([1,100-tf.shape(output2_1_x_temp)[1],20])], axis = 1))


input1_2_x = tf.expand_dims([reordered_points_two_x[:,2]],2)


filter1_2_x = tf.get_variable("a_3", [6, 1, 10], initializer=tf.random_normal_initializer(seed=0.1))#,trainable=False)

output1_2_x = tf.nn.conv1d(input1_2_x, filter1_2_x, stride=2, padding="SAME")

filter2_2_x = tf.get_variable("a_4", [3, 10, 20], initializer=tf.random_normal_initializer(seed=0.1))#,trainable=False)

output2_2_x_temp= tf.nn.conv1d(output1_2_x, filter2_2_x, stride=2, padding="SAME")

output2_2_x = tf.cond(tf.shape(output2_2_x_temp)[1] >= 100, lambda: tf.slice(output2_2_x_temp, [0,0,0], [-1,100,-1]), lambda: tf.concat([output2_2_x_temp, tf.zeros([1,100-tf.shape(output2_2_x_temp)[1],20])], axis = 1))



input1_1_y = tf.expand_dims([reordered_points_one_y[:,2]],2)


filter1_1_y = tf.get_variable("a_5", [6, 1, 10], initializer=tf.random_normal_initializer(seed=0.1))#,trainable=False)

output1_1_y = tf.nn.conv1d(input1_1_y, filter1_1_y, stride=2, padding="SAME")

filter2_1_y = tf.get_variable("a_6", [3, 10, 20],initializer=tf.random_normal_initializer(seed=0.1))#,trainable=False)

output2_1_y_temp = tf.nn.conv1d(output1_1_y, filter2_1_y, stride=2, padding="SAME")

output2_1_y = tf.cond(tf.shape(output2_1_y_temp)[1] >= 100, lambda: tf.slice(output2_1_y_temp, [0,0,0], [-1,100,-1]), lambda: tf.concat([output2_1_y_temp, tf.zeros([1,100-tf.shape(output2_1_y_temp)[1],20])], axis = 1))




input1_2_y = tf.expand_dims([reordered_points_two_y[:,2]],2)


filter1_2_y = tf.get_variable("a_7", [6, 1, 10], initializer=tf.random_normal_initializer(seed=0.1))#,trainable=False)

output1_2_y = tf.nn.conv1d(input1_2_y, filter1_2_y, stride=2, padding="SAME")

filter2_2_y = tf.get_variable("a_8", [3, 10, 20], initializer=tf.random_normal_initializer(seed=0.1))#,trainable=False)

output2_2_y_temp = tf.nn.conv1d(output1_2_y, filter2_2_y, stride=2, padding="SAME")

output2_2_y = tf.cond(tf.shape(output2_2_y_temp)[1] >= 100, lambda: tf.slice(output2_2_y_temp, [0,0,0], [-1,100,-1]), lambda: tf.concat([output2_2_y_temp, tf.zeros([1,100-tf.shape(output2_2_y_temp)[1],20])], axis = 1))


#side_1_descriptor = tf.matmul(tf.transpose(output2_1_x, [0,2,1]), output2_1_y)
#side_2_descriptor = tf.matmul(tf.transpose(output2_2_x, [0,2,1]), output2_2_y)

#print(side_1_descriptor)

# concat_layer = tf.reshape(tf.concat([output2_1_x, output2_2_x, output2_1_y, output2_2_y], axis = 1), [1, 1600])
concat_layer = tf.reshape(tf.concat([output2_1_x ,output2_2_x,output2_1_y,output2_2_y ], axis = 0), [1, 8000])


rot_angles_temp_ = tf.layers.dense(concat_layer,3, trainable=True, name = "a_9")
rot_angles = tf.constant([[math.pi/4.0, -math.pi/4.0, 0.0]])

#rot_angles = tf.nn.l2_normalize(rot_angles_temp_) 

#rot_angles_ = tf.reshape(rot_angles, [3,3,3])

# rotation_matrix_one = tf.squeeze(tf.slice(rot_angles_, [0,0,0], [1,-1,-1]),squeeze_dims=[0])
# rotation_matrix_two =  tf.squeeze(tf.slice(rot_angles_, [1,0,0], [1,-1,-1]),squeeze_dims=[0])
# rotation_matrix_three =  tf.squeeze(tf.slice(rot_angles_, [2,0,0], [1,-1,-1]),squeeze_dims=[0])

#rot_angles = tf.constant([[22.0/28.0,22.0/14.0,0.0]]) 

rotation_matrix_one = tf.concat([tf.constant([[1.0, 0.0, 0.0]]), [[0.0, tf.cos(rot_angles[0,0]), -tf.sin(rot_angles[0,0])]], [[0.0, tf.sin(rot_angles[0,0]), tf.cos(rot_angles[0,0])]]], axis = 0)
rotation_matrix_two = tf.concat([[[tf.cos(rot_angles[0,1]), 0.0, tf.sin(rot_angles[0,1])]], [[0.0, 1.0, 0.0]], [[-tf.sin(rot_angles[0,1]), 0.0,tf.cos(rot_angles[0,1]) ]]], axis = 0)
rotation_matrix_three = tf.concat([[[tf.cos(rot_angles[0,2]), -tf.sin(rot_angles[0,2]),0.0 ]], [[tf.sin(rot_angles[0,2]), tf.cos(rot_angles[0,2]), 0.0]], [[0.0, 0.0,1.0 ]]], axis = 0)

print(rotation_matrix_one)

centered_points_expanded_ = tf.reshape(transformed_coordinates, [-1, 3])
point_count = tf.shape(centered_points_expanded_)[0]

#rotation_matrix_one = tf.placeholder(tf.float32, shape=[3, 3], name="rot_mat_one")
trasformed_points_one = tf.matmul(centered_points_expanded_, rotation_matrix_one, name="trans_point_one")
trasformed_points_one_reshaped = tf.reshape(trasformed_points_one, [-1, point_count, 3], name = "trans_point_one_reshape")

#rotation_matrix_two = tf.placeholder(tf.float32, shape=[3, 3], name="rot_mat_two")
trasformed_points_two = tf.matmul(centered_points_expanded_, rotation_matrix_two, name="trans_point_two")
trasformed_points_two_reshaped = tf.reshape(trasformed_points_two, [-1, point_count, 3], name = "trans_point_two_reshape")

#rotation_matrix_three = tf.placeholder(tf.float32, shape=[3, 3], name="rot_mat_three")
trasformed_points_three = tf.matmul(centered_points_expanded_, rotation_matrix_three, name="trans_point_three")
trasformed_points_three_reshaped = tf.reshape(trasformed_points_three, [-1, point_count, 3], name = "trans_point_three_reshape")

########################################################################################

trasformed_points_one_reshaped_ = tf.reshape(trasformed_points_one_reshaped, [-1, 3])

#point_distance_one = tf.reduce_sum(tf.square(trasformed_points_one_reshaped_), axis=1, keepdims = True)

#point_distance_one = tf.reduce_sum(trasformed_points_one_reshaped_, axis=1, keepdims = True)

#scale_metric_one = tf.exp(-point_distance_one*0.0000001)

#scale_metric_one = tf.multiply(point_distance_one,0.01)

#scale_metric_tiled_one = tf.tile(scale_metric_one, [1, 3], name="cn_W_tiled")

calibrated_points_one = trasformed_points_one_reshaped_ 








trasformed_points_two_reshaped_ = tf.reshape(trasformed_points_two_reshaped, [-1, 3])

#point_distance_two = tf.reduce_sum(tf.square(trasformed_points_two_reshaped_), axis=1, keepdims = True)

#point_distance_one = tf.reduce_sum(trasformed_points_one_reshaped_, axis=1, keepdims = True)

#scale_metric_two = tf.exp(-point_distance_two*0.0000001)

#scale_metric_one = tf.multiply(point_distance_one,0.01)

#scale_metric_tiled_two = tf.tile(scale_metric_two, [1, 3], name="cn_W_tiled")

calibrated_points_two = trasformed_points_two_reshaped_


trasformed_points_three_reshaped_ = tf.reshape(trasformed_points_three_reshaped, [-1, 3])

#point_distance_three = tf.reduce_sum(tf.square(trasformed_points_three_reshaped_), axis=1, keepdims = True)

#point_distance_one = tf.reduce_sum(trasformed_points_one_reshaped_, axis=1, keepdims = True)

#scale_metric_three = tf.exp(-point_distance_three*0.0000001)

#scale_metric_one = tf.multiply(point_distance_one,0.01)

#scale_metric_tiled_three = tf.tile(scale_metric_three, [1, 3], name="cn_W_tiled")

calibrated_points_three = trasformed_points_three_reshaped_

calibrated_points_one_corrected_shape = tf.reshape(calibrated_points_one, [-1, point_count, 3])


centered_calib_points_one_temp_  = tf.subtract(calibrated_points_one,tf.reduce_mean(calibrated_points_one,axis=0,keep_dims=True))
centered_calib_points_two_temp_ = tf.subtract(calibrated_points_two,tf.reduce_mean(calibrated_points_two,axis=0,keep_dims=True))
centered_calib_points_three_temp_  = tf.subtract(calibrated_points_three,tf.reduce_mean(calibrated_points_three,axis=0,keep_dims=True))



b1 = tf.add(tf.slice(centered_calib_points_one_temp_ , [0, 0], [-1, 1]) * 100000, tf.slice(centered_calib_points_one_temp_ , [0, 1], [-1, 1]))

reordered1 = tf.gather(centered_calib_points_one_temp_ , tf.nn.top_k(b1[:, 0], k=tf.shape(centered_calib_points_one_temp_ )[0], sorted=True).indices)
centered_calib_points_one_temp = tf.reverse(reordered1, axis=[0])

b2 = tf.add(tf.slice(centered_calib_points_two_temp_ , [0, 0], [-1, 1]) * 100000, tf.slice(centered_calib_points_two_temp_ , [0, 1], [-1, 1]))

reordered2 = tf.gather(centered_calib_points_two_temp_ , tf.nn.top_k(b2[:, 0], k=tf.shape(centered_calib_points_two_temp_ )[0], sorted=True).indices)
centered_calib_points_two_temp = tf.reverse(reordered2, axis=[0])


b3 = tf.add(tf.slice(centered_calib_points_three_temp_ , [0, 0], [-1, 1]) * 100000, tf.slice(centered_calib_points_three_temp_ , [0, 1], [-1, 1]))

reordered3 = tf.gather(centered_calib_points_three_temp_ , tf.nn.top_k(b3[:, 0], k=tf.shape(centered_calib_points_three_temp_ )[0], sorted=True).indices)
centered_calib_points_three_temp = tf.reverse(reordered3, axis=[0])







#mask_one  = tf.greater(centered_calib_points_one_temp[:,2],0)
indices_one_x_temp = tf.nn.top_k(centered_calib_points_one_temp[:,2], k=tf.shape(centered_calib_points_one_temp)[0]).indices
reordered_points_one_x_temp = tf.gather(centered_calib_points_one_temp, indices_one_x_temp, axis=0)

index_ = tf.shape(reordered_points_one_x_temp)[0]
index = index_ / 2

centered_calib_points_one_t  = tf.slice(reordered_points_one_x_temp, [0, 0], [index, -1])


indices_two_x_temp = tf.nn.top_k(centered_calib_points_two_temp[:,2], k=tf.shape(centered_calib_points_two_temp)[0]).indices
reordered_points_two_x_temp = tf.gather(centered_calib_points_two_temp, indices_two_x_temp, axis=0)


centered_calib_points_two_t  = tf.slice(reordered_points_two_x_temp, [0, 0], [tf.shape(reordered_points_two_x_temp)[0]/2, -1])


indices_three_x_temp = tf.nn.top_k(centered_calib_points_three_temp[:,2], k=tf.shape(centered_calib_points_three_temp)[0]).indices
reordered_points_three_x_temp = tf.gather(centered_calib_points_three_temp, indices_three_x_temp, axis=0)


centered_calib_points_three_t  = tf.slice(reordered_points_three_x_temp, [0, 0], [tf.shape(reordered_points_three_x_temp)[0]/2, -1])





#indices_one_x_tempi = tf.nn.top_k(centered_calib_points_one_temp[:,2], k=tf.shape(centered_calib_points_one_temp[0]).indices
#reordered_points_one_x_tempi = tf.gather(points_from_side_one_temp, indices_one_x_temp, axis=0)


centered_calib_points_one_t_i  = tf.slice(reordered_points_one_x_temp, [tf.shape(reordered_points_one_x_temp)[0]/2 , 0], [tf.shape(reordered_points_one_x_temp)[0]/2, -1])


centered_calib_points_two_t_i  = tf.slice(reordered_points_two_x_temp, [tf.shape(reordered_points_two_x_temp)[0]/2, 0], [tf.shape(reordered_points_two_x_temp)[0]/2, -1])

centered_calib_points_three_t_i  = tf.slice(reordered_points_three_x_temp, [tf.shape(reordered_points_three_x_temp)[0]/2, 0], [tf.shape(reordered_points_three_x_temp)[0]/2, -1])



####################################################################################################3

indices_one_xx_temp = tf.nn.top_k(centered_calib_points_one_temp[:,0], k=tf.shape(centered_calib_points_one_temp)[0]).indices
reordered_points_one_xx_temp = tf.gather(centered_calib_points_one_temp, indices_one_xx_temp, axis=0)

#mask_onex  = tf.greater(centered_calib_points_one_temp[:,0],0)

#centered_calib_points_one_tx  = tf.boolean_mask(centered_calib_points_one_temp, mask_onex)

centered_calib_points_one_tx  = tf.slice(reordered_points_one_xx_temp, [0, 0], [tf.shape(reordered_points_one_xx_temp)[0]/2, -1])


#mask_twox  = tf.greater(centered_calib_points_two_temp[:,0],0)

indices_two_xx_temp = tf.nn.top_k(centered_calib_points_two_temp[:,0], k=tf.shape(centered_calib_points_two_temp)[0]).indices
reordered_points_two_xx_temp = tf.gather(centered_calib_points_two_temp, indices_two_xx_temp, axis=0)

centered_calib_points_two_tx  = tf.slice(reordered_points_two_xx_temp, [0, 0], [tf.shape(reordered_points_two_xx_temp)[0]/2, -1])

#centered_calib_points_two_tx  = tf.boolean_mask(centered_calib_points_two_temp, mask_twox)

#mask_threex  = tf.greater(centered_calib_points_three_temp[:,0],0)

indices_three_xx_temp = tf.nn.top_k(centered_calib_points_three_temp[:,0], k=tf.shape(centered_calib_points_three_temp)[0]).indices
reordered_points_three_xx_temp = tf.gather(centered_calib_points_three_temp, indices_three_xx_temp, axis=0)

#centered_calib_points_three_tx  = tf.boolean_mask(centered_calib_points_three_temp, mask_threex)
centered_calib_points_three_tx  = tf.slice(reordered_points_three_xx_temp, [0, 0], [tf.shape(reordered_points_three_xx_temp)[0]/2, -1])





centered_calib_points_one_txi  = tf.slice(reordered_points_one_xx_temp, [tf.shape(reordered_points_one_xx_temp)[0]/2 , 0], [tf.shape(reordered_points_one_xx_temp)[0]/2, -1])


centered_calib_points_two_txi  = tf.slice(reordered_points_two_xx_temp, [tf.shape(reordered_points_two_xx_temp)[0]/2, 0], [tf.shape(reordered_points_two_xx_temp)[0]/2, -1])

centered_calib_points_three_txi  = tf.slice(reordered_points_three_xx_temp, [tf.shape(reordered_points_three_xx_temp)[0]/2, 0], [tf.shape(reordered_points_three_xx_temp)[0]/2, -1])

#############################################################################################################################3333


indices_one_y_temp = tf.nn.top_k(centered_calib_points_one_temp[:,1], k=tf.shape(centered_calib_points_one_temp)[0]).indices
reordered_points_one_y_temp = tf.gather(centered_calib_points_one_temp, indices_one_y_temp, axis=0)

centered_calib_points_one_ty  = tf.slice(reordered_points_one_y_temp, [0, 0], [tf.shape(reordered_points_one_y_temp)[0]/2, -1])


#mask_oney  = tf.greater(centered_calib_points_one_temp[:,1],0)

#centered_calib_points_one_ty  = tf.boolean_mask(centered_calib_points_one_temp, mask_oney)
indices_two_y_temp = tf.nn.top_k(centered_calib_points_two_temp[:,1], k=tf.shape(centered_calib_points_two_temp)[0]).indices
reordered_points_two_y_temp = tf.gather(centered_calib_points_two_temp, indices_two_y_temp, axis=0)

centered_calib_points_two_ty  = tf.slice(reordered_points_two_y_temp, [0, 0], [tf.shape(reordered_points_two_y_temp)[0]/2, -1])

#mask_twoy  = tf.greater(centered_calib_points_two_temp[:,1],0)

#centered_calib_points_two_ty  = tf.boolean_mask(centered_calib_points_two_temp, mask_twoy)

#mask_threey  = tf.greater(centered_calib_points_three_temp[:,1],0)

#centered_calib_points_three_ty  = tf.boolean_mask(centered_calib_points_three_temp, mask_threey)
indices_three_y_temp = tf.nn.top_k(centered_calib_points_three_temp[:,1], k=tf.shape(centered_calib_points_three_temp)[0]).indices
reordered_points_three_y_temp = tf.gather(centered_calib_points_three_temp, indices_three_y_temp, axis=0)

centered_calib_points_three_ty  = tf.slice(reordered_points_three_y_temp, [0, 0], [tf.shape(reordered_points_three_y_temp)[0]/2, -1])


#mask_oneyi  = tf.less(centered_calib_points_one_temp[:,1],0)
centered_calib_points_one_tyi  = tf.slice(reordered_points_one_y_temp, [tf.shape(reordered_points_one_y_temp)[0]/2 , 0], [tf.shape(reordered_points_one_y_temp)[0]/2, -1])

centered_calib_points_two_tyi  = tf.slice(reordered_points_two_y_temp, [tf.shape(reordered_points_two_y_temp)[0]/2 , 0], [tf.shape(reordered_points_two_y_temp)[0]/2, -1])


centered_calib_points_three_tyi  = tf.slice(reordered_points_three_y_temp, [tf.shape(reordered_points_three_y_temp)[0]/2 , 0], [tf.shape(reordered_points_three_y_temp)[0]/2, -1])










centered_calib_points_one  = tf.cond(tf.shape(centered_calib_points_one_t )[0] >= 1000, lambda: tf.slice(centered_calib_points_one_t , [0,0], [1000,-1]), lambda: tf.concat([centered_calib_points_one_t, tf.zeros([1000-tf.shape(centered_calib_points_one_t)[0],3])], axis = 0))
centered_calib_points_two = tf.cond(tf.shape(centered_calib_points_two_t )[0] >= 1000, lambda: tf.slice(centered_calib_points_two_t , [0,0], [1000,-1]), lambda: tf.concat([centered_calib_points_two_t, tf.zeros([1000-tf.shape(centered_calib_points_two_t)[0],3])], axis = 0))
centered_calib_points_three = tf.cond(tf.shape(centered_calib_points_three_t )[0] >= 1000, lambda: tf.slice(centered_calib_points_three_t , [0,0], [1000,-1]), lambda: tf.concat([centered_calib_points_three_t, tf.zeros([1000-tf.shape(centered_calib_points_three_t)[0],3])], axis = 0))


dist_12 = -1.0 * tf.log(tf.reduce_sum(tf.sqrt(tf.multiply(centered_calib_points_one[:,2], centered_calib_points_two[:,2])+0.0001)))
dist_13 = -1.0 * tf.log(tf.reduce_sum(tf.sqrt(tf.multiply(centered_calib_points_one[:,2], centered_calib_points_three[:,2])+0.0001)))
dist_23 = -1.0 * tf.log(tf.reduce_sum(tf.sqrt(tf.multiply(centered_calib_points_three[:,2], centered_calib_points_two[:,2])+ 0.0001)))

_, var_1 = tf.nn.moments(centered_calib_points_one[:,2], axes = [0])
_, var_2 = tf.nn.moments(centered_calib_points_two[:,2], axes = [0])
_, var_3 = tf.nn.moments(centered_calib_points_three[:,2], axes = [0])



sim_term = -1.0 * (dist_12 + dist_13  + dist_23)/3
info_term = -1.0 * (tf.sqrt(var_1) + tf.sqrt(var_2) + tf.sqrt(var_3))/3

rot_loss = 0.01 *  sim_term + 0.03 * info_term

def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)


def atan2(y, x):
    angle = tf.select(tf.greater(x,0.0), tf.atan(y/x), tf.zeros_like(x))
    angle = tf.select(tf.logical_and(tf.less(x,0.0),  tf.greater_equal(y,0.0)), tf.atan(y/x) + np.pi, angle)
    angle = tf.select(tf.logical_and(tf.less(x,0.0),  tf.less(y,0.0)), tf.atan(y/x) - np.pi, angle)
    angle = tf.select(tf.logical_and(tf.equal(x,0.0), tf.greater(y,0.0)), 0.5*np.pi * tf.ones_like(x), angle)
    angle = tf.select(tf.logical_and(tf.equal(x,0.0), tf.less(y,0.0)), -0.5*np.pi * tf.ones_like(x), angle)
    angle = tf.select(tf.logical_and(tf.equal(x,0.0), tf.equal(y,0.0)), np.nan * tf.zeros_like(x), angle)
    return angle


r_one_temp = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_one_t), axis=1, keepdims = True))
r_one = tf.divide(r_one_temp,tf.reduce_max(r_one_temp, axis = 0, keep_dims = True))
theta_one = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_one_t[:,2],1), tf.maximum(r_one_temp,0.001)),0.99),-0.99))
phi_one = tf.atan2(tf.expand_dims(centered_calib_points_one_t[:,1],1),tf.expand_dims(centered_calib_points_one_t[:,0],1))


r_two_temp = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_two_t), axis=1, keepdims = True))
r_two = tf.divide(r_two_temp,tf.reduce_max(r_two_temp, axis = 0, keep_dims = True))

#r_two = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_two), axis=1, keepdims = True))
theta_two = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_two_t[:,2],1), tf.maximum(r_two_temp,0.001)),0.99),-0.99))
phi_two = tf.atan2(tf.expand_dims(centered_calib_points_two_t[:,1],1),tf.expand_dims(centered_calib_points_two_t[:,0],1))

r_three_temp = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_three_t), axis=1, keepdims = True))
r_three = tf.divide(r_three_temp,tf.reduce_max(r_three_temp, axis = 0, keep_dims = True))

#r_three = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_three), axis=1, keepdims = True))
theta_three = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_three_t [:,2],1), tf.maximum(r_three_temp,0.001)),0.99),-0.99))
phi_three  = tf.atan2(tf.expand_dims(centered_calib_points_three_t [:,1],1),tf.expand_dims(centered_calib_points_three_t [:,0],1))


rp = tf.concat([r_one, r_two, r_three], axis = 0)
thetap = tf.concat([theta_one, theta_two, theta_three], axis = 0)
phip = tf.concat([phi_one, phi_two, phi_three], axis = 0)


#########################################################################################################################

r_one_temp_i = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_one_t_i), axis=1, keepdims = True))
r_one_i = tf.divide(r_one_temp_i,tf.reduce_max(r_one_temp_i, axis = 0, keep_dims = True))
theta_one_i = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_one_t_i[:,2],1), tf.maximum(r_one_temp_i,0.001)),0.99),-0.99))
phi_one_i = tf.atan2(tf.expand_dims(centered_calib_points_one_t_i[:,1],1),tf.expand_dims(centered_calib_points_one_t_i[:,0],1))


r_two_temp_i = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_two_t_i), axis=1, keepdims = True))
r_two_i = tf.divide(r_two_temp_i,tf.reduce_max(r_two_temp_i, axis = 0, keep_dims = True))

#Cr_two = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_two), axis=1, keepdims = True))
theta_two_i = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_two_t_i[:,2],1), tf.maximum(r_two_temp_i,0.001)),0.99),-0.99))
phi_two_i = tf.atan2(tf.expand_dims(centered_calib_points_two_t_i[:,1],1),tf.expand_dims(centered_calib_points_two_t_i[:,0],1))

r_three_temp_i = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_three_t_i), axis=1, keepdims = True))
r_three_i = tf.divide(r_three_temp_i,tf.reduce_max(r_three_temp_i, axis = 0, keep_dims = True))

#r_three = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_three), axis=1, keepdims = True))
theta_three_i = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_three_t_i [:,2],1), tf.maximum(r_three_temp_i,0.001)),0.99),-0.99))
phi_three_i  = tf.atan2(tf.expand_dims(centered_calib_points_three_t_i [:,1],1),tf.expand_dims(centered_calib_points_three_t_i [:,0],1))


r_i = tf.concat([r_one_i, r_two_i, r_three_i], axis = 0)
theta_i = tf.concat([theta_one_i, theta_two_i, theta_three_i], axis = 0)
phi_i = tf.concat([phi_one_i, phi_two_i, phi_three_i], axis = 0)


###############################################################################################################################

r_one_tempx = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_one_tx), axis=1, keepdims = True))
r_onex = tf.divide(r_one_tempx,tf.reduce_max(r_one_tempx, axis = 0, keep_dims = True))
theta_onex = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_one_tx[:,2],1), tf.maximum(r_one_tempx,0.001)),0.99),-0.99))
phi_onex = tf.atan2(tf.expand_dims(centered_calib_points_one_tx[:,1],1),tf.expand_dims(centered_calib_points_one_tx[:,0],1))


r_two_tempx = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_two_tx), axis=1, keepdims = True))
r_twox = tf.divide(r_two_tempx,tf.reduce_max(r_two_tempx, axis = 0, keep_dims = True))

#r_two = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_two), axis=1, keepdims = True))
theta_twox = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_two_tx[:,2],1), tf.maximum(r_two_tempx,0.001)),0.99),-0.99))
phi_twox = tf.atan2(tf.expand_dims(centered_calib_points_two_tx[:,1],1),tf.expand_dims(centered_calib_points_two_tx[:,0],1))

r_three_tempx = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_three_tx), axis=1, keepdims = True))
r_threex = tf.divide(r_three_tempx,tf.reduce_max(r_three_tempx, axis = 0, keep_dims = True))

#r_three = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_three), axis=1, keepdims = True))
theta_threex = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_three_tx [:,2],1), tf.maximum(r_three_tempx,0.001)),0.99),-0.99))
phi_threex  = tf.atan2(tf.expand_dims(centered_calib_points_three_tx [:,1],1),tf.expand_dims(centered_calib_points_three_tx [:,0],1))


rx = tf.concat([r_onex, r_twox, r_threex], axis = 0)
thetax = tf.concat([theta_onex, theta_twox, theta_threex], axis = 0)
phix = tf.concat([phi_onex, phi_twox, phi_threex], axis = 0)


#########################################################################################################################################

r_one_tempxi = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_one_txi), axis=1, keepdims = True))
r_onexi = tf.divide(r_one_tempxi,tf.reduce_max(r_one_tempxi, axis = 0, keep_dims = True))
theta_onexi = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_one_txi[:,2],1), tf.maximum(r_one_tempxi,0.001)),0.99),-0.99))
phi_onexi = tf.atan2(tf.expand_dims(centered_calib_points_one_txi[:,1],1),tf.expand_dims(centered_calib_points_one_txi[:,0],1))


r_two_tempxi = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_two_txi), axis=1, keepdims = True))
r_twoxi = tf.divide(r_two_tempxi,tf.reduce_max(r_two_tempxi, axis = 0, keep_dims = True))

#r_two = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_two), axis=1, keepdims = True))
theta_twoxi = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_two_txi[:,2],1), tf.maximum(r_two_tempxi,0.001)),0.99),-0.99))
phi_twoxi = tf.atan2(tf.expand_dims(centered_calib_points_two_txi[:,1],1),tf.expand_dims(centered_calib_points_two_txi[:,0],1))

r_three_tempxi = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_three_txi), axis=1, keepdims = True))
r_threexi = tf.divide(r_three_tempxi,tf.reduce_max(r_three_tempxi, axis = 0, keep_dims = True))

#r_three = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_three), axis=1, keepdims = True))
theta_threexi = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_three_txi [:,2],1), tf.maximum(r_three_tempxi,0.001)),0.99),-0.99))
phi_threexi  = tf.atan2(tf.expand_dims(centered_calib_points_three_txi [:,1],1),tf.expand_dims(centered_calib_points_three_txi [:,0],1))


rxi = tf.concat([r_onexi, r_twoxi, r_threexi], axis = 0)
thetaxi = tf.concat([theta_onexi, theta_twoxi, theta_threexi], axis = 0)
phixi = tf.concat([phi_onexi, phi_twoxi, phi_threexi], axis = 0)


#############################################################################################################33


r_one_tempy = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_one_ty), axis=1, keepdims = True))
r_oney = tf.divide(r_one_tempy,tf.reduce_max(r_one_tempy, axis = 0, keep_dims = True))
theta_oney = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_one_ty[:,2],1), tf.maximum(r_one_tempy,0.001)),0.99),-0.99))
phi_oney = tf.atan2(tf.expand_dims(centered_calib_points_one_ty[:,1],1),tf.expand_dims(centered_calib_points_one_ty[:,0],1))


r_two_tempy = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_two_ty), axis=1, keepdims = True))
r_twoy = tf.divide(r_two_tempy,tf.reduce_max(r_two_tempy, axis = 0, keep_dims = True))

#r_two = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_two), axis=1, keepdims = True))
theta_twoy = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_two_ty[:,2],1), tf.maximum(r_two_tempy,0.001)),0.99),-0.99))
phi_twoy = tf.atan2(tf.expand_dims(centered_calib_points_two_ty[:,1],1),tf.expand_dims(centered_calib_points_two_ty[:,0],1))

r_three_tempy = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_three_ty), axis=1, keepdims = True))
r_threey = tf.divide(r_three_tempy,tf.reduce_max(r_three_tempy, axis = 0, keep_dims = True))

#r_three = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_three), axis=1, keepdims = True))
theta_threey = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_three_ty [:,2],1), tf.maximum(r_three_tempy,0.001)),0.99),-0.99))
phi_threey  = tf.atan2(tf.expand_dims(centered_calib_points_three_ty [:,1],1),tf.expand_dims(centered_calib_points_three_ty [:,0],1))


ry = tf.concat([r_oney, r_twoy, r_threey], axis = 0)
thetay = tf.concat([theta_oney, theta_twoy, theta_threey], axis = 0)
phiy = tf.concat([phi_oney, phi_twoy, phi_threey], axis = 0)

##########################################################################################################################################333


r_one_tempyi = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_one_tyi), axis=1, keepdims = True))
r_oneyi = tf.divide(r_one_tempyi,tf.reduce_max(r_one_tempyi, axis = 0, keep_dims = True))
theta_oneyi = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_one_tyi[:,2],1), tf.maximum(r_one_tempyi,0.001)),0.99),-0.99))
phi_oneyi = tf.atan2(tf.expand_dims(centered_calib_points_one_tyi[:,1],1),tf.expand_dims(centered_calib_points_one_tyi[:,0],1))


r_two_tempyi = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_two_tyi), axis=1, keepdims = True))
r_twoyi = tf.divide(r_two_tempyi,tf.reduce_max(r_two_tempyi, axis = 0, keep_dims = True))

#r_two = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_two), axis=1, keepdims = True))
theta_twoyi = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_two_tyi[:,2],1), tf.maximum(r_two_tempyi,0.001)),0.99),-0.99))
phi_twoyi = tf.atan2(tf.expand_dims(centered_calib_points_two_tyi[:,1],1),tf.expand_dims(centered_calib_points_two_tyi[:,0],1))

r_three_tempyi = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_three_tyi), axis=1, keepdims = True))
r_threeyi = tf.divide(r_three_tempyi,tf.reduce_max(r_three_tempyi, axis = 0, keep_dims = True))

#r_three = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_three), axis=1, keepdims = True))
theta_threeyi = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_three_tyi [:,2],1), tf.maximum(r_three_tempyi,0.001)),0.99),-0.99))
phi_threeyi  = tf.atan2(tf.expand_dims(centered_calib_points_three_tyi [:,1],1),tf.expand_dims(centered_calib_points_three_tyi [:,0],1))


ryi = tf.concat([r_oneyi, r_twoyi, r_threeyi], axis = 0)
thetayi = tf.concat([theta_oneyi, theta_twoyi, theta_threeyi], axis = 0)
phiyi = tf.concat([phi_oneyi, phi_twoyi, phi_threeyi], axis = 0)









r = tf.concat([rp, r_i, rx,rxi,ry,ryi], axis = 0)
theta = tf.concat([thetap, theta_i, thetax, thetaxi, thetay, thetayi], axis = 0)
phi = tf.concat([phip, phi_i, phix, phixi, phiy, phiyi], axis = 0)




# polar_coordinates_one = tf.concat([r,theta, phi], axis=1)

print("######################################################")
print(phi)





def spherical_harmonic(m,l):
	return math.pow(-1.0,m) * math.sqrt(((2.0*l + 1.0)*7.0/88.0) * (math.factorial(l-m)*1.0/(math.factorial(l+m)*1.0)))  
	
def radial_poly(rho, m, n):
	if n == 0 and m == 0:
		return tf.ones(tf.shape(rho))
        if n == 1 and m == 1:
                return rho
	if n == 2 and m == 0:
                return 2.0 * tf.pow(rho,2) - 1
        if n == 2 and m == 2:
                return tf.pow(rho,2)
        if n == 3 and m == 1:
                return 3.0* tf.pow(rho, 3) - 2.0 * rho
        if n == 3 and m == 3:
                return tf.pow(rho,3)
	if n == 4 and m == 0:
		return 6.0 * tf.pow(rho,4) - 6.0 * tf.pow(rho,2) + 1
        if n == 4 and m == 2:
                return 4.0* tf.pow(rho, 3.5) - 3.0 * tf.pow(rho,1.5)
        if n == 4 and m == 4:
                return tf.pow(rho,3.5)
        if n == 5 and m == 1:
                return 10.0* tf.pow(rho, 4.5) - 12.0 * tf.pow(rho, 2.5) + 3.0 * tf.sqrt(rho+0.0001)
        if n == 5 and m == 3:
                return 5.0 * tf.pow(rho, 4.5) - 4.0 * tf.pow(rho, 2.5)
        if n == 5 and m == 5:
                return tf.pow(rho,4.5)
        if n == 6 and m == 2:
                return 10.0* tf.pow(rho, 5.5) - 20.0 * 10.0* tf.pow(rho, 3.5) + 6.0 * tf.pow(rho,1.5)
        if n == 6 and m == 4:
                return 6.0 * tf.pow(rho, 5.5)  - 5.0 * tf.pow(rho,3.5)
        if n == 6 and m == 6:
                return tf.pow(rho, 5.5)

###########################################################################

y_0_0 =  spherical_harmonic(0.0,0.0)     * (tf.zeros(tf.shape(theta)) + 1)
y_0_1 =  spherical_harmonic(0.0,1.0)   * tf.cos(theta)
y_1_1 =   spherical_harmonic(1.0,1.0)   * (-1.0) * tf.sqrt(1-tf.square(tf.cos(theta)))
y_0_2 =  spherical_harmonic(0.0,2.0)    *(1.0/2.0) * (3* tf.square(tf.cos(theta)) - 1)
y_1_2 =   spherical_harmonic(1.0,2.0)   * (-1.0) * tf.sqrt(1-tf.square(tf.cos(theta))) * 3.0 * tf.cos(theta)
y_2_2 =   spherical_harmonic(2.0,2.0)   * (1-tf.square(tf.cos(theta))) * 3.0
y_0_3 =  spherical_harmonic(0.0,3.0)    * (1.0/2.0) * (5 * tf.pow(tf.cos(theta),3) - 3 * tf.cos(theta))
y_1_3 =   spherical_harmonic(1.0,3.0)    * (-1.0) *  (1.0/2.0)*  tf.sqrt(1-tf.square(tf.cos(theta))) * (15 * tf.square(tf.cos(theta)) - 3 )
y_2_3 =  spherical_harmonic(2.0,3.0)    * 15 * tf.cos(theta) * (1.0-tf.square(tf.cos(theta)))
y_3_3 =   spherical_harmonic(3.0,3.0)     * (-1.0) * 15.0 * tf.pow((1.0-tf.square(tf.cos(theta))),3.0/2.0)
y_0_4 =  spherical_harmonic(0.0,4.0) * (1.0/8.0) * (35.0 * tf.pow(tf.cos(theta),4) - 30.0 * tf.square(tf.cos(theta)) + 3)
y_1_4 = spherical_harmonic(1.0,4.0) * (-5.0/2.0) * (7.0 * tf.pow(tf.cos(theta),3) - 3.0 * tf.cos(theta)) * (tf.sqrt(1-tf.square(tf.cos(theta))))
y_2_4 = spherical_harmonic(2.0,4.0) * (15.0/2.0) * (7.0 * tf.square(tf.cos(theta)) - 1.0) * (1.0 - tf.square(tf.cos(theta)))
y_3_4 = spherical_harmonic(3.0,4.0) * (-105.0) * tf.cos(theta) * tf.pow(tf.sqrt(1.0-tf.square(tf.cos(theta))), 3.0/2)
y_4_4 = spherical_harmonic(4.0,4.0) * (105.0) * tf.square(1.0 - tf.square(tf.cos(theta)))

test_10 =  y_4_4
print("1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(math.factorial(4.0-4.0))


u_0_0_0 = tf.multiply(y_0_0, tf.cos(0.0)) * radial_poly(r,0,0)
u_0_1_1 = tf.multiply(y_0_1, tf.cos(0.0)) * radial_poly(r,1,1)
u_1_1_1 = tf.multiply(y_1_1, tf.cos(phi)) * radial_poly(r,1,1)
u_0_0_2 = tf.multiply(y_0_2, tf.cos(0.0)) * radial_poly(r,0,2)
u_0_2_2 = tf.multiply(y_0_2, tf.cos(0.0)) * radial_poly(r,2,2)
u_1_2_2 = tf.multiply(y_1_2, tf.cos(phi)) * radial_poly(r,2,2)
u_2_2_2 = tf.multiply(y_2_2, tf.cos(2.0*phi)) * radial_poly(r,2,2)
u_0_1_3 = tf.multiply(y_0_1, tf.cos(0.0)) * radial_poly(r,1,3)
u_1_1_3 = tf.multiply(y_1_1, tf.cos(phi)) * radial_poly(r,1,3)
u_0_3_3 = tf.multiply(y_0_3, tf.cos(0.0)) * radial_poly(r,3,3)
u_1_3_3 = tf.multiply(y_1_3, tf.cos(phi)) * radial_poly(r,3,3)
u_2_3_3 = tf.multiply(y_2_3, tf.cos(2.0*phi)) * radial_poly(r,3,3)
u_3_3_3 = tf.multiply(y_3_3, tf.cos(3.0*phi)) * radial_poly(r,3,3)
u_0_0_4 = tf.multiply(y_0_0, tf.cos(0.0)) * radial_poly(r,0,4)
u_0_2_4 = tf.multiply(y_0_2, tf.cos(0.0)) * radial_poly(r,2,4)
u_1_2_4 = tf.multiply(y_1_2, tf.cos(phi)) * radial_poly(r,2,4)
u_2_2_4 = tf.multiply(y_2_2, tf.cos(2.0*phi)) * radial_poly(r,2,4)
u_0_4_4 = tf.multiply(y_0_4, tf.cos(0.0))* radial_poly(r,4,4)
u_1_4_4 = tf.multiply(y_1_4, tf.cos(phi)) * radial_poly(r,4,4)
u_2_4_4 = tf.multiply(y_2_4, tf.cos(2.0 * phi))* radial_poly(r,4,4)
u_3_4_4 = tf.multiply(y_3_4, tf.cos(3.0 * phi))* radial_poly(r,4,4)
u_4_4_4 = tf.multiply(y_4_4, tf.cos(4.0 * phi))* radial_poly(r,4,4)


v_0_0_0 = tf.multiply(y_0_0, tf.sin(0.0)) * radial_poly(r,0,0)
v_0_1_1 = tf.multiply(y_0_1, tf.sin(0.0)) * radial_poly(r,1,1)
v_1_1_1 = tf.multiply(y_1_1, tf.sin(phi)) * radial_poly(r,1,1)
v_0_0_2 = tf.multiply(y_0_2, tf.sin(0.0)) * radial_poly(r,0,2)
v_0_2_2 = tf.multiply(y_0_2, tf.sin(0.0)) * radial_poly(r,2,2)
v_1_2_2 = tf.multiply(y_1_2, tf.sin(phi)) * radial_poly(r,2,2)
v_2_2_2 = tf.multiply(y_2_2, tf.sin(2.0*phi)) * radial_poly(r,2,2)
v_0_1_3 = tf.multiply(y_0_1, tf.sin(0.0)) * radial_poly(r,1,3)
v_1_1_3 = tf.multiply(y_1_1, tf.sin(phi)) * radial_poly(r,1,3)
v_0_3_3 = tf.multiply(y_0_3, tf.sin(0.0)) * radial_poly(r,3,3)
v_1_3_3 = tf.multiply(y_1_3, tf.sin(phi)) * radial_poly(r,3,3)
v_2_3_3 = tf.multiply(y_2_3, tf.sin(2.0*phi)) * radial_poly(r,3,3)
v_3_3_3 = tf.multiply(y_3_3, tf.sin(3.0*phi)) * radial_poly(r,3,3)
v_0_0_4 = tf.multiply(y_0_0, tf.sin(0.0)) * radial_poly(r,0,4)
v_0_2_4 = tf.multiply(y_0_2, tf.sin(0.0)) * radial_poly(r,2,4)
v_1_2_4 = tf.multiply(y_1_2, tf.sin(phi)) * radial_poly(r,2,4)
v_2_2_4 = tf.multiply(y_2_2, tf.sin(2.0*phi)) * radial_poly(r,2,4)
v_0_4_4 = tf.multiply(y_0_4, tf.sin(0.0))* radial_poly(r,4,4)
v_1_4_4 = tf.multiply(y_1_4, tf.sin(phi)) * radial_poly(r,4,4)
v_2_4_4 = tf.multiply(y_2_4, tf.sin(2.0 * phi))* radial_poly(r,4,4)
v_3_4_4 = tf.multiply(y_3_4, tf.sin(3.0 * phi))* radial_poly(r,4,4)
v_4_4_4 = tf.multiply(y_4_4, tf.sin(4.0 * phi))* radial_poly(r,4,4)

################################################################################################
keep_prob = tf.placeholder(tf.float32, name="keep_probabilty")


V = tf.concat([v_0_0_0, v_0_1_1, v_1_1_1, v_0_0_2, v_0_2_2, v_1_2_2, v_2_2_2, 
			v_0_1_3, v_1_1_3, v_0_3_3, v_1_3_3, v_2_3_3, v_3_3_3, 
				v_0_0_4, v_0_2_4, v_1_2_4, v_2_2_4, v_0_4_4, v_1_4_4, v_2_4_4, v_3_4_4, v_4_4_4] ,  axis=1)


U = tf.concat([u_0_0_0, u_0_1_1, u_1_1_1, u_0_0_2, u_0_2_2, u_1_2_2, u_2_2_2, 
                        u_0_1_3, u_1_1_3, u_0_3_3, u_1_3_3, u_2_3_3, u_3_3_3, 
                                u_0_0_4, u_0_2_4, u_1_2_4, u_2_2_4, u_0_4_4, u_1_4_4, u_2_4_4, u_3_4_4, u_4_4_4] ,  axis=1)

X_ = tf.concat([U,V] ,  axis=1)

X__1 = tf.slice(X_, [0,0], [tf.shape(r_one)[0], -1])
X__2 = tf.slice(X_, [tf.shape(r_one)[0], 0], [tf.shape(r_two)[0], -1])
X__3 = tf.slice(X_, [tf.shape(r_two)[0] + tf.shape(r_one)[0], 0], [tf.shape(r_three)[0], -1])




X_10  = 0.0001 * tf.transpose(X__1,[1,0] )
X_20  = 0.0001 * tf.transpose(X__2,[1,0] )
X_30  = 0.0001 * tf.transpose(X__3,[1,0] )

X_cal_1, p_1 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(44) + 1.0/4.0 * tf.matmul(tf.eye(44)-tf.matmul(x,X__1),tf.matmul(3.0*tf.eye(44)-tf.matmul(x,X__1),3.0*tf.eye(44)-tf.matmul(x,X__1))),x),i+1)
    ,(X_10, 0))

X_cal_2, p_2 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(44) + 1.0/4.0 * tf.matmul(tf.eye(44)-tf.matmul(x,X__2),tf.matmul(3.0*tf.eye(44)-tf.matmul(x,X__2),3.0*tf.eye(44)-tf.matmul(x,X__2))),x),i+1)
    ,(X_20, 0))

X_cal_3, p_3 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(44) + 1.0/4.0 * tf.matmul(tf.eye(44)-tf.matmul(x,X__3),tf.matmul(3.0*tf.eye(44)-tf.matmul(x,X__3),3.0*tf.eye(44)-tf.matmul(x,X__3))),x),i+1)
    ,(X_30, 0))



C_1 = tf.tile(tf.expand_dims(tf.matmul(X_cal_1,r_one), axis =0), [64,1,1])

C_2 = tf.tile(tf.expand_dims(tf.matmul(X_cal_2, r_two), axis =0), [64,1,1])
C_3 = tf.tile(tf.expand_dims(tf.matmul(X_cal_3,r_three), axis =0), [64,1,1])

###############################################################################################################################


x_filter1 =  tf.get_variable("a_xfilter1", [64,150,1])
x_filter2 =  tf.get_variable("a_xfilter2", [64,150,1])
x_filter3 =  tf.get_variable("a_xfilter3", [64,150,1])


C_1f = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_cal_1,[0,0],[44,150]), axis =0), [64,1,1]), x_filter1)
C_2f = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_cal_2,[0,0],[44,150]), axis =0), [64,1,1]), x_filter2)
C_3f = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_cal_3,[0,0],[44,150]), axis =0), [64,1,1]), x_filter3)

f_mapz1 = tf.matmul(C_1, tf.transpose(C_1f, [0,2,1]))
f_mapz2 = tf.matmul(C_2, tf.transpose(C_2f, [0,2,1]))
f_mapz3 = tf.matmul(C_3, tf.transpose(C_3f, [0,2,1]))

#tf.maximum(tf.maximum(f_mapz1,f_mapz2),f_mapz3)

f_mapz = tf.nn.dropout(tf.nn.relu(f_mapz1 +f_mapz2 + f_mapz3), keep_prob = keep_prob)



#########################################################################################3

X__1i = tf.slice(X_, [tf.shape(r_two)[0] + tf.shape(r_one)[0] + tf.shape(r_three)[0],0], [tf.shape(r_one_i)[0], -1])
X__2i = tf.slice(X_, [ tf.shape(r_two)[0] + tf.shape(r_one)[0] + tf.shape(r_three)[0] +tf.shape(r_one_i)[0],0   ],[ tf.shape(r_two_i)[0]  , -1])
X__3i = tf.slice(X_, [tf.shape(r_two)[0] + tf.shape(r_one)[0] + tf.shape(r_three)[0] +tf.shape(r_one_i)[0] +  tf.shape(r_two_i)[0], 0], [tf.shape(r_three_i)[0], -1])




X_10i  = 0.0001 * tf.transpose(X__1i,[1,0] )
X_20i  = 0.0001 * tf.transpose(X__2i,[1,0] )
X_30i  = 0.0001 * tf.transpose(X__3i,[1,0] )

X_cal_1i, p_1 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(44) + 1.0/4.0 * tf.matmul(tf.eye(44)-tf.matmul(x,X__1i),tf.matmul(3.0*tf.eye(44)-tf.matmul(x,X__1i),3.0*tf.eye(44)-tf.matmul(x,X__1i))),x),i+1)
    ,(X_10i, 0))

X_cal_2i, p_2 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(44) + 1.0/4.0 * tf.matmul(tf.eye(44)-tf.matmul(x,X__2i),tf.matmul(3.0*tf.eye(44)-tf.matmul(x,X__2i),3.0*tf.eye(44)-tf.matmul(x,X__2i))),x),i+1)
    ,(X_20i, 0))

X_cal_3i, p_3 = tf.while_loop(lambda x, i: tf.logical_and(i < 3 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(44) + 1.0/4.0 * tf.matmul(tf.eye(44)-tf.matmul(x,X__3i),tf.matmul(3.0*tf.eye(44)-tf.matmul(x,X__3i),3.0*tf.eye(44)-tf.matmul(x,X__3i))),x),i+1)
    ,(X_30i, 0))



C_1i = tf.tile(tf.expand_dims(tf.matmul(X_cal_1i,r_one_i), axis =0), [64,1,1])

C_2i = tf.tile(tf.expand_dims(tf.matmul(X_cal_2i, r_two_i), axis =0), [64,1,1])
C_3i = tf.tile(tf.expand_dims(tf.matmul(X_cal_3i,r_three_i), axis =0), [64,1,1])

##########################################################################


xi_filter1 =  tf.get_variable("a_xfilter1i", [64, 150,1])
xi_filter2 =  tf.get_variable("a_xfilter2i", [64, 150,1])
xi_filter3 =  tf.get_variable("a_xfilter3i", [64, 150,1])


C_1fi = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_cal_1i,[0,0],[44,150]),axis =0), [64,1,1]), xi_filter1)
C_2fi = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_cal_2i,[0,0],[44,150]),axis =0), [64,1,1]), xi_filter2)
C_3fi = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_cal_3i,[0,0],[44,150]),axis =0), [64,1,1]), xi_filter3)

f_mapz1i = tf.matmul(C_1i, tf.transpose(C_1fi, [0,2,1]))
f_mapz2i = tf.matmul(C_2i, tf.transpose(C_2fi, [0,2,1]))
f_mapz3i = tf.matmul(C_3i, tf.transpose(C_3fi, [0,2,1]))

f_mapzi = tf.nn.dropout(tf.nn.relu(f_mapz1i +f_mapz2i + f_mapz3i),keep_prob = keep_prob)


##########################################################################################

s1 = tf.shape(r_two)[0] + tf.shape(r_one)[0] + tf.shape(r_three)[0] + tf.shape(r_one_i)[0]
s2 = s1 + tf.shape(r_onex)[0]
s3 = s2 + tf.shape(r_twox)[0]



X__1x = tf.slice(X_, [s1,0], [tf.shape(r_onex)[0], -1])
X__2x = tf.slice(X_, [ s2, 0   ],[ tf.shape(r_twox)[0]  , -1])
X__3x = tf.slice(X_, [s3, 0], [tf.shape(r_threex)[0], -1])




X_10x  = 0.0001 * tf.transpose(X__1x,[1,0] )
X_20x  = 0.0001 * tf.transpose(X__2x,[1,0] )
X_30x  = 0.0001 * tf.transpose(X__3x,[1,0] )

X_cal_1x, p_1 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(44) + 1.0/4.0 * tf.matmul(tf.eye(44)-tf.matmul(x,X__1x),tf.matmul(3.0*tf.eye(44)-tf.matmul(x,X__1x),3.0*tf.eye(44)-tf.matmul(x,X__1x))),x),i+1)
    ,(X_10x, 0))

X_cal_2x, p_2 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(44) + 1.0/4.0 * tf.matmul(tf.eye(44)-tf.matmul(x,X__2x),tf.matmul(3.0*tf.eye(44)-tf.matmul(x,X__2x),3.0*tf.eye(44)-tf.matmul(x,X__2x))),x),i+1)
    ,(X_20x, 0))

X_cal_3x, p_3 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(44) + 1.0/4.0 * tf.matmul(tf.eye(44)-tf.matmul(x,X__3x),tf.matmul(3.0*tf.eye(44)-tf.matmul(x,X__3x),3.0*tf.eye(44)-tf.matmul(x,X__3x))),x),i+1)
    ,(X_30x, 0))



C_1x = tf.tile(tf.expand_dims(tf.matmul(X_cal_1x,r_onex), axis =0), [64,1,1])

C_2x = tf.tile(tf.expand_dims(tf.matmul(X_cal_2x, r_twox), axis =0), [64,1,1])
C_3x = tf.tile(tf.expand_dims(tf.matmul(X_cal_3x,r_threex), axis =0), [64,1,1])

###########################################################################################################################################33

x_filter1x =  tf.get_variable("a_xfilter1x", [64,150,1])
x_filter2x =  tf.get_variable("a_xfilter2x", [64,150,1])
x_filter3x =  tf.get_variable("a_xfilter3x", [64,150,1])


C_1fx = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_cal_1x,[0,0],[44,150]),axis =0), [64,1,1]), x_filter1x)
C_2fx = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_cal_2x,[0,0],[44,150]),axis =0), [64,1,1]), x_filter2x)
C_3fx = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_cal_3x,[0,0],[44,150]),axis =0), [64,1,1]), x_filter3x)

f_mapx1x = tf.matmul(C_1x, tf.transpose(C_1fx, [0,2,1]))
f_mapx2x = tf.matmul(C_2x, tf.transpose(C_2fx, [0,2,1]))
f_mapx3x = tf.matmul(C_3x, tf.transpose(C_3fx, [0,2,1]))

f_mapx = tf.nn.dropout(tf.nn.relu(f_mapx1x +f_mapx2x + f_mapx3x),keep_prob = keep_prob)



##############################################################33

s4 = s3 + tf.shape(r_threex)[0]
s5 = s4 + tf.shape(r_onexi)[0]
s6 = s5 + tf.shape(r_twoxi)[0]



X__1xi = tf.slice(X_, [s4,0], [tf.shape(r_onexi)[0], -1])
X__2xi = tf.slice(X_, [ s5, 0   ],[ tf.shape(r_twoxi)[0]  , -1])
X__3xi = tf.slice(X_, [s6, 0], [tf.shape(r_threexi)[0], -1])




X_10xi  = 0.0001 * tf.transpose(X__1xi,[1,0] )
X_20xi  = 0.0001 * tf.transpose(X__2xi,[1,0] )
X_30xi  = 0.0001 * tf.transpose(X__3xi,[1,0] )

X_cal_1xi, p_1 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(44) + 1.0/4.0 * tf.matmul(tf.eye(44)-tf.matmul(x,X__1xi),tf.matmul(3.0*tf.eye(44)-tf.matmul(x,X__1xi),3.0*tf.eye(44)-tf.matmul(x,X__1xi))),x),i+1)
    ,(X_10xi, 0))

X_cal_2xi, p_2 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(44) + 1.0/4.0 * tf.matmul(tf.eye(44)-tf.matmul(x,X__2xi),tf.matmul(3.0*tf.eye(44)-tf.matmul(x,X__2xi),3.0*tf.eye(44)-tf.matmul(x,X__2xi))),x),i+1)
    ,(X_20xi, 0))

X_cal_3xi, p_3 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(44) + 1.0/4.0 * tf.matmul(tf.eye(44)-tf.matmul(x,X__3xi),tf.matmul(3.0*tf.eye(44)-tf.matmul(x,X__3xi),3.0*tf.eye(44)-tf.matmul(x,X__3xi))),x),i+1)
    ,(X_30xi, 0))



C_1xi = tf.tile(tf.expand_dims(tf.matmul(X_cal_1xi,r_onexi), axis =0), [64,1,1])

C_2xi = tf.tile(tf.expand_dims(tf.matmul(X_cal_2xi, r_twoxi), axis =0), [64,1,1])
C_3xi = tf.tile(tf.expand_dims(tf.matmul(X_cal_3xi,r_threexi), axis =0), [64,1,1])

#####################################################################3


x_filter1xi =  tf.get_variable("a_xfilter1xi", [64,150,1])
x_filter2xi =  tf.get_variable("a_xfilter2xi", [64,150,1])
x_filter3xi =  tf.get_variable("a_xfilter3xi", [64,150,1])


C_1fxi = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_cal_1xi,[0,0],[44,150]),axis =0), [64,1,1]), x_filter1xi)
C_2fxi = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_cal_2xi,[0,0],[44,150]),axis =0), [64,1,1]), x_filter2xi)
C_3fxi = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_cal_3xi,[0,0],[44,150]),axis =0), [64,1,1]), x_filter3xi)

f_mapx1xi = tf.matmul(C_1xi, tf.transpose(C_1fxi, [0,2,1]))
f_mapx2xi = tf.matmul(C_2xi, tf.transpose(C_2fxi, [0,2,1]))
f_mapx3xi = tf.matmul(C_3xi, tf.transpose(C_3fxi, [0,2,1]))

f_mapxxi = tf.nn.dropout(tf.nn.relu(f_mapx1xi +f_mapx2xi +f_mapx3xi), keep_prob = keep_prob)




###################################################################################################################################3


s7 = s6 + tf.shape(r_threexi)[0]
s8 = s7 + tf.shape(r_oney)[0]
s9 = s8 + tf.shape(r_twoy)[0]



X__1y = tf.slice(X_, [s7,0], [tf.shape(r_oney)[0], -1])
X__2y = tf.slice(X_, [ s8, 0   ],[ tf.shape(r_twoy)[0]  , -1])
X__3y = tf.slice(X_, [s9, 0], [tf.shape(r_threey)[0], -1])




X_10y  = 0.0001 * tf.transpose(X__1y,[1,0] )
X_20y  = 0.0001 * tf.transpose(X__2y,[1,0] )
X_30y  = 0.0001 * tf.transpose(X__3y,[1,0] )

X_cal_1y, p_1 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(44) + 1.0/4.0 * tf.matmul(tf.eye(44)-tf.matmul(x,X__1y),tf.matmul(3.0*tf.eye(44)-tf.matmul(x,X__1y),3.0*tf.eye(44)-tf.matmul(x,X__1y))),x),i+1)
    ,(X_10y, 0))

X_cal_2y, p_2 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(44) + 1.0/4.0 * tf.matmul(tf.eye(44)-tf.matmul(x,X__2y),tf.matmul(3.0*tf.eye(44)-tf.matmul(x,X__2y),3.0*tf.eye(44)-tf.matmul(x,X__2y))),x),i+1)
    ,(X_20y, 0))

X_cal_3y, p_3 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(44) + 1.0/4.0 * tf.matmul(tf.eye(44)-tf.matmul(x,X__3y),tf.matmul(3.0*tf.eye(44)-tf.matmul(x,X__3y),3.0*tf.eye(44)-tf.matmul(x,X__3y))),x),i+1)
    ,(X_30y, 0))



C_1y = tf.tile(tf.expand_dims(tf.matmul(X_cal_1y,r_oney), axis =0), [64,1,1])

C_2y = tf.tile(tf.expand_dims(tf.matmul(X_cal_2y, r_twoy), axis =0), [64,1,1])
C_3y = tf.tile(tf.expand_dims(tf.matmul(X_cal_3y,r_threey), axis =0), [64,1,1])

###############################################################################################

x_filter1y =  tf.get_variable("a_xfilter1y", [64,150,1])
x_filter2y =  tf.get_variable("a_xfilter2y", [64,150,1])
x_filter3y =  tf.get_variable("a_xfilter3y", [64,150,1])


C_1fy = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_cal_1y,[0,0],[44,150]),axis =0), [64,1,1]), x_filter1y)
C_2fy = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_cal_2y,[0,0],[44,150]),axis =0), [64,1,1]), x_filter2y)
C_3fy = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_cal_3y,[0,0],[44,150]),axis =0), [64,1,1]), x_filter3y)

f_mapx1y = tf.matmul(C_1y, tf.transpose(C_1fy, [0,2,1]))
f_mapx2y = tf.matmul(C_2y, tf.transpose(C_2fy, [0,2,1]))
f_mapx3y = tf.matmul(C_3y, tf.transpose(C_3fy, [0,2,1]))

f_mapxy = tf.nn.dropout(tf.nn.relu(f_mapx1y+ f_mapx2y+ f_mapx3y), keep_prob = keep_prob)


#################################################################################################################33



s10 = s9 + tf.shape(r_threey)[0]
s11 = s10 + tf.shape(r_oneyi)[0]
s12 = s11 + tf.shape(r_twoyi)[0]



X__1yi = tf.slice(X_, [s10,0], [tf.shape(r_oneyi)[0], -1])
X__2yi = tf.slice(X_, [ s11, 0   ],[ tf.shape(r_twoyi)[0]  , -1])
X__3yi = tf.slice(X_, [s12, 0], [tf.shape(r_threeyi)[0], -1])




X_10yi  = 0.0001 * tf.transpose(X__1yi,[1,0] )
X_20yi  = 0.0001 * tf.transpose(X__2yi,[1,0] )
X_30yi  = 0.0001 * tf.transpose(X__3yi,[1,0] )

X_cal_1yi, p_1 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(44) + 1.0/4.0 * tf.matmul(tf.eye(44)-tf.matmul(x,X__1yi),tf.matmul(3.0*tf.eye(44)-tf.matmul(x,X__1yi),3.0*tf.eye(44)-tf.matmul(x,X__1yi))),x),i+1)
    ,(X_10yi, 0))

#X_cal_1yi = tf.matmul( tf.eye(30) + 1.0/4.0 * tf.matmul(tf.eye(30)-tf.matmul(X_10yi,X__1yi),tf.matmul(3.0*tf.eye(30)-tf.matmul(X_10yi,X__1yi),3.0*tf.eye(30)-tf.matmul(X_10yi,X__1yi))),X_10yi)

X_cal_2yi, p_2 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(44) + 1.0/4.0 * tf.matmul(tf.eye(44)-tf.matmul(x,X__2yi),tf.matmul(3.0*tf.eye(44)-tf.matmul(x,X__2yi),3.0*tf.eye(44)-tf.matmul(x,X__2yi))),x),i+1)
    ,(X_20yi, 0))

#X_cal_2yi =  tf.matmul( tf.eye(30) + 1.0/4.0 * tf.matmul(tf.eye(30)-tf.matmul(X_20yi,X__2yi),tf.matmul(3.0*tf.eye(30)-tf.matmul(X_20yi,X__2yi),3.0*tf.eye(30)-tf.matmul(X_20yi,X__2yi)))


X_cal_3yi, p_3 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(44) + 1.0/4.0 * tf.matmul(tf.eye(44)-tf.matmul(x,X__3yi),tf.matmul(3.0*tf.eye(44)-tf.matmul(x,X__3yi),3.0*tf.eye(44)-tf.matmul(x,X__3yi))),x),i+1)
    ,(X_30yi, 0))

#X_cal_3yi = tf.matmul( tf.eye(30) + 1.0/4.0 * tf.matmul(tf.eye(30)-tf.matmul(X_30yi,X__3yi),tf.matmul(3.0*tf.eye(30)-tf.matmul(X_30yi,X__3yi),3.0*tf.eye(30)-tf.matmul(X_30yi,X__3yi)))

C_1yi = tf.tile(tf.expand_dims(tf.matmul(X_cal_1yi,r_oneyi), axis =0), [64,1,1])

C_2yi = tf.tile(tf.expand_dims(tf.matmul(X_cal_2yi, r_twoyi), axis =0), [64,1,1])
C_3yi = tf.tile(tf.expand_dims(tf.matmul(X_cal_3yi,r_threeyi), axis =0), [64,1,1])


##############################################################################


x_filter1yi =  tf.get_variable("a_xfilter1yi", [64,150,1])
x_filter2yi =  tf.get_variable("a_xfilter2yi", [64,150,1])
x_filter3yi =  tf.get_variable("a_xfilter3yi", [64, 150,1])


C_1fyi = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_cal_1yi,[0,0],[44,150]),axis =0), [64,1,1]), x_filter1yi)
C_2fyi = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_cal_2yi,[0,0],[44,150]),axis =0), [64,1,1]), x_filter2yi)
C_3fyi = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_cal_3yi,[0,0],[44,150]),axis =0), [64,1,1]), x_filter3yi)

f_mapx1yi = tf.matmul(C_1yi, tf.transpose(C_1fyi, [0,2,1]))
f_mapx2yi = tf.matmul(C_2yi, tf.transpose(C_2fyi, [0,2,1]))
f_mapx3yi = tf.matmul(C_3yi, tf.transpose(C_3fyi, [0,2,1]))

f_mapxyi = tf.nn.dropout(tf.nn.relu(f_mapx1yi+f_mapx2yi+f_mapx3yi),keep_prob = keep_prob)



print("##################!!!!!!!!!!!!!!!!!!!!!")
print(f_mapxyi)













"""

U_4 = tf.concat([u_0_4, u_1_4, u_2_4, u_3_4, u_4_4] ,  axis=1)
V_4 = tf.concat([v_0_4,v_1_4,v_2_4,v_3_4,v_4_4] ,  axis=1)

X_4 = tf.concat([U_4,V_4] ,  axis=1)

X_41 = tf.slice(X_4, [0,0], [tf.shape(r_one)[0], -1])
X_42 = tf.slice(X_4, [tf.shape(r_one)[0], 0], [tf.shape(r_two)[0], -1])
X_43 = tf.slice(X_4, [tf.shape(r_two)[0] + tf.shape(r_one)[0], 0], [tf.shape(r_three)[0], -1])




f_411 = safe_norm(tf.matmul(X_41,tf.slice(C_1,[20,0],[10,-1])),axis = None, keep_dims = True)
f_412 = safe_norm(tf.matmul(X_42,tf.slice(C_2,[20,0],[10,-1])),axis = None, keep_dims = True)
f_413 = safe_norm(tf.matmul(X_43,tf.slice(C_3,[20,0],[10,-1])),axis = None, keep_dims = True)


_, f_421 = tf.nn.moments(tf.matmul(X_41,tf.slice(C_1,[20,0],[10,-1])),axes = [0],  keep_dims = True)
_, f_422 = tf.nn.moments(tf.matmul(X_42,tf.slice(C_2,[20,0],[10,-1])),axes = [0],  keep_dims = True)
_, f_423 = tf.nn.moments(tf.matmul(X_43,tf.slice(C_3,[20,0],[10,-1])),axes = [0],  keep_dims = True)

   
#############################################################################################

U_3 = tf.concat([u_0_3, u_1_3, u_2_3, u_3_3] ,  axis=1)
V_3 = tf.concat([v_0_3,v_1_3,v_2_3,v_3_3] ,  axis=1)

X_3 = tf.concat([U_3,V_3] ,  axis=1)

X_31 = tf.slice(X_3, [0,0], [tf.shape(r_one)[0], -1])
X_32 = tf.slice(X_3, [tf.shape(r_one)[0], 0], [tf.shape(r_two)[0], -1])
X_33 = tf.slice(X_3, [tf.shape(r_two)[0] + tf.shape(r_one)[0], 0], [tf.shape(r_three)[0], -1])


f_311 = safe_norm(tf.matmul(X_31,tf.slice(C_1,[12,0],[8,-1])),axis = None, keep_dims = True)
f_312 = safe_norm(tf.matmul(X_32,tf.slice(C_2,[12,0],[8,-1])),axis = None, keep_dims = True)
f_313 = safe_norm(tf.matmul(X_33,tf.slice(C_3,[12,0],[8,-1])),axis = None, keep_dims = True)


_, f_321 = tf.nn.moments(tf.matmul(X_31,tf.slice(C_1,[12,0],[8,-1])),axes = [0],  keep_dims = True)
_, f_322 = tf.nn.moments(tf.matmul(X_32,tf.slice(C_2,[12,0],[8,-1])),axes = [0],  keep_dims = True)
_, f_323 = tf.nn.moments(tf.matmul(X_33,tf.slice(C_3,[12,0],[8,-1])),axes = [0],  keep_dims = True)


#################################################################################################################
U_2 = tf.concat([u_0_2, u_1_2, u_2_2] ,  axis=1)
V_2 = tf.concat([v_0_2,v_1_2,v_2_2] ,  axis=1)

X_2 = tf.concat([U_2,V_2] ,  axis=1)

X_21 = tf.slice(X_2, [0,0], [tf.shape(r_one)[0], -1])
X_22 = tf.slice(X_2, [tf.shape(r_one)[0], 0], [tf.shape(r_two)[0], -1])
X_23 = tf.slice(X_2, [tf.shape(r_two)[0] + tf.shape(r_one)[0], 0], [tf.shape(r_three)[0], -1])



f_211 = safe_norm(tf.matmul(X_21,tf.slice(C_1,[6,0],[6,-1])),axis = None, keep_dims = True)
f_212 = safe_norm(tf.matmul(X_22,tf.slice(C_2,[6,0],[6,-1])),axis = None, keep_dims = True)
f_213 = safe_norm(tf.matmul(X_23,tf.slice(C_3,[6,0],[6,-1])),axis = None, keep_dims = True)


_, f_221 = tf.nn.moments(tf.matmul(X_21,tf.slice(C_1,[6,0],[6,-1])),axes = [0],  keep_dims = True)
_, f_222 = tf.nn.moments(tf.matmul(X_22,tf.slice(C_2,[6,0],[6,-1])),axes = [0],  keep_dims = True)
_, f_223 = tf.nn.moments(tf.matmul(X_23,tf.slice(C_3,[6,0],[6,-1])),axes = [0],  keep_dims = True)

#########################################################################################################################

U_1 = tf.concat([u_0_1, u_1_1] ,  axis=1)
V_1 = tf.concat([v_0_1,v_1_1] ,  axis=1)

X_1 = tf.concat([U_1,V_1] ,  axis=1)

X_11 = tf.slice(X_1, [0,0], [tf.shape(r_one)[0], -1])
X_12 = tf.slice(X_1, [tf.shape(r_one)[0], 0], [tf.shape(r_two)[0], -1])
X_13 = tf.slice(X_1, [tf.shape(r_two)[0] + tf.shape(r_one)[0], 0], [tf.shape(r_three)[0], -1])

f_111 = safe_norm(tf.matmul(X_11,tf.slice(C_1,[2,0],[4,-1])),axis = None, keep_dims = True)
f_112 = safe_norm(tf.matmul(X_12,tf.slice(C_2,[2,0],[4,-1])),axis = None, keep_dims = True)
f_113 = safe_norm(tf.matmul(X_13,tf.slice(C_3,[2,0],[4,-1])),axis = None, keep_dims = True)


_, f_121 = tf.nn.moments(tf.matmul(X_11,tf.slice(C_1,[2,0],[4,-1])),axes = [0],  keep_dims = True)
_, f_122 = tf.nn.moments(tf.matmul(X_12,tf.slice(C_2,[2,0],[4,-1])),axes = [0],  keep_dims = True)
_, f_123 = tf.nn.moments(tf.matmul(X_13,tf.slice(C_3,[2,0],[4,-1])),axes = [0],  keep_dims = True)


##############################################################################################


U_0 = tf.concat([u_0_0] ,  axis=1)
V_0 = tf.concat([v_0_0] ,  axis=1)

X_0 = tf.concat([U_0,V_0] ,  axis=1)

X_01 = tf.slice(X_0, [0,0], [tf.shape(r_one)[0], -1])
X_02 = tf.slice(X_0, [tf.shape(r_one)[0], 0], [tf.shape(r_two)[0], -1])
X_03 = tf.slice(X_0, [tf.shape(r_two)[0] + tf.shape(r_one)[0] ,0] , [tf.shape(r_three)[0], -1])



f_011 = safe_norm(tf.matmul(X_01,tf.slice(C_1,[0,0],[2,-1])),axis = None, keep_dims = True)
f_012 = safe_norm(tf.matmul(X_02,tf.slice(C_2,[0,0],[2,-1])),axis = None, keep_dims = True)
f_013 = safe_norm(tf.matmul(X_03,tf.slice(C_3,[0,0],[2,-1])),axis = None, keep_dims = True)


_, f_021 = tf.nn.moments(tf.matmul(X_01,tf.slice(C_1,[0,0],[2,-1])),axes = [0],  keep_dims = True)
_, f_022 = tf.nn.moments(tf.matmul(X_02,tf.slice(C_2,[0,0],[2,-1])),axes = [0],  keep_dims = True)
_, f_023 = tf.nn.moments(tf.matmul(X_03,tf.slice(C_3,[0,0],[2,-1])),axes = [0],  keep_dims = True)

#####################################################################################################3

def radial_poly(rho, n, m):
	if n == 1 and m == 1:
		return tf.sqrt(rho+0.0001)
	if n == 2 and m == 1:
		return tf.pow(rho,1.5)
	if n == 3 and m == 1:
		return 3.0* tf.pow(rho, 1.5) - 2.0 * tf.sqrt(rho + 0.0001)
	if n == 3 and m == 3:
		return
	
	
	

l = tf.greater(theta_one[:,0],0)

theta_1 = tf.boolean_mask(theta_one, l)
phi_1 = tf.boolean_mask(phi_one, l)
r_1 = tf.boolean_mask(r_one, l)




r_pow = tf.transpose(tf.pow(r_1, 3), [1,0])
radial_real = tf.multiply(radial_poly(r_1,3,1), tf.cos(theta_1 * 3.0 + phi_1))
radial_imag = tf.multiply(radial_poly(r_1,3,1), tf.sin(theta_1 * 3.0 + phi_1))

real_term = tf.square(tf.matmul(r_pow,radial_imag))
imag_term = tf.square(tf.matmul(r_pow,radial_imag))

moment_31 = -0.277 * tf.sqrt(real_term + imag_term+ 0.0001)



radial_real2 = tf.multiply(radial_poly(r_1,2,1), tf.cos(theta_1 * 2.0 + phi_1))
radial_imag2 = tf.multiply(radial_poly(r_1,2,1), tf.sin(theta_1 * 2.0 + phi_1))

real_term2 = tf.square(tf.matmul(r_pow,radial_imag2))
imag_term2 = tf.square(tf.matmul(r_pow,radial_imag2))

moment_21 = 0.207 * tf.sqrt(real_term2 + imag_term2)

C_1_softmax = tf.nn.softmax(C_1, axis = 0)
C_2_softmax = tf.nn.softmax(C_2, axis = 0)
C_3_softmax = tf.nn.softmax(C_3, axis = 0)

f_1_softmax = tf.concat([ f_011, f_021, f_111, f_121,f_211, f_221, f_311, f_321,f_411, f_421 ], axis =0)
f_2_softmax = tf.concat([  f_012, f_022, f_112, f_122, f_212, f_222, f_312, f_322, f_412, f_422  ], axis = 0)
f_3_softmax = tf.concat([f_013, f_023, f_113, f_123, f_213, f_223,  f_313, f_323, f_413, f_423 ], axis = 0)

m1_softmax =  tf.concat([ moment_31, moment_21  ], axis = 0)
m2_softmax =  tf.concat([  moment_31, moment_21  ], axis = 0)
m3_softmax =  tf.concat([  moment_31, moment_21  ], axis = 0) 



#B_1 = tf.concat([C_1, f_011, f_021, f_111, f_121,f_211, f_221, f_311, f_321,f_411, f_421,moment_31,moment_21  ], axis =0)
#B_2 = tf.concat([C_2,  f_012, f_022, f_112, f_122, f_212, f_222, f_312, f_322, f_412, f_422, moment_31, moment_21   ], axis = 0)
#B_3 = tf.concat([C_3, f_013, f_023, f_113, f_123, f_213, f_223,  f_313, f_323, f_413, f_423, moment_31, moment_21  ], axis = 0)

B_1 =  tf.concat([C_1,C_1i, C_1x, C_1xi, C_1y, C_1yi, f_1_softmax,m1_softmax  ], axis =0)
B_2 =  tf.concat([C_2, C_2i,  C_2x, C_2xi, C_2y, C_2yi, f_2_softmax,m2_softmax  ], axis =0)
B_3 =  tf.concat([C_3, C_3i,  C_3x, C_3xi, C_3y, C_3yi, f_3_softmax,m3_softmax  ], axis =0)
"""

z_reshape = tf.expand_dims(tf.reshape(f_mapz, [64,1,1936]), axis = 1)
#z_reshape = tf.reshape(f_mapz, [1,8,8,1936])
zi_reshape = tf.expand_dims(tf.reshape(f_mapzi, [64,1,1936]), axis = 1)
#zi_reshape = tf.reshape(f_mapzi, [1,8,8,1936])

x_reshape = tf.expand_dims(tf.reshape(f_mapx, [64,1,1936]), axis = 1)
#x_reshape = tf.reshape(f_mapx, [1,8,8,1936])
xi_reshape = tf.expand_dims(tf.reshape(f_mapxxi, [64,1,1936]), axis = 1)
#xi_reshape = tf.reshape(f_mapxxi, [1,8,8,1936])

y_reshape = tf.expand_dims(tf.reshape(f_mapxy, [64,1,1936]), axis = 1)
#y_reshape = tf.reshape(f_mapxy, [1,8,8,1936])
yi_reshape = tf.expand_dims(tf.reshape(f_mapxyi, [64,1,1936]), axis = 1)
#yi_reshape = tf.reshape(f_mapxyi, [1,8,8,1936])

#z_compact = tf.reshape(compact_bilinear_pooling_layer(z_reshape, zi_reshape, 1000 , sum_pool=False, sequential=False),[1,8,8,1000])
#x_compact =  tf.reshape(compact_bilinear_pooling_layer(x_reshape, xi_reshape, 1000 , sum_pool=False, sequential=False),[1,8,8,1000])
#y_compact =  tf.reshape(compact_bilinear_pooling_layer(y_reshape, yi_reshape, 1000 , sum_pool=False, sequential=False),[1,8,8,1000])


#compact1 = tf.reshape(compact_bilinear_pooling_layer(z_compact, x_compact, 5000 , sum_pool=False, sequential=False),[1,8,8,5000])



#feature_map1 = tf.nn.max_pool(tf.reshape(tf.concat([z_reshape, x_reshape, y_reshape], axis = 1), [1,44,44*3,64]),ksize = (1,8,8,1), strides = (1,6,6,1), padding = "SAME")
feature_map1 = tf.reshape(tf.concat([z_reshape, x_reshape, y_reshape], axis = 1), [1,8,8,1936*3])

feature_map2 = tf.reshape(tf.concat([zi_reshape, xi_reshape, yi_reshape], axis = 1), [1,8,8,1936*3])

#feature_map2 = tf.nn.max_pool(tf.reshape(tf.concat([zi_reshape, xi_reshape, yi_reshape], axis = 1), [1,44,44*3,64]),ksize = (1,8,8,1), strides = (1,6,6,1), padding = "SAME")


B = tf.concat([f_mapz,f_mapzi, f_mapx, f_mapxxi,f_mapxy, f_mapxyi ], axis =0)
top = compact_bilinear_pooling_layer(feature_map1, feature_map2, 10000 , sum_pool=False, sequential=False)
#top = tf.nn.relu(compact_bilinear_pooling_layer(compact1, y_compact, 10000 , sum_pool=False, sequential=False))
top_descriptor = compact_bilinear_pooling_layer(feature_map1, feature_map2, 10000 , sum_pool=True, sequential=False)

#B_1 = tf.concat([C_1,moment_31,moment_21 ], axis =0)
#B_2 = tf.concat([C_2, moment_31, moment_21  ], axis = 0)
#B_3 = tf.concat([C_3, moment_31, moment_21 ], axis = 0)






#test10 = tf.norm(tf.matmul(X_1,B_1))
#test_11 = tf.reduce_max(tf.matmul(X_1,B_1))

#  = tf.concat([B_1, B_2, B_3], axis = 0)







# print(B)
#estimate_1 = tf.matmul(tf.transpose(u, perm=[0, 2, 1]),r_temp)
init_sigma = 0.01

# A_init = tf.random_normal(
# 		  shape=(3, 10,1),
# 		  stddev=init_sigma, dtype=tf.float32, name="cn_W_init")
# A = tf.Variable(A_init, name="cn_W")


#layer_1 = tf.concat([tf.reshape(B, [1,90]),f_11,f_12,f_13], axis = 1)
layer_1 = tf.reshape(B, [1,126])

#layer_1 = tf.concat([f_11,f_12,f_13], axis = 1)


layer_2 = tf.layers.dense(layer_1, 50)

layer_2_output = layer_2

layer_3 = tf.layers.dense(layer_2_output, 50)

layer_3_output = layer_3

layer_4 = tf.layers.dense(layer_3_output, 50)

layer_4_output =  layer_4

layer_5 = tf.layers.dense(layer_4_output, 50)

layer_5_output =  layer_5

layer_6 = tf.layers.dense(layer_5_output, 50)

layer_6_output =  layer_6

layer_7 = tf.layers.dense(layer_6_output, 50)

layer_7_output =  layer_7

layer_8 = tf.layers.dense(layer_7_output, 20)

layer_8_output =  layer_8

layer_9 = tf.layers.dense(layer_8_output, 10)

layer_9_output =  tf.nn.relu(layer_9)

  

caps1_raw_temp = tf.reshape(top, [-1,10000,64],
                       name="caps1_raw")

#caps1_raw =  tf.nn.l2_normalize(caps1_raw_temp,axis = 2)

caps1_raw =1.0 *  tf.div(
   tf.subtract(
      caps1_raw_temp,
      tf.reduce_min(caps1_raw_temp,axis = 2, keepdims = True)
   ),tf.maximum(
   tf.subtract(
      tf.reduce_max(caps1_raw_temp,axis = 2, keepdims = True),
      tf.reduce_min(caps1_raw_temp,axis = 2, keepdims = True)
   ),0.0000001)
)

#caps1_raw_temp_ = tf.reshape(caps1_raw, [-1, 192,3],
 #                      name="caps1_raw_")


def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keepdims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector
      
#caps1_output = squash(caps1_raw, name="caps1_output")
caps1_output = caps1_raw_temp

caps2_n_caps = 10
caps2_n_dims = 8


W_init = tf.random_normal(
    shape=(1,10000, 10, caps2_n_dims, 64),
    stddev=init_sigma, dtype=tf.float32, name="a_W_init")
W = tf.Variable(W_init, name="a_W")



batch_size = 1
W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="a_W_tiled")


caps1_output_expanded = tf.expand_dims(caps1_output, -1,
                                       name="caps1_output_expanded")
caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2,
                                   name="caps1_output_tile")
caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, caps2_n_caps, 1, 1],
                             name="caps1_output_tiled")


caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled,
                            name="caps2_predicted")
 
#########################################################

raw_weights = tf.zeros([batch_size,10000, caps2_n_caps, 1, 1],
                       dtype=np.float32, name="raw_weights")

# caps1_output_expanded = tf.expand_dims(caps1_output, -1,
#                                        name="caps1_output_expanded")

routing_weights = tf.nn.softmax(raw_weights, dim=2, name="routing_weights")

weighted_predictions = tf.multiply(routing_weights, caps2_predicted,
                                   name="weighted_predictions")
weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True,
                             name="weighted_sum")


caps2_output_round_1 = tf.nn.dropout(squash(weighted_sum, axis=-2,
                             name="caps2_output_round_1"), keep_prob = keep_prob)

#caps2_output_round_1 = weighted_sum


caps2_output_round_1_tiled = tf.tile(
    caps2_output_round_1, [1, 10000, 1, 1, 1],
    name="caps2_output_round_1_tiled")


agreement = tf.matmul(caps2_predicted, caps2_output_round_1_tiled,
                      transpose_a=True, name="agreement")

raw_weights_round_2 = tf.add(raw_weights, agreement,
                             name="raw_weights_round_2")

routing_weights_round_2 = tf.nn.softmax(raw_weights_round_2,
                                        dim=2,
                                        name="routing_weights_round_2")
weighted_predictions_round_2 = tf.multiply(routing_weights_round_2,
                                           caps2_predicted,
                                           name="weighted_predictions_round_2")
weighted_sum_round_2 = tf.reduce_sum(weighted_predictions_round_2,
                                     axis=1, keep_dims=True,
                                     name="weighted_sum_round_2")
caps2_output_round_2 = tf.nn.dropout(squash(weighted_sum_round_2,
                              axis=-2,
                              name="caps2_output_round_2"), keep_prob = keep_prob)


#caps2_output = caps2_output_round_2
#caps2_output_round_2 =weighted_sum_round_2

caps2_output_round_2_tiled = tf.tile(
    caps2_output_round_2, [1, 10000, 1, 1, 1],
    name="caps2_output_round_2_tiled")


agreement2 = tf.matmul(caps2_predicted, caps2_output_round_2_tiled,
                      transpose_a=True, name="agreement2")

raw_weights_round_3 = tf.add(raw_weights_round_2, agreement2,
                             name="raw_weights_round_3")

routing_weights_round_3 = tf.nn.softmax(raw_weights_round_3,
                                        dim=2,
                                        name="routing_weights_round_2")
weighted_predictions_round_3 = tf.multiply(routing_weights_round_3,
                                           caps2_predicted,
                                           name="weighted_predictions_round_2")
weighted_sum_round_3 = tf.reduce_sum(weighted_predictions_round_3,
                                     axis=1, keep_dims=True,
                                     name="weighted_sum_round_2")
caps2_output_round_3 = squash(weighted_sum_round_3,
                              axis=-2,
                              name="caps2_output_round_2")


caps2_output_ =weighted_sum_round_3



print("weighted")
print(weighted_predictions_round_3)


#u_initial = safe_norm(weighted_predictions_round_3, axis=3,keep_dims = True, name="u_initial")

#u_initial = tf.nn.softmax(weighted_predictions_round_3, axis = 2)

#u_initial_ = tf.transpose(u_initial, [0,1,3,2,4])

#u_conv_input = tf.reshape(u_initial,[1,384,10])

#label_function = tf.get_variable("a_label", [1, 10, 10], initializer = tf.random_normal_initializer(seed = 0.1))

#u_label_compat = tf.nn.conv1d(u_conv_input, label_function, stride = 1, padding = "VALID")



#u_initial_reshape =  tf.reshape(u_label_compat, [1,192,2,10,1])

#u_init_transposed =  tf.transpose(u_initial_reshape, [0,1,3,2,4])

#u_added = tf.subtract(weighted_predictions_round_3,u_init_transposed)

#Q_round_1 = tf.nn.softmax(u_added, axis = 2)

#########################################################################
#u_initial2 = Q_round_1

#u_initial_2 = tf.transpose(u_initial2, [0,1,3,2,4])

#u_conv_input2 = tf.reshape(u_initial2,[1,384,10])


#u_label_compat2 = tf.nn.conv1d(u_conv_input2, label_function, stride = 1, padding = "VALID")



#u_initial_reshape2 =  tf.reshape(u_label_compat2, [1,192,2,10,1])

#u_init_transposed2 =  tf.transpose(u_initial_reshape2, [0,1,3,2,4])

#u_added2 = tf.subtract(weighted_predictions_round_3, u_init_transposed2)

#Q_round_2 = tf.nn.softmax(u_added2, axis = 2)

#############################################################################3


#u_initial3 = Q_round_2

#u_initial_3 = tf.transpose(u_initial3, [0,1,3,2,4])

#u_conv_input3 = tf.reshape(u_initial3,[1,384,10])


#u_label_compat3 = tf.nn.conv1d(u_conv_input3, label_function, stride = 1, padding = "VALID")



#u_initial_reshape3 =  tf.reshape(u_label_compat3, [1,192,2,10,1])

#u_init_transposed3 =  tf.transpose(u_initial_reshape3, [0,1,3,2,4])

#u_added3 = tf.subtract(weighted_predictions_round_3, u_init_transposed3)

#Q_round_3 = tf.nn.softmax(u_added3, axis = 2)



#print(u_label_compat)


#weighted_sum_Q = tf.reduce_sum(Q_round_3,
#                                     axis=1, keep_dims=True,
#                                     name="weightedQ_sum_round_2")
#caps2_output = weighted_sum_Q

#caps2_output__ =weighted_sum_round_3


caps2_output =1.0 *  tf.div(
   tf.subtract(
      weighted_sum_round_3 ,
      tf.reduce_min(weighted_sum_round_3 , keepdims = True)
   ),
   tf.subtract(
      tf.reduce_max(weighted_sum_round_3 , keepdims = True),
      tf.reduce_min(weighted_sum_round_3 , keepdims = True)
   )
)

"""
caps2_output =2.0 * tf.div(
   tf.subtract(
      weighted_sum_round_3 ,
      tf.reduce_min(weighted_sum_round_3 , keepdims = True)
   ),
   tf.subtract(
      tf.reduce_max(weighted_sum_round_3 , keepdims = True),
      tf.reduce_min(weighted_sum_round_3 , keepdims = True)
   )
) - 
"""













      
y_proba = safe_norm(caps2_output, axis=-2, name="y_proba")
y_proba_argmax = tf.argmax(y_proba, axis=2, name="y_proba")
y_pred = tf.squeeze(y_proba_argmax, axis=[1,2], name="y_pred")


#y_pred = tf.argmax(layer_9_output, axis=1)


y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")

m_plus = 0.9
m_minus = 0.1
lambda_ = 0.5

T = tf.one_hot(y, depth=caps2_n_caps, name="T")

caps2_output_norm = tf.reshape(safe_norm(caps2_output, axis=-2, keep_dims=True,
                             name="caps2_output_norm"), [-1,10])

#caps2_output_norm = safe_norm(caps2_output, axis=-2, keep_dims=True,
#                              name="caps2_output_norm")

present_error_raw = tf.square(tf.maximum(0., m_plus - caps2_output_norm),
                              name="present_error_raw")
present_error = tf.reshape(present_error_raw, shape=(-1, caps2_n_caps),
                           name="present_error")
absent_error_raw = tf.square(tf.maximum(0., caps2_output_norm - m_minus),
                             name="absent_error_raw")
absent_error = tf.reshape(absent_error_raw, shape=(-1,  caps2_n_caps),
                          name="absent_error")

L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error,
           name="L")


loss = tf.losses.softmax_cross_entropy(T, caps2_output_norm)
#loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")


trainable_vars = tf.trainable_variables()

rot_vars = [var for var in trainable_vars if 'a_' in var.name]
caps_vars =  [var for var in trainable_vars if 'a_' not  in var.name]

#names = [n.name for n in tf.get_default_graph().as_graph_def().node]


batch = tf.Variable(0, trainable = False)

learning_rate = tf.train.exponential_decay(
  0.1,                # Base learning rate.
  batch,  # Current index into the dataset.
  1500,          # Decay step.
  0.95,                # Decay rate.
  staircase=True)

optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
grads = optimizer.compute_gradients(loss, var_list = rot_vars)
training_op = optimizer.minimize(loss, name="training_op", global_step=batch) #,  var_list = caps_vars)

assign_op = batch.assign(1)

#accum_tvars = [tf.Variable(tf.zeros_like(t_var.initialized_value()),trainable=False) for t_var in rot_vars] 

#zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_tvars]

#batch_grads_vars = optimizer.compute_gradients(loss, rot_vars)
#accum_ops = [accum_tvars[i].assign_add(batch_grad_var[0]) for i, batch_grad_var in enumerate(batch_grads_vars)]

#grad_mean = [tv/20.0  for tv in accum_tvars]

#grad_mean = tf.reduce_mean(expanded_grad, axis = 0)


#train_step = optimizer.apply_gradients([(grad_mean[i], batch_grad_var[1]) for i, batch_grad_var in enumerate(batch_grads_vars)])

def tf_count(t, val):
    elements_equal_to_value = tf.equal(t, val)
    as_ints = tf.cast(elements_equal_to_value, tf.int32)
    count = tf.reduce_sum(as_ints)
    return count



optimizer_2 =  tf.train.GradientDescentOptimizer(learning_rate=0.1)
#grads_2 = optimizer_2.compute_gradients(rot_loss, var_list = rot_vars)
#training_op_2 = optimizer_2.minimize(rot_loss, name="training_op_2", var_list = rot_vars)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

print('before_sess')
sess = tf.Session()
print('after_sess')

def condition(x, i, index, axis):
    return tf.logical_and(tf.equal(x[0,i,0], index), tf.equal(x[0,i,2], axis))
  
  


correct = tf.equal(y, y_pred, name="correct")
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
  

# block_hankel = tf.slice(calibrated_points_one_corrected_shape, [0, 0, 0], [-1,10,-1])
sess.run(tf.global_variables_initializer())



def read_datapoint(data, filename):
  points = np.array([[0, 0, 0]])
  
  for line in data:
    if 'OFF' != line.strip() and len([float(s) for s in line.strip().split(' ')]) == 3: 
        if 'bathtub' in filename:
           y = [0]
        if 'bed' in filename:
           y=[1]
	if 'chair' in filename:
	   y=[2]
	if 'desk' in filename:
           y=[3]
	if 'dresser' in filename:
           y=[4]
	if 'monitor' in filename:
           y=[5]
	if 'night_stand' in filename:
           y=[6]
	if 'sofa' in filename:
           y=[7]
	if "table" in filename:
           y=[8]
	if 'toilet' in filename:
           y=[9]	
        points_list = [float(s) for s in line.strip().split(' ')]
        points = np.append(points,np.expand_dims(np.array(points_list), axis=0), axis = 0)
  
 
  return points[1:], y

saver = tf.train.Saver()

training_files = '/home/ram095/sameera/3d_obj/training_files/'
testing_files = '/home/ram095/sameera/3d_obj/testing_files/'

saver.restore(sess, "./model.ckpt")
loss_train_vals = []
loss_vals = []
acc_vals = []
file_names = []
main_itr = 0
i = 0
grad_vals = []
gradients = np.array([])
#np.set_printoptions(threshold=np.nan)
file_list = glob.glob(os.path.join(training_files, '*.off'))
pre_acc_val = 0.83
#sess.run(assign_op)
#sess.run(zero_ops)
file_names = open('file_names_train.txt', 'w')
top_vals = np.empty((0,10000))
for j in range(1):
	loss_train_vals = []
	random.shuffle(file_list)

	for filename in file_list:
		
	#	main_itr = main_itr+1
	#	print(filename)
		f = open(filename, 'r')
	
	
		points_raw, y_annot = read_datapoint(f, filename)
		print(y_annot)
		points =np.vstack(set(map(tuple, points_raw))) 
	        if points_raw.size > 1000 and points_raw.size <  1000000:	
			main_itr = main_itr+1
	                print(filename)
			print(main_itr)
			top_val = sess.run([top_descriptor], feed_dict = {y:y_annot, raw_points_init:points_raw, keep_prob :1.0})
			file_names.write("%s\n" % filename)
			print(j)
			top_vals = np.append(top_vals, np.reshape(top_val,(1,10000)), axis=0)	
	np.savetxt('top_vals_train.txt', top_vals)
	
	for filename in glob.glob(os.path.join(testing_files, '*.off')):
                		
            	f = open(filename, 'r')
               	print(filename)
               	file_names.append(filename)
               	points_raw, y_annot = read_datapoint(f, filename)
               	points =np.vstack(set(map(tuple, points_raw)))
		idx = np.random.randint(int(points.size/3.0), size= int(points.size/3.0 * 0.95)) 
		loss_val, acc_val = sess.run([loss, accuracy], feed_dict = {y:y_annot, raw_points_init:points_raw,  keep_prob :1.0})
                loss_vals.append(loss_val)
                acc_vals.append(acc_val)
		print("validation")
                print(acc_val)		
		acc_val = np.mean(acc_vals)
          	print(acc_val)
	loss_vals = []
	acc_vals = []      

