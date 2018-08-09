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

#centered_points = tf.subtract(raw_points_init, tf.reduce_mean(raw_points_init, axis = 0, keepdims = True))
#centered_points = tf.subtract(raw_points_init, (tf.reduce_max(raw_points_init, axis = 0, keepdims = True)+tf.reduce_min(raw_points_init, axis = 0, keepdims = True))/2.0)
centered_points = raw_points_init 

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
rot_angles = tf.constant([[math.pi/4.0, math.pi/4.0, math.pi/4.0, 0.0]])

_rot_angles = tf.nn.l2_normalize(rot_angles_temp_) 

#rot_angles_ = tf.reshape(rot_angles, [3,3,3])

# rotation_matrix_one = tf.squeeze(tf.slice(rot_angles_, [0,0,0], [1,-1,-1]),squeeze_dims=[0])
# rotation_matrix_two =  tf.squeeze(tf.slice(rot_angles_, [1,0,0], [1,-1,-1]),squeeze_dims=[0])
# rotation_matrix_three =  tf.squeeze(tf.slice(rot_angles_, [2,0,0], [1,-1,-1]),squeeze_dims=[0])

#rot_angles = tf.constant([[22.0/28.0,22.0/14.0,0.0]]) 

rotation_matrix_one = tf.concat([tf.constant([[1.0, 0.0, 0.0]]), [[0.0, tf.cos(rot_angles[0,0]), -tf.sin(rot_angles[0,0])]], [[0.0, tf.sin(rot_angles[0,0]), tf.cos(rot_angles[0,0])]]], axis = 0)
rotation_matrix_two = tf.concat([[[tf.cos(rot_angles[0,1]), 0.0, tf.sin(rot_angles[0,1])]], [[0.0, 1.0, 0.0]], [[-tf.sin(rot_angles[0,1]), 0.0,tf.cos(rot_angles[0,1]) ]]], axis = 0)
rotation_matrix_three = tf.concat([[[tf.cos(rot_angles[0,2]), -tf.sin(rot_angles[0,2]),0.0 ]], [[tf.sin(rot_angles[0,2]), tf.cos(rot_angles[0,2]), 0.0]], [[0.0, 0.0,1.0 ]]], axis = 0)
rotation_matrix_four = tf.concat([tf.constant([[1.0, 0.0, 0.0]]), [[0.0, tf.cos(rot_angles[0,3]), -tf.sin(rot_angles[0,3])]], [[0.0, tf.sin(rot_angles[0,3]), tf.cos(rot_angles[0,3])]]], axis = 0)


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

trasformed_points_four = tf.matmul(centered_points_expanded_, rotation_matrix_four, name="trans_point_four")
trasformed_points_four_reshaped = tf.reshape(trasformed_points_four, [-1, point_count, 3], name = "trans_point_four_reshape")

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



trasformed_points_four_reshaped_ = tf.reshape(trasformed_points_four_reshaped, [-1, 3])
calibrated_points_four = trasformed_points_four_reshaped_

calibrated_points_one_corrected_shape = tf.reshape(calibrated_points_one, [-1, point_count, 3])


#centered_calib_points_one_temp_  = tf.subtract(calibrated_points_one,tf.reduce_mean(calibrated_points_one,axis=0,keep_dims=True))
#centered_calib_points_two_temp_ = tf.subtract(calibrated_points_two,tf.reduce_mean(calibrated_points_two,axis=0,keep_dims=True))
#centered_calib_points_three_temp_  = tf.subtract(calibrated_points_three,tf.reduce_mean(calibrated_points_three,axis=0,keep_dims=True))

centered_calib_points_one_temp_ = calibrated_points_one
centered_calib_points_two_temp_ = calibrated_points_two
centered_calib_points_three_temp_ = calibrated_points_three
centered_calib_points_four_temp_ = calibrated_points_four

b1 = tf.add(tf.slice(centered_calib_points_one_temp_ , [0, 0], [-1, 1]) * 100000, tf.slice(centered_calib_points_one_temp_ , [0, 1], [-1, 1]))

reordered1 = tf.gather(centered_calib_points_one_temp_ , tf.nn.top_k(b1[:, 0], k=tf.shape(centered_calib_points_one_temp_ )[0], sorted=True).indices)
centered_calib_points_one_temp = tf.reverse(reordered1, axis=[0])

b2 = tf.add(tf.slice(centered_calib_points_two_temp_ , [0, 0], [-1, 1]) * 100000, tf.slice(centered_calib_points_two_temp_ , [0, 1], [-1, 1]))

reordered2 = tf.gather(centered_calib_points_two_temp_ , tf.nn.top_k(b2[:, 0], k=tf.shape(centered_calib_points_two_temp_ )[0], sorted=True).indices)
centered_calib_points_two_temp = tf.reverse(reordered2, axis=[0])


b3 = tf.add(tf.slice(centered_calib_points_three_temp_ , [0, 0], [-1, 1]) * 100000, tf.slice(centered_calib_points_three_temp_ , [0, 1], [-1, 1]))

reordered3 = tf.gather(centered_calib_points_three_temp_ , tf.nn.top_k(b3[:, 0], k=tf.shape(centered_calib_points_three_temp_ )[0], sorted=True).indices)
centered_calib_points_three_temp = tf.reverse(reordered3, axis=[0])


b4 = tf.add(tf.slice(centered_calib_points_four_temp_ , [0, 0], [-1, 1]) * 100000, tf.slice(centered_calib_points_four_temp_ , [0, 1], [-1, 1]))

reordered4 = tf.gather(centered_calib_points_four_temp_ , tf.nn.top_k(b3[:, 0], k=tf.shape(centered_calib_points_four_temp_ )[0], sorted=True).indices)
centered_calib_points_four_temp = tf.reverse(reordered4, axis=[0])




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



indices_four_x_temp = tf.nn.top_k(centered_calib_points_four_temp[:,2], k=tf.shape(centered_calib_points_four_temp)[0]).indices
reordered_points_four_x_temp = tf.gather(centered_calib_points_four_temp, indices_four_x_temp, axis=0)


centered_calib_points_four_t  = tf.slice(reordered_points_four_x_temp, [0, 0], [tf.shape(reordered_points_four_x_temp)[0]/2, -1])

#indices_one_x_tempi = tf.nn.top_k(centered_calib_points_one_temp[:,2], k=tf.shape(centered_calib_points_one_temp[0]).indices
#reordered_points_one_x_tempi = tf.gather(points_from_side_one_temp, indices_one_x_temp, axis=0)


centered_calib_points_one_t_i  = tf.slice(reordered_points_one_x_temp, [tf.shape(reordered_points_one_x_temp)[0]/2 , 0], [tf.shape(reordered_points_one_x_temp)[0]/2, -1])


centered_calib_points_two_t_i  = tf.slice(reordered_points_two_x_temp, [tf.shape(reordered_points_two_x_temp)[0]/2, 0], [tf.shape(reordered_points_two_x_temp)[0]/2, -1])

centered_calib_points_three_t_i  = tf.slice(reordered_points_three_x_temp, [tf.shape(reordered_points_three_x_temp)[0]/2, 0], [tf.shape(reordered_points_three_x_temp)[0]/2, -1])


centered_calib_points_four_t_i  = tf.slice(reordered_points_four_x_temp, [tf.shape(reordered_points_four_x_temp)[0]/2, 0], [tf.shape(reordered_points_four_x_temp)[0]/2, -1])


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


indices_four_xx_temp = tf.nn.top_k(centered_calib_points_four_temp[:,0], k=tf.shape(centered_calib_points_four_temp)[0]).indices
reordered_points_four_xx_temp = tf.gather(centered_calib_points_four_temp, indices_four_xx_temp, axis=0)

#centered_calib_points_three_tx  = tf.boolean_mask(centered_calib_points_three_temp, mask_threex)
centered_calib_points_four_tx  = tf.slice(reordered_points_four_xx_temp, [0, 0], [tf.shape(reordered_points_four_xx_temp)[0]/2, -1])




centered_calib_points_one_txi  = tf.slice(reordered_points_one_xx_temp, [tf.shape(reordered_points_one_xx_temp)[0]/2 , 0], [tf.shape(reordered_points_one_xx_temp)[0]/2, -1])


centered_calib_points_two_txi  = tf.slice(reordered_points_two_xx_temp, [tf.shape(reordered_points_two_xx_temp)[0]/2, 0], [tf.shape(reordered_points_two_xx_temp)[0]/2, -1])

centered_calib_points_three_txi  = tf.slice(reordered_points_three_xx_temp, [tf.shape(reordered_points_three_xx_temp)[0]/2, 0], [tf.shape(reordered_points_three_xx_temp)[0]/2, -1])


centered_calib_points_four_txi  = tf.slice(reordered_points_four_xx_temp, [tf.shape(reordered_points_four_xx_temp)[0]/2, 0], [tf.shape(reordered_points_four_xx_temp)[0]/2, -1])

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



indices_four_y_temp = tf.nn.top_k(centered_calib_points_four_temp[:,1], k=tf.shape(centered_calib_points_four_temp)[0]).indices
reordered_points_four_y_temp = tf.gather(centered_calib_points_four_temp, indices_four_y_temp, axis=0)

centered_calib_points_four_ty  = tf.slice(reordered_points_four_y_temp, [0, 0], [tf.shape(reordered_points_four_y_temp)[0]/2, -1])




#mask_oneyi  = tf.less(centered_calib_points_one_temp[:,1],0)
centered_calib_points_one_tyi  = tf.slice(reordered_points_one_y_temp, [tf.shape(reordered_points_one_y_temp)[0]/2 , 0], [tf.shape(reordered_points_one_y_temp)[0]/2, -1])

centered_calib_points_two_tyi  = tf.slice(reordered_points_two_y_temp, [tf.shape(reordered_points_two_y_temp)[0]/2 , 0], [tf.shape(reordered_points_two_y_temp)[0]/2, -1])


centered_calib_points_three_tyi  = tf.slice(reordered_points_three_y_temp, [tf.shape(reordered_points_three_y_temp)[0]/2 , 0], [tf.shape(reordered_points_three_y_temp)[0]/2, -1])



centered_calib_points_four_tyi  = tf.slice(reordered_points_four_y_temp, [tf.shape(reordered_points_four_y_temp)[0]/2 , 0], [tf.shape(reordered_points_four_y_temp)[0]/2, -1])






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


r_four_temp = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_four_t), axis=1, keepdims = True))
r_four = tf.divide(r_four_temp,tf.reduce_max(r_four_temp, axis = 0, keep_dims = True))

#r_three = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_three), axis=1, keepdims = True))
theta_four = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_four_t [:,2],1), tf.maximum(r_four_temp,0.001)),0.99),-0.99))
phi_four  = tf.atan2(tf.expand_dims(centered_calib_points_four_t [:,1],1),tf.expand_dims(centered_calib_points_four_t [:,0],1))


rp = tf.concat([r_one, r_two, r_three,r_four], axis = 0)
thetap = tf.concat([theta_one, theta_two, theta_three,theta_four], axis = 0)
phip = tf.concat([phi_one, phi_two, phi_three, phi_four], axis = 0)


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


r_four_temp_i = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_four_t_i), axis=1, keepdims = True))
r_four_i = tf.divide(r_four_temp_i,tf.reduce_max(r_four_temp_i, axis = 0, keep_dims = True))

#r_three = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_three), axis=1, keepdims = True))
theta_four_i = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_four_t_i [:,2],1), tf.maximum(r_four_temp_i,0.001)),0.99),-0.99))
phi_four_i  = tf.atan2(tf.expand_dims(centered_calib_points_four_t_i [:,1],1),tf.expand_dims(centered_calib_points_four_t_i [:,0],1))

r_i = tf.concat([r_one_i, r_two_i, r_three_i, r_four_i], axis = 0)
theta_i = tf.concat([theta_one_i, theta_two_i, theta_three_i, theta_four_i], axis = 0)
phi_i = tf.concat([phi_one_i, phi_two_i, phi_three_i, phi_four_i], axis = 0)


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

r_four_tempx = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_four_tx), axis=1, keepdims = True))
r_fourx = tf.divide(r_four_tempx,tf.reduce_max(r_four_tempx, axis = 0, keep_dims = True))

#r_three = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_three), axis=1, keepdims = True))
theta_fourx = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_four_tx [:,2],1), tf.maximum(r_four_tempx,0.001)),0.99),-0.99))
phi_fourx  = tf.atan2(tf.expand_dims(centered_calib_points_four_tx [:,1],1),tf.expand_dims(centered_calib_points_four_tx [:,0],1))


rx = tf.concat([r_onex, r_twox, r_threex, r_fourx], axis = 0)
thetax = tf.concat([theta_onex, theta_twox, theta_threex, theta_fourx], axis = 0)
phix = tf.concat([phi_onex, phi_twox, phi_threex, phi_fourx], axis = 0)


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


r_four_tempxi = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_four_txi), axis=1, keepdims = True))
r_fourxi = tf.divide(r_four_tempxi,tf.reduce_max(r_four_tempxi, axis = 0, keep_dims = True))

#r_three = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_three), axis=1, keepdims = True))
theta_fourxi = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_four_txi [:,2],1), tf.maximum(r_four_tempxi,0.001)),0.99),-0.99))
phi_fourxi  = tf.atan2(tf.expand_dims(centered_calib_points_four_txi [:,1],1),tf.expand_dims(centered_calib_points_four_txi [:,0],1))

rxi = tf.concat([r_onexi, r_twoxi, r_threexi, r_fourxi], axis = 0)
thetaxi = tf.concat([theta_onexi, theta_twoxi, theta_threexi, theta_fourxi], axis = 0)
phixi = tf.concat([phi_onexi, phi_twoxi, phi_threexi, phi_fourxi], axis = 0)


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

r_four_tempy = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_four_ty), axis=1, keepdims = True))
r_foury = tf.divide(r_four_tempy,tf.reduce_max(r_four_tempy, axis = 0, keep_dims = True))

#r_three = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_three), axis=1, keepdims = True))
theta_foury = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_four_ty [:,2],1), tf.maximum(r_four_tempy,0.001)),0.99),-0.99))
phi_foury  = tf.atan2(tf.expand_dims(centered_calib_points_four_ty [:,1],1),tf.expand_dims(centered_calib_points_four_ty [:,0],1))



ry = tf.concat([r_oney, r_twoy, r_threey, r_foury], axis = 0)
thetay = tf.concat([theta_oney, theta_twoy, theta_threey, theta_foury], axis = 0)
phiy = tf.concat([phi_oney, phi_twoy, phi_threey, phi_foury], axis = 0)

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


r_four_tempyi = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_four_tyi), axis=1, keepdims = True))
r_fouryi = tf.divide(r_four_tempyi,tf.reduce_max(r_four_tempyi, axis = 0, keep_dims = True))

#r_three = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_three), axis=1, keepdims = True))
theta_fouryi = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_four_tyi [:,2],1), tf.maximum(r_four_tempyi,0.001)),0.99),-0.99))
phi_fouryi  = tf.atan2(tf.expand_dims(centered_calib_points_four_tyi [:,1],1),tf.expand_dims(centered_calib_points_four_tyi [:,0],1))


ryi = tf.concat([r_oneyi, r_twoyi, r_threeyi, r_fouryi], axis = 0)
thetayi = tf.concat([theta_oneyi, theta_twoyi, theta_threeyi, theta_fouryi], axis = 0)
phiyi = tf.concat([phi_oneyi, phi_twoyi, phi_threeyi, phi_fouryi], axis = 0)









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
                return 4.0* tf.pow(rho, 4) - 3.0 * tf.pow(rho,2)
        if n == 4 and m == 4:
                return tf.pow(rho,5)
        if n == 5 and m == 1:
                return 10.0* tf.pow(rho, 5) - 12.0 * tf.pow(rho, 3) + 3.0 * rho
        if n == 5 and m == 3:
                return 5.0 * tf.pow(rho, 5) - 4.0 * tf.pow(rho, 3)
        if n == 5 and m == 5:
                return tf.pow(rho,5)
	if n == 6 and m == 0:
		return 20.0*tf.pow(rho, 6) - 30.0*tf.pow(rho, 4)*12.0*tf.pow(rho, 2) - 1
        if n == 6 and m == 2:
                return 15.0* tf.pow(rho, 6) - 20.0 *tf.pow(rho, 4) + 6.0 * tf.pow(rho,2)
        if n == 6 and m == 4:
                return 6.0 * tf.pow(rho, 6)  - 5.0 * tf.pow(rho,4)
        if n == 6 and m == 6:
                return tf.pow(rho, 6)

###########################################################################
def get_harmonics(theta, scope):
        with tf.variable_scope(scope):
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
		y_0_5 = spherical_harmonic(0.0,5.0) * (1.0/8.0)*tf.cos(theta)*(63.0*tf.pow(tf.cos(theta),4)-70.0*tf.square(tf.cos(theta)) +  15)
		y_1_5 = spherical_harmonic(1.0,5.0)* (-15.0/8.0)*tf.sqrt(1-tf.square(tf.cos(theta)))*(21.0*tf.pow(tf.cos(theta),4) - 14.0*tf.square(tf.cos(theta)) + 1)
		y_2_5 = spherical_harmonic(2.0,5.0)* (105.0/2.0)*tf.cos(theta)*(1-tf.square(tf.cos(theta)))*(3.0*tf.square(tf.cos(theta))-1)
		y_3_5 = spherical_harmonic(3.0,5.0) * (-105.0/2.0)*tf.pow(tf.sin(theta),3)*(9.0*tf.square(tf.cos(theta))-1)
		y_4_5 = spherical_harmonic(4.0,5.0)*945.0*tf.cos(theta)*tf.pow(tf.sin(theta),4)
		y_5_5 = spherical_harmonic(5.0,5.0)*-945.0*tf.pow(tf.sin(theta),5)
		y_0_6 = spherical_harmonic(0.0,6.0)*(1.0/16)*(231.0*tf.pow(tf.cos(theta),6)-315.0*tf.pow(tf.cos(theta),315.0)+105.0*tf.square(tf.cos(theta))-5)
		y_1_6 = spherical_harmonic(1.0,6.0)*(-21.0/8.0)*tf.cos(theta)*(33.0*tf.pow(tf.cos(theta),4)-30.0*tf.square(tf.cos(theta))+5)*tf.sin(theta)
		y_2_6 = spherical_harmonic(2.0,6.0)*(105.0/8.0)*tf.square(tf.sin(theta))*(33.0*tf.pow(tf.cos(theta),4)-18.0*tf.square(tf.cos(theta))+1)
		y_3_6 = spherical_harmonic(3.0,6.0)*(-315.0/2.0)*(11*tf.square(tf.cos(theta))-3)*tf.cos(theta)*tf.pow(tf.sin(theta),3)
		y_4_6 = spherical_harmonic(4.0,6.0)*(945.0/2.0)*(11.0*tf.square(tf.cos(theta))-1)*tf.pow(tf.sin(theta),4)
		y_5_6 = spherical_harmonic(5.0,6.0)*-10395.0*tf.cos(theta)*tf.pow(tf.sin(theta),5)
		y_6_6 = spherical_harmonic(6.0,6.0)*10395.0*tf.pow(tf.sin(theta),6)

		
                return tf.concat([y_0_0,y_0_1,y_1_1,y_0_2,y_1_2,y_2_2,y_0_3,y_1_3,y_2_3,y_3_3,y_0_4,y_1_4,y_2_4,y_3_4,y_4_4, y_0_5, y_1_5,y_2_5, y_3_5, y_4_5, y_5_5, y_0_6, y_1_6, y_2_6, y_3_6,y_4_6,y_5_6,y_6_6],axis = 1)



def get_zernike_mat(r,theta,phi,scope):
        with tf.variable_scope(scope):
                harmonics = get_harmonics(theta,scope)
                u_0_0_0 = tf.multiply(tf.slice(harmonics,[0,0],[-1,1]), tf.cos(0.0)) * radial_poly(r,0,0)
                u_0_1_1 = tf.multiply(tf.slice(harmonics,[0,1],[-1,1]), tf.cos(0.0)) * radial_poly(r,1,1)
                u_1_1_1 = tf.multiply(tf.slice(harmonics,[0,2],[-1,1]), tf.cos(phi)) * radial_poly(r,1,1)
                u_0_0_2 = tf.multiply(tf.slice(harmonics,[0,3],[-1,1]), tf.cos(0.0)) * radial_poly(r,0,2)
                u_0_2_2 = tf.multiply(tf.slice(harmonics,[0,3],[-1,1]), tf.cos(0.0)) * radial_poly(r,2,2)
                u_1_2_2 = tf.multiply(tf.slice(harmonics,[0,4],[-1,1]), tf.cos(phi)) * radial_poly(r,2,2)
                u_2_2_2 = tf.multiply(tf.slice(harmonics,[0,5],[-1,1]), tf.cos(2.0*phi)) * radial_poly(r,2,2)
                u_0_1_3 = tf.multiply(tf.slice(harmonics,[0,1],[-1,1]), tf.cos(0.0)) * radial_poly(r,1,3)
                u_1_1_3 = tf.multiply(tf.slice(harmonics,[0,2],[-1,1]), tf.cos(phi)) * radial_poly(r,1,3)
                u_0_3_3 = tf.multiply(tf.slice(harmonics,[0,6],[-1,1]), tf.cos(0.0)) * radial_poly(r,3,3)
                u_1_3_3 = tf.multiply(tf.slice(harmonics,[0,7],[-1,1]), tf.cos(phi)) * radial_poly(r,3,3)
                u_2_3_3 = tf.multiply(tf.slice(harmonics,[0,8],[-1,1]), tf.cos(2.0*phi)) * radial_poly(r,3,3)
                u_3_3_3 = tf.multiply(tf.slice(harmonics,[0,9],[-1,1]), tf.cos(3.0*phi)) * radial_poly(r,3,3)
                u_0_0_4 = tf.multiply(tf.slice(harmonics,[0,0],[-1,1]), tf.cos(0.0)) * radial_poly(r,0,4)
                u_0_2_4 = tf.multiply(tf.slice(harmonics,[0,3],[-1,1]), tf.cos(0.0)) * radial_poly(r,2,4)
                u_1_2_4 = tf.multiply(tf.slice(harmonics,[0,4],[-1,1]), tf.cos(phi)) * radial_poly(r,2,4)
                u_2_2_4 = tf.multiply(tf.slice(harmonics,[0,5],[-1,1]), tf.cos(2.0*phi)) * radial_poly(r,2,4)
                u_0_4_4 = tf.multiply(tf.slice(harmonics,[0,10],[-1,1]), tf.cos(0.0))* radial_poly(r,4,4)
                u_1_4_4 = tf.multiply(tf.slice(harmonics,[0,11],[-1,1]), tf.cos(phi)) * radial_poly(r,4,4)
                u_2_4_4 = tf.multiply(tf.slice(harmonics,[0,12],[-1,1]), tf.cos(2.0 * phi))* radial_poly(r,4,4)
                u_3_4_4 = tf.multiply(tf.slice(harmonics,[0,13],[-1,1]), tf.cos(3.0 * phi))* radial_poly(r,4,4)
                u_4_4_4 = tf.multiply(tf.slice(harmonics,[0,14],[-1,1]), tf.cos(4.0 * phi))* radial_poly(r,4,4)
		u_0_1_5 = tf.multiply(tf.slice(harmonics,[0,1],[-1,1]), tf.cos(0.0 * phi))* radial_poly(r,1,5)
		u_1_1_5 = tf.multiply(tf.slice(harmonics,[0,2],[-1,1]), tf.cos(1.0 * phi))* radial_poly(r,1,5)
		u_0_3_5 = tf.multiply(tf.slice(harmonics,[0,6],[-1,1]), tf.cos(0.0 * phi))* radial_poly(r,3,5)
		u_1_3_5 = tf.multiply(tf.slice(harmonics,[0,7],[-1,1]), tf.cos(1.0 * phi))* radial_poly(r,3,5)
		u_2_3_5 = tf.multiply(tf.slice(harmonics,[0,8],[-1,1]), tf.cos(2.0 * phi))* radial_poly(r,3,5)
		u_3_3_5 = tf.multiply(tf.slice(harmonics,[0,9],[-1,1]), tf.cos(3.0 * phi))* radial_poly(r,3,5)
		u_0_5_5 = tf.multiply(tf.slice(harmonics,[0,15],[-1,1]), tf.cos(0.0 * phi))* radial_poly(r,5,5)
		u_1_5_5 = tf.multiply(tf.slice(harmonics,[0,16],[-1,1]), tf.cos(1.0 * phi))* radial_poly(r,5,5)
		u_2_5_5 = tf.multiply(tf.slice(harmonics,[0,17],[-1,1]), tf.cos(2.0 * phi))* radial_poly(r,5,5)
		u_3_5_5 = tf.multiply(tf.slice(harmonics,[0,18],[-1,1]), tf.cos(3.0 * phi))* radial_poly(r,5,5)
		u_4_5_5 = tf.multiply(tf.slice(harmonics,[0,19],[-1,1]), tf.cos(4.0 * phi))* radial_poly(r,5,5)
		u_5_5_5 = tf.multiply(tf.slice(harmonics,[0,20],[-1,1]), tf.cos(5.0 * phi))* radial_poly(r,5,5)
		u_0_0_6 = tf.multiply(tf.slice(harmonics,[0,0],[-1,1]), tf.cos(0.0 * phi))* radial_poly(r,0,6)
		u_0_2_6 = tf.multiply(tf.slice(harmonics,[0,3],[-1,1]), tf.cos(0.0 * phi))* radial_poly(r,2,6)
		u_1_2_6 = tf.multiply(tf.slice(harmonics,[0,4],[-1,1]), tf.cos(1.0 * phi))* radial_poly(r,2,6)
		u_2_2_6 = tf.multiply(tf.slice(harmonics,[0,5],[-1,1]), tf.cos(2.0 * phi))* radial_poly(r,2,6)
		u_0_4_6 = tf.multiply(tf.slice(harmonics,[0,10],[-1,1]), tf.cos(0.0 * phi))* radial_poly(r,4,6)
		u_1_4_6 = tf.multiply(tf.slice(harmonics,[0,11],[-1,1]), tf.cos(1.0 * phi))* radial_poly(r,4,6)
		u_2_4_6 = tf.multiply(tf.slice(harmonics,[0,13],[-1,1]), tf.cos(2.0 * phi))* radial_poly(r,4,6)
		u_3_4_6 = tf.multiply(tf.slice(harmonics,[0,14],[-1,1]), tf.cos(3.0 * phi))* radial_poly(r,4,6)
		u_4_4_6 = tf.multiply(tf.slice(harmonics,[0,14],[-1,1]), tf.cos(4.0 * phi))* radial_poly(r,4,6)
		u_0_6_6 = tf.multiply(tf.slice(harmonics,[0,21],[-1,1]), tf.cos(0.0 * phi))* radial_poly(r,6,6)
		u_1_6_6 = tf.multiply(tf.slice(harmonics,[0,22],[-1,1]), tf.cos(1.0 * phi))* radial_poly(r,6,6)
		u_2_6_6 = tf.multiply(tf.slice(harmonics,[0,23],[-1,1]), tf.cos(2.0 * phi))* radial_poly(r,6,6)
		u_3_6_6 = tf.multiply(tf.slice(harmonics,[0,24],[-1,1]), tf.cos(3.0 * phi))* radial_poly(r,6,6)
		u_4_6_6 = tf.multiply(tf.slice(harmonics,[0,25],[-1,1]), tf.cos(4.0 * phi))* radial_poly(r,6,6)
		u_5_6_6 = tf.multiply(tf.slice(harmonics,[0,26],[-1,1]), tf.cos(5.0 * phi))* radial_poly(r,6,6)
		u_6_6_6 = tf.multiply(tf.slice(harmonics,[0,27],[-1,1]), tf.cos(6.0 * phi))* radial_poly(r,6,6)

		v_0_0_0 = tf.multiply(tf.slice(harmonics,[0,0],[-1,1]), tf.sin(0.0)) * radial_poly(r,0,0)
                v_0_1_1 = tf.multiply(tf.slice(harmonics,[0,1],[-1,1]), tf.sin(0.0)) * radial_poly(r,1,1)
                v_1_1_1 = tf.multiply(tf.slice(harmonics,[0,2],[-1,1]), tf.sin(phi)) * radial_poly(r,1,1)
                v_0_0_2 = tf.multiply(tf.slice(harmonics,[0,3],[-1,1]), tf.sin(0.0)) * radial_poly(r,0,2)
                v_0_2_2 = tf.multiply(tf.slice(harmonics,[0,3],[-1,1]), tf.sin(0.0)) * radial_poly(r,2,2)
                v_1_2_2 = tf.multiply(tf.slice(harmonics,[0,4],[-1,1]), tf.sin(phi)) * radial_poly(r,2,2)
                v_2_2_2 = tf.multiply(tf.slice(harmonics,[0,5],[-1,1]), tf.sin(2.0*phi)) * radial_poly(r,2,2)
                v_0_1_3 = tf.multiply(tf.slice(harmonics,[0,1],[-1,1]), tf.sin(0.0)) * radial_poly(r,1,3)
                v_1_1_3 = tf.multiply(tf.slice(harmonics,[0,2],[-1,1]), tf.sin(phi)) * radial_poly(r,1,3)
                v_0_3_3 = tf.multiply(tf.slice(harmonics,[0,6],[-1,1]), tf.sin(0.0)) * radial_poly(r,3,3)
                v_1_3_3 = tf.multiply(tf.slice(harmonics,[0,7],[-1,1]), tf.sin(phi)) * radial_poly(r,3,3)
                v_2_3_3 = tf.multiply(tf.slice(harmonics,[0,8],[-1,1]), tf.sin(2.0*phi)) * radial_poly(r,3,3)
                v_3_3_3 = tf.multiply(tf.slice(harmonics,[0,9],[-1,1]), tf.sin(3.0*phi)) * radial_poly(r,3,3)
                v_0_0_4 = tf.multiply(tf.slice(harmonics,[0,0],[-1,1]), tf.sin(0.0)) * radial_poly(r,0,4)
                v_0_2_4 = tf.multiply(tf.slice(harmonics,[0,3],[-1,1]), tf.sin(0.0)) * radial_poly(r,2,4)
                v_1_2_4 = tf.multiply(tf.slice(harmonics,[0,4],[-1,1]), tf.sin(phi)) * radial_poly(r,2,4)
                v_2_2_4 = tf.multiply(tf.slice(harmonics,[0,5],[-1,1]), tf.sin(2.0*phi)) * radial_poly(r,2,4)
                v_0_4_4 = tf.multiply(tf.slice(harmonics,[0,10],[-1,1]), tf.sin(0.0))* radial_poly(r,4,4)
                v_1_4_4 = tf.multiply(tf.slice(harmonics,[0,11],[-1,1]), tf.sin(phi)) * radial_poly(r,4,4)
                v_2_4_4 = tf.multiply(tf.slice(harmonics,[0,12],[-1,1]), tf.sin(2.0 * phi))* radial_poly(r,4,4)
                v_3_4_4 = tf.multiply(tf.slice(harmonics,[0,13],[-1,1]), tf.sin(3.0 * phi))* radial_poly(r,4,4)
                v_4_4_4 = tf.multiply(tf.slice(harmonics,[0,14],[-1,1]), tf.sin(4.0 * phi))* radial_poly(r,4,4)
                v_0_1_5 = tf.multiply(tf.slice(harmonics,[0,1],[-1,1]), tf.sin(0.0 * phi))* radial_poly(r,1,5)
                v_1_1_5 = tf.multiply(tf.slice(harmonics,[0,2],[-1,1]), tf.sin(1.0 * phi))* radial_poly(r,1,5)
                v_0_3_5 = tf.multiply(tf.slice(harmonics,[0,6],[-1,1]), tf.sin(0.0 * phi))* radial_poly(r,3,5)
                v_1_3_5 = tf.multiply(tf.slice(harmonics,[0,7],[-1,1]), tf.sin(1.0 * phi))* radial_poly(r,3,5)
                v_2_3_5 = tf.multiply(tf.slice(harmonics,[0,8],[-1,1]), tf.sin(2.0 * phi))* radial_poly(r,3,5)
                v_3_3_5 = tf.multiply(tf.slice(harmonics,[0,9],[-1,1]), tf.sin(3.0 * phi))* radial_poly(r,3,5)
                v_0_5_5 = tf.multiply(tf.slice(harmonics,[0,15],[-1,1]), tf.sin(0.0 * phi))* radial_poly(r,5,5)
                v_1_5_5 = tf.multiply(tf.slice(harmonics,[0,16],[-1,1]), tf.sin(1.0 * phi))* radial_poly(r,5,5)
                v_2_5_5 = tf.multiply(tf.slice(harmonics,[0,17],[-1,1]), tf.sin(2.0 * phi))* radial_poly(r,5,5)
                v_3_5_5 = tf.multiply(tf.slice(harmonics,[0,18],[-1,1]), tf.sin(3.0 * phi))* radial_poly(r,5,5)
                v_4_5_5 = tf.multiply(tf.slice(harmonics,[0,19],[-1,1]), tf.sin(4.0 * phi))* radial_poly(r,5,5)
                v_5_5_5 = tf.multiply(tf.slice(harmonics,[0,20],[-1,1]), tf.sin(5.0 * phi))* radial_poly(r,5,5)
		v_0_0_6 = tf.multiply(tf.slice(harmonics,[0,0],[-1,1]), tf.sin(0.0 * phi))* radial_poly(r,0,6)
                v_0_2_6 = tf.multiply(tf.slice(harmonics,[0,3],[-1,1]), tf.sin(0.0 * phi))* radial_poly(r,2,6)
                v_1_2_6 = tf.multiply(tf.slice(harmonics,[0,4],[-1,1]), tf.sin(1.0 * phi))* radial_poly(r,2,6)
                v_2_2_6 = tf.multiply(tf.slice(harmonics,[0,5],[-1,1]), tf.sin(2.0 * phi))* radial_poly(r,2,6)
                v_0_4_6 = tf.multiply(tf.slice(harmonics,[0,10],[-1,1]), tf.sin(0.0 * phi))* radial_poly(r,4,6)
                v_1_4_6 = tf.multiply(tf.slice(harmonics,[0,11],[-1,1]), tf.sin(1.0 * phi))* radial_poly(r,4,6)
                v_2_4_6 = tf.multiply(tf.slice(harmonics,[0,13],[-1,1]), tf.sin(2.0 * phi))* radial_poly(r,4,6)
                v_3_4_6 = tf.multiply(tf.slice(harmonics,[0,14],[-1,1]), tf.sin(3.0 * phi))* radial_poly(r,4,6)
                v_4_4_6 = tf.multiply(tf.slice(harmonics,[0,14],[-1,1]), tf.sin(4.0 * phi))* radial_poly(r,4,6)
                v_0_6_6 = tf.multiply(tf.slice(harmonics,[0,21],[-1,1]), tf.sin(0.0 * phi))* radial_poly(r,6,6)
                v_1_6_6 = tf.multiply(tf.slice(harmonics,[0,22],[-1,1]), tf.sin(1.0 * phi))* radial_poly(r,6,6)
                v_2_6_6 = tf.multiply(tf.slice(harmonics,[0,23],[-1,1]), tf.sin(2.0 * phi))* radial_poly(r,6,6)
                v_3_6_6 = tf.multiply(tf.slice(harmonics,[0,24],[-1,1]), tf.sin(3.0 * phi))* radial_poly(r,6,6)
                v_4_6_6 = tf.multiply(tf.slice(harmonics,[0,25],[-1,1]), tf.sin(4.0 * phi))* radial_poly(r,6,6)
                v_5_6_6 = tf.multiply(tf.slice(harmonics,[0,26],[-1,1]), tf.sin(5.0 * phi))* radial_poly(r,6,6)
                v_6_6_6 = tf.multiply(tf.slice(harmonics,[0,27],[-1,1]), tf.sin(6.0 * phi))* radial_poly(r,6,6)		

		V = tf.concat([v_0_0_0, v_0_1_1, v_1_1_1, v_0_0_2, v_0_2_2, v_1_2_2, v_2_2_2,
                        v_0_1_3, v_1_1_3, v_0_3_3, v_1_3_3, v_2_3_3, v_3_3_3,
                                v_0_0_4, v_0_2_4, v_1_2_4, v_2_2_4, v_0_4_4, v_1_4_4, v_2_4_4, v_3_4_4, v_4_4_4, v_0_1_5,v_1_1_5,v_0_3_5,v_1_3_5,v_2_3_5,v_3_3_5, v_0_5_5,v_1_5_5,v_2_5_5,v_3_5_5,v_4_5_5,v_5_5_5, v_0_0_6,v_0_2_6,v_1_2_6,v_2_2_6,v_0_4_6,v_1_4_6,v_2_4_6,v_3_4_6,v_4_4_6,v_0_6_6,v_1_6_6,v_2_6_6,v_3_6_6,v_4_6_6,v_5_6_6,v_6_6_6] ,  axis=1)


		U = tf.concat([u_0_0_0, u_0_1_1, u_1_1_1, u_0_0_2, u_0_2_2, u_1_2_2, u_2_2_2,
                        u_0_1_3, u_1_1_3, u_0_3_3, u_1_3_3, u_2_3_3, u_3_3_3,
                                u_0_0_4, u_0_2_4, u_1_2_4, u_2_2_4, u_0_4_4, u_1_4_4, u_2_4_4, u_3_4_4, u_4_4_4, u_0_1_5,u_1_1_5,u_0_3_5,u_1_3_5,u_2_3_5,u_3_3_5, u_0_5_5,u_1_5_5,u_2_5_5,u_3_5_5,u_4_5_5,u_5_5_5, v_0_0_6,v_0_2_6,v_1_2_6,v_2_2_6,v_0_4_6,v_1_4_6,v_2_4_6,v_3_4_6,v_4_4_6,v_0_6_6,v_1_6_6,v_2_6_6,v_3_6_6,v_4_6_6,v_5_6_6,v_6_6_6] ,  axis=1)

		X_ = tf.concat([U,V] ,  axis=1)
		
		return X_
################################################################################################
keep_prob = tf.placeholder(tf.float32, name="keep_probabilty")

def calc_inverse(in_mat, scope):
        with tf.variable_scope(scope):
                init_guess  = 0.0001 * tf.transpose(in_mat,[1,0] )
                X_cal_1, p_1 = tf.while_loop(lambda x, i: tf.logical_and(i < 4 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
                        lambda x, i:( tf.matmul( tf.eye(100) + 1.0/4.0 * tf.matmul(tf.eye(100)-tf.matmul(x,in_mat),tf.matmul(3.0*tf.eye(100)-tf.matmul(x,in_mat),3.0*tf.eye(100)-tf.matmul(x,in_mat))),x),i+1), (init_guess, 0))
                return X_cal_1


"""
mask_0_2_ = tf.greater(r_one, 0)
res_0_2_ = tf.boolean_mask(r_one, mask_0_2_)
theta_0_2_ = tf.boolean_mask(theta_one, mask_0_2_)
phi_0_2_ = tf.boolean_mask(phi_one, mask_0_2_)
mask_0_2 = tf.less(res_0_2_,0.2)
res_0_2 = tf.reshape(tf.boolean_mask(res_0_2_, mask_0_2),[-1,1])
theta_0_2 = tf.reshape(tf.boolean_mask(theta_0_2_, mask_0_2),[-1,1])
phi_0_2 = tf.reshape(tf.boolean_mask(phi_0_2_, mask_0_2),[-1,1])



mask_2_4_ = tf.greater(r_one, 0.2)
res_2_4_ = tf.boolean_mask(r_one, mask_2_4_)
theta_2_4_ = tf.boolean_mask(theta_one, mask_2_4_)
phi_2_4_ = tf.boolean_mask(phi_one, mask_2_4_)
mask_2_4 = tf.less(res_2_4_,0.4)
res_2_4 = tf.reshape(tf.boolean_mask(res_2_4_, mask_2_4),[-1,1])
theta_2_4 = tf.reshape(tf.boolean_mask(theta_2_4_, mask_2_4),[-1,1])
phi_2_4 = tf.reshape(tf.boolean_mask(phi_2_4_, mask_2_4),[-1,1])

mask_4_6_ = tf.greater(r_one, 0.4)
res_4_6_ = tf.boolean_mask(r_one, mask_4_6_)
theta_4_6_ = tf.boolean_mask(theta_one, mask_4_6_)
phi_4_6_ = tf.boolean_mask(phi_one, mask_4_6_)
mask_4_6 = tf.less(res_4_6_,0.6)
res_4_6 = tf.reshape(tf.boolean_mask(res_4_6_, mask_4_6),[-1,1])
theta_4_6 = tf.reshape(tf.boolean_mask(theta_4_6_, mask_4_6),[-1,1])
phi_4_6 = tf.reshape(tf.boolean_mask(phi_4_6_, mask_4_6),[-1,1])


mask_6_8_ = tf.greater(r_one, 0.6)
res_6_8_ = tf.boolean_mask(r_one, mask_6_8_)
theta_6_8_ = tf.boolean_mask(theta_one, mask_6_8_)
phi_6_8_ = tf.boolean_mask(phi_one, mask_6_8_)
mask_6_8 = tf.less(res_4_6_,0.8)
res_6_8 = tf.reshape(tf.boolean_mask(res_6_8_, mask_6_8),[-1,1])
theta_6_8 = tf.reshape(tf.boolean_mask(theta_6_8_, mask_6_8),[-1,1])
phi_6_8 = tf.reshape(tf.boolean_mask(phi_6_8_, mask_6_8),[-1,1])


mask_8_10_= tf.greater(r_one, 0.8)
res_8_10_ = tf.boolean_mask(r_one, mask_8_10_)
theta_8_10_ = tf.boolean_mask(theta_one, mask_8_10_)
phi_8_10_ = tf.boolean_mask(phi_one, mask_8_10_)
mask_8_10 = tf.less(res_8_10_,1.0)
res_8_10 = tf.reshape(tf.boolean_mask(res_8_10_, mask_8_10),[-1,1])
theta_8_10 = tf.reshape(tf.boolean_mask(theta_8_10_, mask_8_10),[-1,1])
phi_8_10 = tf.reshape(tf.boolean_mask(phi_8_10_, mask_8_10),[-1,1])
"""
def get_slice(r_one, theta_one, phi_one, low, high, scope):
	with tf.variable_scope(scope):
		mask_0_2_ = tf.greater(r_one, low)
		res_0_2_ = tf.boolean_mask(r_one, mask_0_2_)
		theta_0_2_ = tf.boolean_mask(theta_one, mask_0_2_)
		phi_0_2_ = tf.boolean_mask(phi_one, mask_0_2_)
		mask_0_2 = tf.less(res_0_2_,high)
		res_0_2 = tf.reshape(tf.boolean_mask(res_0_2_, mask_0_2),[-1,1])
		theta_0_2 = tf.reshape(tf.boolean_mask(theta_0_2_, mask_0_2),[-1,1])
		phi_0_2 = tf.reshape(tf.boolean_mask(phi_0_2_, mask_0_2),[-1,1])

		return res_0_2, theta_0_2, phi_0_2

res_1, theta_1, phi_1 = get_slice(r_one, theta_one, phi_one, 0.0, 1.0, "s1")
res_11, theta_11, phi_11 = get_slice(r_one, theta_one, phi_one, 0.025, 1.0, "s11")
res_2, theta_2, phi_2 = get_slice(r_one, theta_one, phi_one, 0.05, 1.0, "s2")
res_12, theta_12, phi_12 = get_slice(r_one, theta_one, phi_one, 0.075, 1.0, "s12")

res_3, theta_3, phi_3 = get_slice(r_one, theta_one, phi_one, 0.1, 1.0, "s3")
res_13, theta_13, phi_13 = get_slice(r_one, theta_one, phi_one, 0.125, 1.0, "s13")
res_4, theta_4, phi_4 = get_slice(r_one, theta_one, phi_one, 0.15, 1.0, "s4")
res_14, theta_14, phi_14 = get_slice(r_one, theta_one, phi_one, 0.175, 1.0, "s14")

res_5, theta_5, phi_5 = get_slice(r_one, theta_one, phi_one, 0.2, 1.0, "s5")
res_15, theta_15, phi_15 = get_slice(r_one, theta_one, phi_one, 0.225, 1.0, "s15")
res_6, theta_6, phi_6 = get_slice(r_one, theta_one, phi_one, 0.25, 1.0, "s6")
res_16, theta_16, phi_16 = get_slice(r_one, theta_one, phi_one, 0.275, 1.0, "s16")

res_7, theta_7, phi_7 = get_slice(r_one, theta_one, phi_one, 0.3, 1.0, "s7")
res_17, theta_17, phi_17 = get_slice(r_one, theta_one, phi_one, 0.325, 1.0, "s17")
res_8, theta_8, phi_8 = get_slice(r_one, theta_one, phi_one, 0.35, 1.0, "s8")
res_18, theta_18, phi_18 = get_slice(r_one, theta_one, phi_one, 0.375, 1.0, "s18")

res_9, theta_9, phi_9 = get_slice(r_one, theta_one, phi_one, 0.4, 1.0, "s9")
res_19, theta_19, phi_19 = get_slice(r_one, theta_one, phi_one, 0.425, 1.0, "s19")
res_10, theta_10, phi_10 = get_slice(r_one, theta_one, phi_one, 0.45, 1.0, "s10")
res_20, theta_20, phi_20 = get_slice(r_one, theta_one, phi_one, 0.475, 1.0, "s20")



X_1 = get_zernike_mat(res_1,theta_1,phi_1,"x1")
X_2 = get_zernike_mat(res_2,theta_2,phi_2,"x2")
X_3 = get_zernike_mat(res_3,theta_3,phi_3,"x3")
X_4 = get_zernike_mat(res_4,theta_4,phi_4,"x4")
X_5 = get_zernike_mat(res_5,theta_5,phi_5,"x5")
X_6 = get_zernike_mat(res_6,theta_6,phi_6,"x6")
X_7 = get_zernike_mat(res_7,theta_7,phi_7,"x7")
X_8 = get_zernike_mat(res_8,theta_8,phi_8,"x8")
X_9 = get_zernike_mat(res_9,theta_9,phi_9,"x9")
X_10 = get_zernike_mat(res_10,theta_10,phi_10,"x10")

X_11 = get_zernike_mat(res_11,theta_11,phi_11,"x11")
X_12 = get_zernike_mat(res_12,theta_12,phi_12,"x12")
X_13 = get_zernike_mat(res_13,theta_13,phi_13,"x13")
X_14 = get_zernike_mat(res_14,theta_14,phi_14,"x14")
X_15 = get_zernike_mat(res_15,theta_15,phi_15,"x15")
X_16 = get_zernike_mat(res_16,theta_16,phi_16,"x16")
X_17 = get_zernike_mat(res_17,theta_17,phi_17,"x17")
X_18 = get_zernike_mat(res_18,theta_18,phi_18,"x18")
X_19 = get_zernike_mat(res_19,theta_19,phi_19,"x19")
X_20 = get_zernike_mat(res_20,theta_20,phi_20,"x20")



X_inv_1 = calc_inverse(X_1, "inv_1")
X_inv_2 = calc_inverse(X_2, "inv_2")
X_inv_3 = calc_inverse(X_3, "inv_3")
X_inv_4 = calc_inverse(X_4, "inv_4")
X_inv_5 = calc_inverse(X_5, "inv_5")
X_inv_6 = calc_inverse(X_6, "inv_6")
X_inv_7 = calc_inverse(X_7, "inv_7")
X_inv_8 = calc_inverse(X_8, "inv_8")
X_inv_9 = calc_inverse(X_9, "inv_9")
X_inv_10 = calc_inverse(X_10, "inv_10")

X_inv_11 = calc_inverse(X_11, "inv_11")
X_inv_12 = calc_inverse(X_12, "inv_12")
X_inv_13 = calc_inverse(X_13, "inv_13")
X_inv_14 = calc_inverse(X_14, "inv_14")
X_inv_15 = calc_inverse(X_15, "inv_15")
X_inv_16 = calc_inverse(X_16, "inv_16")
X_inv_17 = calc_inverse(X_17, "inv_17")
X_inv_18 = calc_inverse(X_18, "inv_18")
X_inv_19 = calc_inverse(X_19, "inv_19")
X_inv_20 = calc_inverse(X_20, "inv_20")





C_1 = tf.tile(tf.expand_dims(tf.matmul(X_inv_1,res_1),axis=0), [64,1,1])
C_2 = tf.tile(tf.expand_dims(tf.matmul(X_inv_2,res_2),axis=0), [64,1,1])
C_3 = tf.tile(tf.expand_dims(tf.matmul(X_inv_3,res_3),axis=0), [64,1,1])
C_4 = tf.tile(tf.expand_dims(tf.matmul(X_inv_4,res_4),axis=0), [64,1,1])
C_5 = tf.tile(tf.expand_dims(tf.matmul(X_inv_5,res_5),axis=0), [64,1,1])
C_6 = tf.tile(tf.expand_dims(tf.matmul(X_inv_6,res_6),axis=0), [64,1,1])
C_7 = tf.tile(tf.expand_dims(tf.matmul(X_inv_7,res_7),axis=0), [64,1,1])
C_8 = tf.tile(tf.expand_dims(tf.matmul(X_inv_8,res_8),axis=0), [64,1,1])
C_9 = tf.tile(tf.expand_dims(tf.matmul(X_inv_9,res_9),axis=0), [64,1,1])
C_10 = tf.tile(tf.expand_dims(tf.matmul(X_inv_10,res_10),axis=0), [64,1,1])

C_11 = tf.tile(tf.expand_dims(tf.matmul(X_inv_11,res_11),axis=0), [64,1,1])
C_12 = tf.tile(tf.expand_dims(tf.matmul(X_inv_12,res_12),axis=0), [64,1,1])
C_13 = tf.tile(tf.expand_dims(tf.matmul(X_inv_13,res_13),axis=0), [64,1,1])
C_14 = tf.tile(tf.expand_dims(tf.matmul(X_inv_14,res_14),axis=0), [64,1,1])
C_15 = tf.tile(tf.expand_dims(tf.matmul(X_inv_15,res_15),axis=0), [64,1,1])
C_16 = tf.tile(tf.expand_dims(tf.matmul(X_inv_16,res_16),axis=0), [64,1,1])
C_17 = tf.tile(tf.expand_dims(tf.matmul(X_inv_17,res_17),axis=0), [64,1,1])
C_18 = tf.tile(tf.expand_dims(tf.matmul(X_inv_18,res_18),axis=0), [64,1,1])
C_19 = tf.tile(tf.expand_dims(tf.matmul(X_inv_19,res_19),axis=0), [64,1,1])
C_20 = tf.tile(tf.expand_dims(tf.matmul(X_inv_20,res_20),axis=0), [64,1,1])




x_filter1 =  tf.get_variable("a_xfilter1", [64,50,1])

def shift_kernel(val_r, val_phi, val_theta, shift, surf, scope):
	with tf.variable_scope(scope):
		val_ = val_r + shift
		mask = tf.less(val_, 1.0)
		r_ = tf.reshape(tf.boolean_mask(val_, mask),[-1,1])
		phi_ = tf.reshape(tf.boolean_mask(val_phi, mask),[-1,1])
		theta_ = tf.reshape(tf.boolean_mask(val_theta, mask),[-1,1])
		r = tf.concat([r_, tf.zeros([tf.shape(val_r)[0]-tf.shape(r_)[0],1])], axis = 0)
		p = tf.concat([phi_, tf.zeros([tf.shape(val_r)[0]-tf.shape(phi_)[0],1])], axis = 0)
		t = tf.concat([theta_, tf.zeros([tf.shape(val_r)[0]-tf.shape(theta_)[0],1])], axis=0)

		X_ = get_zernike_mat(r,t,p,"Zp_0_8")
		X_inv = calc_inverse(X_, "inverseZp_0_8")
		C_f = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_inv,[0,0],[100,50]), axis =0), [64,1,1]), surf)
		
		return C_f
		
C_pf1 = shift_kernel(r_one, phi_one, theta_one, 0.0, x_filter1, "cf1")
C_pf11 = shift_kernel(r_one, phi_one, theta_one, 0.025, x_filter1, "cf11")
C_pf2 = shift_kernel(r_one, phi_one, theta_one, 0.05, x_filter1, "cf2")
C_pf12 = shift_kernel(r_one, phi_one, theta_one, 0.0575, x_filter1, "cf12")
C_pf3 = shift_kernel(r_one, phi_one, theta_one, 0.1, x_filter1, "cf3")
C_pf13 = shift_kernel(r_one, phi_one, theta_one, 0.125, x_filter1, "cf13")
C_pf4 = shift_kernel(r_one, phi_one, theta_one, 0.15, x_filter1, "cf4")
C_pf14 = shift_kernel(r_one, phi_one, theta_one, 0.175, x_filter1, "cf14")
C_pf5 = shift_kernel(r_one, phi_one, theta_one, 0.2, x_filter1, "cf5")
C_pf15 = shift_kernel(r_one, phi_one, theta_one, 0.225, x_filter1, "cf15")
C_pf6 = shift_kernel(r_one, phi_one, theta_one, 0.25, x_filter1, "cf6")
C_pf16 = shift_kernel(r_one, phi_one, theta_one, 0.275, x_filter1, "cf16")
C_pf7 = shift_kernel(r_one, phi_one, theta_one, 0.3, x_filter1, "cf7")
C_pf17 = shift_kernel(r_one, phi_one, theta_one, 0.325, x_filter1, "cf17")
C_pf8 = shift_kernel(r_one, phi_one, theta_one, 0.35, x_filter1, "cf8")
C_pf18 = shift_kernel(r_one, phi_one, theta_one, 0.375, x_filter1, "cf18")
C_pf9 = shift_kernel(r_one, phi_one, theta_one, 0.4, x_filter1, "cf9")
C_pf19 = shift_kernel(r_one, phi_one, theta_one, 0.425, x_filter1, "cf19")
C_pf10 = shift_kernel(r_one, phi_one, theta_one, 0.45, x_filter1, "cf10")
C_pf20 = shift_kernel(r_one, phi_one, theta_one, 0.475, x_filter1, "cf20")





#C_pf_0_2 = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_inv_0_0f,[0,0],[100,50]), axis =0), [64,1,1]), x_filter1)
f_mapz1 = tf.matmul(C_1, tf.transpose(C_pf1, [0,2,1]))
f_mapz2 = tf.matmul(C_2, tf.transpose(C_pf2, [0,2,1]))
f_mapz3 = tf.matmul(C_3, tf.transpose(C_pf3, [0,2,1]))
f_mapz4 = tf.matmul(C_4, tf.transpose(C_pf4, [0,2,1]))
f_mapz5 = tf.matmul(C_5, tf.transpose(C_pf5, [0,2,1]))
f_mapz6 = tf.matmul(C_6, tf.transpose(C_pf6, [0,2,1]))
f_mapz7 = tf.matmul(C_7, tf.transpose(C_pf7, [0,2,1]))
f_mapz8 = tf.matmul(C_8, tf.transpose(C_pf8, [0,2,1]))
f_mapz9 = tf.matmul(C_9, tf.transpose(C_pf9, [0,2,1]))
f_mapz10 = tf.matmul(C_10, tf.transpose(C_pf10, [0,2,1]))


f_mapz11 = tf.matmul(C_11, tf.transpose(C_pf11, [0,2,1]))
f_mapz12 = tf.matmul(C_12, tf.transpose(C_pf12, [0,2,1]))
f_mapz13 = tf.matmul(C_13, tf.transpose(C_pf13, [0,2,1]))
f_mapz14 = tf.matmul(C_14, tf.transpose(C_pf14, [0,2,1]))
f_mapz15 = tf.matmul(C_15, tf.transpose(C_pf15, [0,2,1]))
f_mapz16 = tf.matmul(C_16, tf.transpose(C_pf16, [0,2,1]))
f_mapz17 = tf.matmul(C_17, tf.transpose(C_pf17, [0,2,1]))
f_mapz18 = tf.matmul(C_18, tf.transpose(C_pf18, [0,2,1]))
f_mapz19 = tf.matmul(C_19, tf.transpose(C_pf19, [0,2,1]))
f_mapz20 = tf.matmul(C_20, tf.transpose(C_pf20, [0,2,1]))


map1 = tf.maximum(tf.maximum(f_mapz1,f_mapz2),tf.maximum(f_mapz11,f_mapz12))
map2 = tf.maximum(tf.maximum(f_mapz3,f_mapz4),tf.maximum(f_mapz13,f_mapz14))
map3 = tf.maximum(tf.maximum(f_mapz5,f_mapz6),tf.maximum(f_mapz15,f_mapz16))
map4 = tf.maximum(tf.maximum(f_mapz7,f_mapz8),tf.maximum(f_mapz17,f_mapz18))
map5 = tf.maximum(tf.maximum(f_mapz9,f_mapz10),tf.maximum(f_mapz19,f_mapz20))

#f_map = tf.nn.dropout(tf.nn.relu(tf.reshape(tf.concat([map1,map2,map3,map4,map5],axis = 2), [1,10000*5*64])), keep_prob=keep_prob)


"""
C_pf_2_4 = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_inv_0_2f,[0,0],[100,50]), axis =0), [64,1,1]), x_filter1)
f_mapz_2_4 = tf.matmul(C_2_4, tf.transpose(C_pf_2_4, [0,2,1]))

C_pf_4_6 = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_inv_0_4f,[0,0],[100,50]), axis =0), [64,1,1]), x_filter1)
f_mapz_4_6 = tf.matmul(C_4_6, tf.transpose(C_pf_4_6, [0,2,1]))

C_pf_6_8 = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_inv_0_6f,[0,0],[100,50]), axis =0), [64,1,1]), x_filter1)
f_mapz_6_8 = tf.matmul(C_6_8, tf.transpose(C_pf_6_8, [0,2,1]))

C_pf_8_10 = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_inv_0_8f,[0,0],[100,50]), axis =0), [64,1,1]), x_filter1)
f_mapz_8_10 = tf.matmul(C_8_10, tf.transpose(C_pf_8_10, [0,2,1]))

"""


f_map = tf.nn.dropout(tf.nn.relu(tf.reshape(tf.concat([f_mapz1, f_mapz3, f_mapz5, f_mapz7, f_mapz9],axis = 2), [1,10000*5*64])),keep_prob=keep_prob)

#f_map = tf.nn.dropout(tf.nn.relu(tf.reshape(tf.concat([tf.maximum(f_mapz1,f_mapz2), tf.maximum(f_mapz3,f_mapz4), tf.maximum(f_mapz5,f_mapz6), tf.maximum(f_mapz7,f_mapz8), tf.maximum(f_mapz9,f_mapz10)],axis = 2), [1,10000*5*64])),keep_prob=keep_prob)

###################################################################################################################
resi_1, thetai_1, phii_1 = get_slice(r_one_i, theta_one_i, phi_one_i, 0.0, 1.0, "s1i")
resi_11, thetai_11, phii_11 = get_slice(r_one_i, theta_one_i, phi_one_i, 0.025, 1.0, "s11i")
resi_2, thetai_2, phii_2 = get_slice(r_one_i, theta_one_i, phi_one_i, 0.05, 1.0, "s2i")
resi_12, thetai_12, phii_12 = get_slice(r_one_i, theta_one_i, phi_one_i, 0.075, 1.0, "s13i")

resi_3, thetai_3, phii_3 = get_slice(r_one_i, theta_one_i, phi_one_i, 0.1, 1.0, "s3i")
resi_13, thetai_13, phii_13 = get_slice(r_one_i, theta_one_i, phi_one_i, 0.125, 1.0, "s13i")
resi_4, thetai_4, phii_4 = get_slice(r_one_i, theta_one_i, phi_one_i, 0.15, 1.0, "s4i")
resi_14, thetai_14, phii_14 = get_slice(r_one_i, theta_one_i, phi_one_i, 0.175, 1.0, "s14i")

resi_5, thetai_5, phii_5 = get_slice(r_one_i, theta_one_i, phi_one_i, 0.20, 1.0, "s5i")
resi_15, thetai_15, phii_15 = get_slice(r_one_i, theta_one_i, phi_one_i, 0.225, 1.0, "s15i")
resi_6, thetai_6, phii_6 = get_slice(r_one_i, theta_one_i, phi_one_i, 0.25, 1.0, "s6i")
resi_16, thetai_16, phii_16 = get_slice(r_one_i, theta_one_i, phi_one_i, 0.275, 1.0, "s16i")

resi_7, thetai_7, phii_7 = get_slice(r_one_i, theta_one_i, phi_one_i, 0.30, 1.0, "s7i")
resi_17, thetai_17, phii_17 = get_slice(r_one_i, theta_one_i, phi_one_i, 0.325, 1.0, "s17i")
resi_8, thetai_8, phii_8 = get_slice(r_one_i, theta_one_i, phi_one_i, 0.35, 1.0, "s8i")
resi_18, thetai_18, phii_18 = get_slice(r_one_i, theta_one_i, phi_one_i, 0.375, 1.0, "s18i")

resi_9, thetai_9, phii_9 = get_slice(r_one_i, theta_one_i, phi_one_i, 0.40, 1.0, "s9i")
resi_19, thetai_19, phii_19 = get_slice(r_one_i, theta_one_i, phi_one_i, 0.425, 1.0, "s19i")
resi_10, thetai_10, phii_10 = get_slice(r_one_i, theta_one_i, phi_one_i, 0.45, 1.0, "s10i")
resi_20, thetai_20, phii_20 = get_slice(r_one_i, theta_one_i, phi_one_i, 0.475, 1.0, "s20i")


Xi_1 = get_zernike_mat(resi_1,thetai_1,phii_1,"xi1")
Xi_2 = get_zernike_mat(resi_2,thetai_2,phii_2,"xi2")
Xi_3 = get_zernike_mat(resi_3,thetai_3,phii_3,"xi3")
Xi_4 = get_zernike_mat(resi_4,thetai_4,phii_4,"xi4")
Xi_5 = get_zernike_mat(resi_5,thetai_5,phii_5,"xi5")
Xi_6 = get_zernike_mat(resi_6,thetai_6,phii_6,"xi6")
Xi_7 = get_zernike_mat(resi_7,thetai_7,phii_7,"xi7")
Xi_8 = get_zernike_mat(resi_8,thetai_8,phii_8,"xi8")
Xi_9 = get_zernike_mat(resi_9,thetai_9,phii_9,"xi9")
Xi_10 = get_zernike_mat(resi_10,thetai_10,phii_10,"xi10")

Xi_11 = get_zernike_mat(resi_11,thetai_11,phii_11,"xi11")
Xi_12 = get_zernike_mat(resi_12,thetai_12,phii_12,"xi12")
Xi_13 = get_zernike_mat(resi_13,thetai_13,phii_13,"xi13")
Xi_14 = get_zernike_mat(resi_14,thetai_14,phii_14,"xi14")
Xi_15 = get_zernike_mat(resi_15,thetai_15,phii_15,"xi15")
Xi_16 = get_zernike_mat(resi_16,thetai_16,phii_16,"xi16")
Xi_17 = get_zernike_mat(resi_17,thetai_17,phii_17,"xi17")
Xi_18 = get_zernike_mat(resi_18,thetai_18,phii_18,"xi18")
Xi_19 = get_zernike_mat(resi_19,thetai_19,phii_19,"xi19")
Xi_20 = get_zernike_mat(resi_20,thetai_20,phii_20,"xi20")



Xi_invi_1 = calc_inverse(Xi_1, "invi_1")
Xi_invi_2 = calc_inverse(Xi_2, "invi_2")
Xi_invi_3 = calc_inverse(Xi_3, "invi_3")
Xi_invi_4 = calc_inverse(Xi_4, "invi_4")
Xi_invi_5 = calc_inverse(Xi_5, "invi_5")
Xi_invi_6 = calc_inverse(Xi_6, "invi_6")
Xi_invi_7 = calc_inverse(Xi_7, "invi_7")
Xi_invi_8 = calc_inverse(Xi_8, "invi_8")
Xi_invi_9 = calc_inverse(Xi_9, "invi_9")
Xi_invi_10 = calc_inverse(Xi_10, "invi_10")
Xi_invi_11 = calc_inverse(Xi_11, "invi_11")
Xi_invi_12 = calc_inverse(Xi_12, "invi_12")
Xi_invi_13 = calc_inverse(Xi_13, "invi_13")
Xi_invi_14 = calc_inverse(Xi_14, "invi_14")
Xi_invi_15 = calc_inverse(Xi_15, "invi_15")
Xi_invi_16 = calc_inverse(Xi_16, "invi_16")
Xi_invi_17 = calc_inverse(Xi_17, "invi_17")
Xi_invi_18 = calc_inverse(Xi_18, "invi_18")
Xi_invi_19 = calc_inverse(Xi_19, "invi_19")
Xi_invi_20 = calc_inverse(Xi_20, "invi_20")



Ci_1 = tf.tile(tf.expand_dims(tf.matmul(Xi_invi_1,resi_1),axis=0), [64,1,1])
Ci_2 = tf.tile(tf.expand_dims(tf.matmul(Xi_invi_2,resi_2),axis=0), [64,1,1])
Ci_3 = tf.tile(tf.expand_dims(tf.matmul(Xi_invi_3,resi_3),axis=0), [64,1,1])
Ci_4 = tf.tile(tf.expand_dims(tf.matmul(Xi_invi_4,resi_4),axis=0), [64,1,1])
Ci_5 = tf.tile(tf.expand_dims(tf.matmul(Xi_invi_5,resi_5),axis=0), [64,1,1])
Ci_6 = tf.tile(tf.expand_dims(tf.matmul(Xi_invi_6,resi_6),axis=0), [64,1,1])
Ci_7 = tf.tile(tf.expand_dims(tf.matmul(Xi_invi_7,resi_7),axis=0), [64,1,1])
Ci_8 = tf.tile(tf.expand_dims(tf.matmul(Xi_invi_8,resi_8),axis=0), [64,1,1])
Ci_9 = tf.tile(tf.expand_dims(tf.matmul(Xi_invi_9,resi_9),axis=0), [64,1,1])
Ci_10 = tf.tile(tf.expand_dims(tf.matmul(Xi_invi_10,resi_10),axis=0), [64,1,1])
Ci_11 = tf.tile(tf.expand_dims(tf.matmul(Xi_invi_11,resi_11),axis=0), [64,1,1])
Ci_12 = tf.tile(tf.expand_dims(tf.matmul(Xi_invi_12,resi_12),axis=0), [64,1,1])
Ci_13 = tf.tile(tf.expand_dims(tf.matmul(Xi_invi_13,resi_13),axis=0), [64,1,1])
Ci_14 = tf.tile(tf.expand_dims(tf.matmul(Xi_invi_14,resi_14),axis=0), [64,1,1])
Ci_15 = tf.tile(tf.expand_dims(tf.matmul(Xi_invi_15,resi_15),axis=0), [64,1,1])
Ci_16 = tf.tile(tf.expand_dims(tf.matmul(Xi_invi_16,resi_16),axis=0), [64,1,1])
Ci_17 = tf.tile(tf.expand_dims(tf.matmul(Xi_invi_17,resi_17),axis=0), [64,1,1])
Ci_18 = tf.tile(tf.expand_dims(tf.matmul(Xi_invi_18,resi_18),axis=0), [64,1,1])
Ci_19 = tf.tile(tf.expand_dims(tf.matmul(Xi_invi_19,resi_19),axis=0), [64,1,1])
Ci_20 = tf.tile(tf.expand_dims(tf.matmul(Xi_invi_20,resi_20),axis=0), [64,1,1])


x_filter1i =  tf.get_variable("a_xfilter1i", [64,50,1])

Ci_pfi1 = shift_kernel(r_one_i, phi_one_i, theta_one_i, 0.0, x_filter1i, "cfi1")
Ci_pfi11 = shift_kernel(r_one_i, phi_one_i, theta_one_i, 0.025, x_filter1i, "cfi11")
Ci_pfi2 = shift_kernel(r_one_i, phi_one_i, theta_one_i, 0.05, x_filter1i, "cfi2")
Ci_pfi12 = shift_kernel(r_one_i, phi_one_i, theta_one_i, 0.075, x_filter1i, "cfi12")
Ci_pfi3 = shift_kernel(r_one_i, phi_one_i, theta_one_i, 0.1, x_filter1i, "cfi3")
Ci_pfi13 = shift_kernel(r_one_i, phi_one_i, theta_one_i, 0.125, x_filter1i, "cfi13")
Ci_pfi4 = shift_kernel(r_one_i, phi_one_i, theta_one_i, 0.15, x_filter1i, "cfi4")
Ci_pfi14 = shift_kernel(r_one_i, phi_one_i, theta_one_i, 0.175, x_filter1i, "cfi14")
Ci_pfi5 = shift_kernel(r_one_i, phi_one_i, theta_one_i, 0.2, x_filter1i, "cfi5")
Ci_pfi15 = shift_kernel(r_one_i, phi_one_i, theta_one_i, 0.225, x_filter1i, "cfi15")
Ci_pfi6 = shift_kernel(r_one_i, phi_one_i, theta_one_i, 0.25, x_filter1i, "cfi6")
Ci_pfi16 = shift_kernel(r_one_i, phi_one_i, theta_one_i, 0.275, x_filter1i, "cfi16")
Ci_pfi7 = shift_kernel(r_one_i, phi_one_i, theta_one_i, 0.3, x_filter1i, "cfi7")
Ci_pfi17 = shift_kernel(r_one_i, phi_one_i, theta_one_i, 0.325, x_filter1i, "cfi17")
Ci_pfi8 = shift_kernel(r_one_i, phi_one_i, theta_one_i, 0.35, x_filter1i, "cfi8")
Ci_pfi18 = shift_kernel(r_one_i, phi_one_i, theta_one_i, 0.375, x_filter1i, "cfi18")
Ci_pfi9 = shift_kernel(r_one_i, phi_one_i, theta_one_i, 0.4, x_filter1i, "cfi9")
Ci_pfi19 = shift_kernel(r_one_i, phi_one_i, theta_one_i, 0.425, x_filter1i, "cfi19")
Ci_pfi10 = shift_kernel(r_one_i, phi_one_i, theta_one_i, 0.45, x_filter1i, "cfi10")
Ci_pfi20 = shift_kernel(r_one_i, phi_one_i, theta_one_i, 0.475, x_filter1i, "cfi20")


#Ci_pfi_0_2 = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_inv_0_0f,[0,0],[100,50]), axis =0), [64,1,1]), x_filter1i)
f_mapzi1 = tf.matmul(Ci_1, tf.transpose(Ci_pfi1, [0,2,1]))
f_mapzi2 = tf.matmul(Ci_2, tf.transpose(Ci_pfi2, [0,2,1]))
f_mapzi3 = tf.matmul(Ci_3, tf.transpose(Ci_pfi3, [0,2,1]))
f_mapzi4 = tf.matmul(Ci_4, tf.transpose(Ci_pfi4, [0,2,1]))
f_mapzi5 = tf.matmul(Ci_5, tf.transpose(Ci_pfi5, [0,2,1]))
f_mapzi6 = tf.matmul(Ci_6, tf.transpose(Ci_pfi6, [0,2,1]))
f_mapzi7 = tf.matmul(Ci_7, tf.transpose(Ci_pfi7, [0,2,1]))
f_mapzi8 = tf.matmul(Ci_8, tf.transpose(Ci_pfi8, [0,2,1]))
f_mapzi9 = tf.matmul(Ci_9, tf.transpose(Ci_pfi9, [0,2,1]))
f_mapzi10 = tf.matmul(Ci_10, tf.transpose(Ci_pfi10, [0,2,1]))

f_mapzi11 = tf.matmul(Ci_11, tf.transpose(Ci_pfi11, [0,2,1]))
f_mapzi12 = tf.matmul(Ci_12, tf.transpose(Ci_pfi12, [0,2,1]))
f_mapzi13 = tf.matmul(Ci_13, tf.transpose(Ci_pfi13, [0,2,1]))
f_mapzi14 = tf.matmul(Ci_14, tf.transpose(Ci_pfi14, [0,2,1]))
f_mapzi15 = tf.matmul(Ci_15, tf.transpose(Ci_pfi15, [0,2,1]))
f_mapzi16 = tf.matmul(Ci_16, tf.transpose(Ci_pfi16, [0,2,1]))
f_mapzi17 = tf.matmul(Ci_17, tf.transpose(Ci_pfi17, [0,2,1]))
f_mapzi18 = tf.matmul(Ci_18, tf.transpose(Ci_pfi18, [0,2,1]))
f_mapzi19 = tf.matmul(Ci_19, tf.transpose(Ci_pfi19, [0,2,1]))
f_mapzi20 = tf.matmul(Ci_20, tf.transpose(Ci_pfi20, [0,2,1]))


map1i = tf.maximum(tf.maximum(f_mapzi1,f_mapzi2),tf.maximum(f_mapzi11,f_mapzi12))
map2i = tf.maximum(tf.maximum(f_mapzi3,f_mapzi4),tf.maximum(f_mapzi13,f_mapzi14))
map3i = tf.maximum(tf.maximum(f_mapzi5,f_mapzi6),tf.maximum(f_mapzi15,f_mapzi16))
map4i = tf.maximum(tf.maximum(f_mapzi7,f_mapzi8),tf.maximum(f_mapzi17,f_mapzi18))
map5i = tf.maximum(tf.maximum(f_mapzi9,f_mapzi10),tf.maximum(f_mapzi19,f_mapzi20))

#f_map_i = tf.nn.dropout(tf.nn.relu(tf.reshape(tf.concat([map1i,map2i,map3i,map4i,map5i],axis = 2), [1,10000*5*64])), keep_prob=keep_prob)

f_map_i = tf.nn.dropout(tf.nn.relu(tf.reshape(tf.concat([f_mapzi1, f_mapzi3 , f_mapzi5 , f_mapzi7 , f_mapzi9 ],axis = 2), [1,10000*5*64])), keep_prob=keep_prob)

#f_map_i = tf.nn.dropout(tf.nn.relu(tf.reshape(tf.concat([tf.maximum(f_mapzi1,f_mapzi2), tf.maximum(f_mapzi3 ,f_mapzi4), tf.maximum(f_mapzi5 ,f_mapzi6), tf.maximum(f_mapzi7 ,f_mapzi8), tf.maximum(f_mapzi9,f_mapzi10) ],axis = 2), [1,10000*5*64])), keep_prob=keep_prob)

B = tf.concat([f_map,f_map_i], axis =1)
def dense_batch_relu(x, phase, scope,units):
    with tf.variable_scope(scope):
        h1 = tf.contrib.layers.fully_connected(x, units, 
                                               activation_fn=None,
                                               scope='dense')
        h2 = tf.contrib.layers.batch_norm(h1, 
                                          center=True, scale=True, 
                                          is_training=phase,
                                          scope='bn')
        return tf.nn.relu(h2, 'relu')


phase = tf.placeholder(tf.bool, name='phase')


# print(B)
#estimate_1 = tf.matmul(tf.transpose(u, perm=[0, 2, 1]),r_temp)
init_sigma = 0.01

#layer_1 = tf.concat([tf.reshape(B, [1,90]),f_11,f_12,f_13], axis = 1)
layer_1 = tf.nn.dropout(tf.nn.relu(tf.reshape(B, [1, 64*10000*10])),keep_prob=keep_prob)

#layer_1 = tf.concat([f_11,f_12,f_13], axis = 1)


#layer_2 = dense_batch_relu(layer_1,phase,"layer1", 200)
layer_2 = tf.layers.dense(layer_1, 10)

layer_2_output = layer_2


y_pred = tf.squeeze(tf.argmax(layer_2_output,axis = 1))

#y_pred = tf.argmax(layer_3_output, axis=1)


y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")
T = tf.one_hot(y, depth=10, name="T")

loss = tf.losses.softmax_cross_entropy(T, layer_2_output)
#loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")


trainable_vars = tf.trainable_variables()

rot_vars = [var for var in trainable_vars if 'a_' in var.name]
caps_vars =  [var for var in trainable_vars if 'a_' not  in var.name]

#names = [n.name for n in tf.get_default_graph().as_graph_def().node]


batch = tf.Variable(0, trainable = False)

learning_rate = tf.train.exponential_decay(
  0.01,                # Base learning rate.
  batch,  # Current index into the dataset.
  500,          # Decay step.
  0.95,                # Decay rate.
  staircase=True)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
grads = optimizer.compute_gradients(loss, var_list = rot_vars)
training_op = optimizer.minimize(loss, name="training_op", global_step=batch) #,  var_list = caps_vars)

assign_op = batch.assign(1)

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

training_files = '/media/ram095/329CCC2B9CCBE785/datasets/ModelNet10/train/'
testing_files = '/media/ram095/329CCC2B9CCBE785/datasets/ModelNet10/test/'

#saver.restore(sess, "./model.ckpt")
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
pre_acc_val = 0.75

#sess.run(assign_op)
#sess.run(zero_ops)
for j in range(10):
	loss_train_vals = []
	random.shuffle(file_list)

	for filename in file_list:
		
	#	main_itr = main_itr+1
	#	print(filename)
		f = open(filename, 'r')
	
	
		points_raw_, y_annot = read_datapoint(f, filename)
	#	print(y_annot)	
		#sorted_idx = np.lexsort(points_raw.T)
		#sorted_data =  points_raw[sorted_idx,:
		points_raw =np.vstack(set(map(tuple, points_raw_))) 
#		test  = sess.run([layer_3_output ], feed_dict = {y:y_annot, raw_points_init:points_raw, phase:True, keep_prob :0.6})
#		print(test)
	#	print(caps2_output)
	#	test  = sess.run([centered_calib_points_one_temp ], feed_dict = {y:y_annot, raw_points_init:points})
	#	print(test)			
	#	test  = sess.run([centered_calib_points_two_temp ], feed_dict = {y:y_annot, raw_points_init:points})
         #       print(test)
	#	test  = sess.run([centered_calib_points_three_temp ], feed_dict = {y:y_annot, raw_points_init:points})
         #       print(test)
	
	#	print(caps2_output)
         #       test  = sess.run([centered_calib_points_one_t ], feed_dict = {y:y_annot, raw_points_init:points})
          #      print(test)
           #     test  = sess.run([centered_calib_points_two_t ], feed_dict = {y:y_annot, raw_points_init:points})
            #    print(test)
             #   test  = sess.run([centered_calib_points_three_t ], feed_dict = {y:y_annot, raw_points_init:points})
              #  print(test)
	
	        if points_raw.size > 300 and points_raw.size <  1000000:	
		#sess.run([accum_ops], feed_dict = {y:y_annot, raw_points_init:points})
		#	test  = sess.run([weighted_sum_round_2], feed_dict = {y:y_annot, raw_points_init:points_raw})
			test  = sess.run([loss ], feed_dict = {y:y_annot, raw_points_init:points_raw, phase:True, keep_prob :0.5})
          	 	print(test)
			
#			test1  = sess.run([res_2_4i], feed_dict = {y:y_annot, raw_points_init:points_raw})
 #                       print(test1)
#			print(filename)
			if not np.isnan(test).any() and np.less(test, 1000) and np.greater(test, 0.0): 
				main_itr = main_itr+1
	                	print(filename)
				print(main_itr)
				loss_train, _ = sess.run([loss,training_op], feed_dict = {y:y_annot, raw_points_init:points_raw, phase:True, keep_prob :0.6})
				print(j)
                		loss_train_vals.append(loss_train)
                		print(np.mean(loss_train_vals))
			#print(sess.run([batch]))
		if False:# main_itr % 20 == 0:
			#sess.run([grad_mean], feed_dict = {y:y_annot, raw_points_init:points})
			#sess.run(train_step)
			#sess.run(zero_ops)
							
			loss_train = sess.run([loss], feed_dict = {y:y_annot, raw_points_init:points})
                	
			print(j)
                	loss_train_vals.append(loss_train)
                	print(np.mean(loss_train_vals))
#print(gradient.shape)
				#gradients = np.array([])

#		points = points_raw	
# Get unique row mask
		#row_mask = np.append([True],np.any(np.diff(sorted_data,axis=0),1))
		#points = sorted_data[row_mask]
		#points = np.unique(points_raw, axis=0)
		#print(points)
#		points_ = sess.run(caps1_raw_temp, feed_dict = {y:y_annot, raw_points_init:points})
	#	print(np.mean(points_))

	#	points_ = sess.run(X_cal_2, feed_dict = {y:y_annot, raw_points_init:points})
         #       print(np.mean(points_))
	#	points_ = sess.run(X_cal_3, feed_dict = {y:y_annot, raw_points_init:points})
              #  print(np.mean(points_))

	#	points_ = sess.run(X_cal_1, feed_dict = {y:y_annot, raw_points_init:points})
 #               print(points_)
  #              points_ = sess.run(caps1_output, feed_dict = {y:y_annot, raw_points_init:points})
   #             print(points_)
           #     points_ = sess.run(X_cal_3, feed_dict = {y:y_annot, raw_points_init:points})
            #    print(points_)
	#	points_ = sess.run(X__1, feed_dict = {y:y_annot, raw_points_init:points})
         #       print(points_)
          #      points_ = sess.run(X__2, feed_dict = {y:y_annot, raw_points_init:points})
           #     print(points_)
            #    points_ = sess.run(X__3, feed_dict = {y:y_annot, raw_points_init:points})
             #   print(points_)



		#print(j)
		#print("outer_loop")
		#for k in range(5):
		#	_, rot_loss_ = sess.run([training_op_2, rot_loss], feed_dict = {y:y_annot, raw_points_init:points})
		#	print("loss rot")
		#	print(rot_loss_)
	#	oints_ = sess.run(r, feed_dict = {y:y_annot, raw_points_init:points})
                #print(np.isnan(points_).any())

		#_, loss_train = sess.run([training_op, loss], feed_dict = {y:y_annot, raw_points_init:points})
		#print(loss_train)
	
		#loss_train_vals.append(loss_train)
		#print(np.mean(loss_train_vals))
	#	nput("Press Enter to continue...")
		if main_itr % 1500 == 0:
                      #	saver.save(sess, "./model.ckpt")
			for filename in glob.glob(os.path.join(testing_files, '*.off')):
                		
               			f = open(filename, 'r')
               			print(filename)
               			file_names.append(filename)
               			points_raw, y_annot = read_datapoint(f, filename)
               			points =np.vstack(set(map(tuple, points_raw)))
				idx = np.random.randint(int(points.size/3.0), size= int(points.size/3.0 * 0.95)) 
#				points = points_raw
				
				print(int(points.size/3.0))
				print(int(points.size/3.0 * 0.95))
				points_ = points[idx,:]
				if (points.size > 300):
					rot_mat1 =  [[1, 0, 0], [0, np.cos(random.uniform(-2, 2)), -np.sin(random.uniform(-2, 2))], [0, np.sin(random.uniform(-2, 2)), np.cos(random.uniform(-2, 2))] ]
					rot_mat2 =  [[np.cos(random.uniform(-2, 2)), 0, np.sin(random.uniform(-2, 2))], [0,1,0], [np.cos(random.uniform(-2, 2)), -np.sin(random.uniform(-2, 2)), 0]]
					rot_mat3 = [[np.cos(random.uniform(-2, 2)), -np.sin(random.uniform(-2, 2)),0], [np.sin(random.uniform(-2, 2)), np.cos(random.uniform(-2, 2)), 0], [0,0,1]]
					#rot_points =np.matmul(np.matmul(np.matmul(points, rot_mat1), rot_mat2),rot_mat3)
				#	print(rot_points)
					rot_points = np.matmul(points, rot_mat1)
					loss_val, acc_val = sess.run(
                	        	[loss, accuracy], feed_dict = {y:y_annot, raw_points_init:points,phase:False,  keep_prob :1.0})
                			loss_vals.append(loss_val)
                			acc_vals.append(acc_val)
					print("validation")
                			print(acc_val)
				
					acc_val = np.mean(acc_vals)
		               		print(acc_val)
			if acc_val > pre_acc_val:
				print("saving best model")
				saver.save(sess, "./model.ckpt")
				pre_acc_val = acc_val
			print(pre_acc_val)
			loss_vals = []
			acc_vals = []

  #print(t)	
		if False:# main_itr % 10==0:
			for filename in file_names:
				print("inner_loop")
				f = open(filename, 'r')
				points_raw, y_annot = read_datapoint(f, filename)

                #sorted_idx = np.lexsort(points_raw.T)
                #sorted_data =  points_raw[sorted_idx,:]
                		l = points_raw.tolist()
                		l.sort()
                		unique = [x for i, x in enumerate(l) if not i or x != l[i-1]]
               			points  = np.asarray(unique)
				points =np.vstack(set(map(tuple, points_raw)))


				_, loss_train = sess.run([training_op, loss], feed_dict = {y:y_annot, raw_points_init:points})
				print(loss_train)
		#saver.save(sess, "./model.ckpt")
			file_names = []
		#saver.save(sess, "./model.ckpt")

		#loss_val, acc_val = sess.run(
	        #           [loss, accuracy], feed_dict = {y:y_annot, raw_points_init:points})
		#loss_vals.append(loss_val)
	       	#acc_vals.append(acc_val)	
		#print(acc_val)

		#acc_val = np.mean(acc_vals)
		#print(acc_val)
#	saver.save(sess, "./model.ckpt")  
  
#saver.save(sess, "./model.ckpt")

# for itr in xrange(100000000):
#             # train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
            
            
#             #sess.run(train,feed_dict = {rotation_matrix_one:[[1, 0, 0], [0, 1.0/math.sqrt(2), -1.0/math.sqrt(2)], [0, 1.0/math.sqrt(2), 1.0/math.sqrt(2)]], rotation_matrix_two:[[1, 0, 0], [0, 1.0/2, -math.sqrt(3)/2], [0, math.sqrt(3)/2, 1.0/2]], rotation_matrix_three:[[1, 0, 0], [0, math.sqrt(3)/2, -1.0/2], [0, 1.0/2, math.sqrt(3)/2]], raw_points_init:points})

#             if itr % 1000 == 0:
#                 train_loss = sess.run([loss_estimate], feed_dict = {rotation_matrix_one:[[1, 0, 0], [0, 1.0/math.sqrt(2), -1.0/math.sqrt(2)], [0, 1.0/math.sqrt(2), 1.0/math.sqrt(2)]], rotation_matrix_two:[[1, 0, 0], [0, 1.0/2, -math.sqrt(3)/2], [0, math.sqrt(3)/2, 1.0/2]], rotation_matrix_three:[[1, 0, 0], [0, math.sqrt(3)/2, -1.0/2], [0, 1.0/2, math.sqrt(3)/2]], raw_points_init:points})
#                 print(train_loss)
                

