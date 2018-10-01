import tensorflow as tf
import numpy as np

# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder, 
              output_size,
              scope, 
              n_layers=2, 
              size=500, 
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out

class NNDynamicsModel():
    def __init__(self, 
                 env, 
                 n_layers,
                 size, 
                 activation, 
                 output_activation, 
                 normalization,      #The Normalization statistics, the output of the compute_normalization function given (data)
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        """ YOUR CODE HERE """
        """ Note: Be careful about normalization """
        self.env=env
        self.n_layers=n_layers
        self.size=size
        self.activation=activation
        self.output_activation=output_activation
        self.normalization=normalization
        self.batch_size=batch_size
        self.iterations=iterations
        self.learning_rate=learning_rate
        self.sess=sess
        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.shape[0]
        #Placeholders
        #self.normalized_state_action=tf.placeholder(tf.float64, shape=[None,self.ob_dim+self.ac_dim])
        self.input_state = tf.placeholder(shape=(None, self.ob_dim), dtype=tf.float64)
        self.input_act = tf.placeholder(shape=(None, self.ac_dim), dtype=tf.float64)
        self.Unnormalized_state=tf.placeholder(tf.float64,shape=[None,self.ob_dim])
        self.real_next_states=tf.placeholder(tf.float64,shape=[None,self.ob_dim])
        self.target_delta = tf.placeholder(shape=(None, self.ob_dim), dtype=tf.float32)
        #fitting calculations
        self.normalized_delta=build_mlp(tf.concat([self.input_state, self.input_act], axis=1),output_size=self.ob_dim,scope="Unnormalized",n_layers=self.n_layers,size=self.size,activation=self.activation,output_activation=self.output_activation)
        self.Unnormalized_delta=(self.normalized_delta*(self.normalization[3]+1e-8))+self.normalization[2]
        #print(self.Unnormalized_delta.shape)
        
        self.Unnormalized_next_state=self.Unnormalized_delta+self.Unnormalized_state

        #self.actual_delta=((self.real_next_states-self.Unnormalized_state)-self.normalization[2])/(self.normalization[3]+1e-8)

        #Optimization
        self.loss_function=tf.losses.mean_squared_error(labels=self.target_delta,predictions=self.normalized_delta)
        self.update_parameters= tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_function)

        


    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """

        """YOUR CODE HERE """
        states_normalized=(data[0]-self.normalization[0])/(self.normalization[1]+1e-8)
        actions_normalized=(data[1]-self.normalization[4])/(self.normalization[5]+1e-8)
        delta_normalized=((data[2]-data[0])-self.normalization[2])/(self.normalization[3]+1e-8)
        for _ in range(self.iterations):
            iteration_number=int(np.size(data[0],0)/self.batch_size)
            #print(iteration_number,"iterations",np.size(data[0]),"size",data[0].shape)
            #mini-batch operation
            a=0
            b=self.batch_size
            for i in range(0,iteration_number):
                #Unnormalized_n_s=self.predict(data[0],data[1])
                #Problem with making the data inputs to NN 
                #preparation of the data S
                #input_NN=np.concatenate((states_normalized,actions_normalized),axis=1)
                #print(self.sess.run(self.loss_function,feed_dict={self.input_state: states_normalized[i:i+self.batch_size],
                                                                    #self.input_act: actions_normalized[i:i+self.batch_size],
                                                                    #self.Unnormalized_state:(data[0])[i:i+self.batch_size],
                                                                    #self.target_delta:delta_normalized[i:i+self.batch_size]}))
                self.sess.run(self.update_parameters,feed_dict={self.input_state: states_normalized[a:b],
                                                                    self.input_act: actions_normalized[a:b],
                                                                    self.Unnormalized_state:(data[0])[a:b],
                                                                    self.target_delta:delta_normalized[a:b]})

                a=b
                b+=self.batch_size

                if(i==iteration_number-1):
                    self.sess.run(self.update_parameters,feed_dict={self.input_state: states_normalized[a:np.size(data[0],0)],
                                                                    self.input_act: actions_normalized[a:np.size(data[0],0)],
                                                                    self.Unnormalized_state:(data[0])[a:np.size(data[0],0)],
                                                                    self.target_delta:delta_normalized[a:np.size(data[0],0)]})

                


                



        



    def predict(self, states, actions):# ask about if it is just one batch or not
        #I have done here a merging process to the input of the NN to be the combination between states and actions (normalized)
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model """
        """ YOUR CODE HERE """
        states_normalized=(states-self.normalization[0])/(self.normalization[1]+1e-8)
        actions_normalized=(actions-self.normalization[4])/(self.normalization[5]+1e-8)
        #Problem with making the data inputs to NN 
        #preparation of the data 
        #input_NN=np.concatenate((states_normalized,actions_normalized),axis=1)
        return self.sess.run(self.Unnormalized_next_state,feed_dict={self.input_state: states_normalized,
                                                                self.input_act: actions_normalized,
                                                                self.Unnormalized_state:states})


    def predict_(self, states, actions):# ask about if it is just one batch or not
        #I have done here a merging process to the input of the NN to be the combination between states and actions (normalized)
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model """
        #print(len(self.normalization[0]))
        states_normalized=(states-self.normalization[0])/(self.normalization[1]+1e-8)
        actions_normalized=(actions-self.normalization[4])/(self.normalization[5]+1e-8)
        #Problem with making the data inputs to NN 
        #preparation of the data 
        #input_NN=np.concatenate((states_normalized,actions_normalized),axis=1)
        return self.sess.run(self.Unnormalized_next_state,feed_dict={self.input_state: np.reshape(states_normalized,(1,self.ob_dim)),
                                                                self.input_act: np.reshape(actions_normalized,(1,self.ac_dim)),
                                                                self.Unnormalized_state:np.reshape(states,(1,self.ob_dim))})