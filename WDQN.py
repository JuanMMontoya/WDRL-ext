"""
Author: Juan M. Montoya

V1.1
    -Added parameters for using prioritized experience replay
    -add td_errors


"""
import tensorflow as tf


class WDQN:
    def __init__(self, params, scope):
        """
        Creates the Wide Deep Q-Network Agent's model using Tensorflow
        Can also initialize the DQN or Linear Agent's model
        """
        self.scope = scope
        with tf.variable_scope(scope):
            self.x_dqn = tf.placeholder(tf.float32, shape=(None, params['width'], params['height'], params["mat_dim"]),
                                        name="x_dqn")  # DQN
            # DQN
            self.x_lin = tf.placeholder(tf.float32, [None, params["k"], params["features"]],
                                        name="x_lin")  # Linear approximator
            self.qt_dqn = tf.placeholder(tf.float32, shape=(None,), name="G1")
            self.qt_lin = tf.placeholder(tf.float32, shape=(None,), name="G2")
            self.qt_wdqn = tf.placeholder(tf.float32, shape=(None,), name="G2")
            self.actions = tf.placeholder(tf.float32, shape=(None, params["k"]), name="actions")
            self.rewards = tf.placeholder(tf.float32, [None], name='rewards')
            self.terminals = tf.placeholder(tf.float32, [None], name='terminals')
            self.imp_weights = tf.placeholder(tf.float32, [None], name="weights")

            self.rate = tf.placeholder(tf.float32)
            self.discount = tf.constant(params['discount'])

            # Linear Agent
            Z_lin = self.x_lin
            w_lin = tf.Variable(tf.zeros([params["features"], 1], dtype=tf.float32, name="weightsLin"))
            self.y_lin = tf.einsum('ijk,kl->ijl', Z_lin, w_lin)
            self.y_lin = tf.squeeze(self.y_lin, axis=2)

            # Q, td error and cost
            self.yj_lin = tf.add(self.rewards,
                                 tf.multiply(1.0 - self.terminals, tf.multiply(self.discount, self.qt_lin)))
            self.Q_pred_lin = tf.reduce_sum(tf.multiply(self.y_lin, self.actions), reduction_indices=1)
            self.td_error_lin = tf.subtract(self.yj_lin, self.Q_pred_lin)
            self.cost_lin = tf.reduce_sum(tf.pow(self.td_error_lin, 2))

            # ConvNets
            Z_dqn = self.x_dqn
            channels = params["mat_dim"]
            for filters, size, stride in params["conv_layer_sizes"]:
                w = tf.Variable(tf.random_normal([size, size, channels, filters], stddev=0.01))
                b = tf.Variable(tf.constant(0.1, shape=[filters]))
                c = tf.nn.conv2d(Z_dqn, w, strides=[1, stride, stride, 1], padding='SAME')
                Z_dqn = tf.nn.relu(tf.add(c, b))
                Z_dqn = tf.nn.dropout(Z_dqn, self.rate)  # DROPOUT
                channels = filters

            # Fully connected layers
            Z_dqn = tf.contrib.layers.flatten(Z_dqn)
            _, dim = Z_dqn.shape
            dim = int(dim)
            for hidden in params["hidden_layer_sizes"]:
                w = tf.Variable(tf.random_normal([dim, hidden], stddev=0.01))
                b = tf.Variable(tf.constant(0.1, shape=[hidden]))
                mult = tf.add(tf.matmul(Z_dqn, w), b)
                Z_dqn = tf.nn.relu(mult)
                Z_dqn = tf.nn.dropout(Z_dqn, self.rate)  # DROPOUT
                dim = hidden

            # Final output layer
            w_dqn = tf.Variable(tf.random_normal([dim, params["k"]], stddev=0.01))
            b = tf.Variable(tf.constant(0.1, shape=[params["k"]]))
            self.y_dqn = tf.add(tf.matmul(Z_dqn, w_dqn), b)

            # Q, td error and cost
            self.yj_dqn = tf.add(self.rewards,
                                 tf.multiply(1.0 - self.terminals, tf.multiply(self.discount, self.qt_dqn)))
            self.Q_pred_dqn = tf.reduce_sum(tf.multiply(self.y_dqn, self.actions), reduction_indices=1)
            self.td_error_dqn = tf.subtract(self.yj_dqn, self.Q_pred_dqn)
            self.cost_dqn = tf.reduce_sum(tf.multiply(tf.pow(self.td_error_dqn, 2), self.imp_weights))

            if params['global_step_lin'] is not None:
                self.global_step_lin = tf.Variable(int(params['global_step_lin']), name='global_step_lin',
                                                   trainable=False)
            else:
                self.global_step_lin = tf.Variable(0, name='global_step_lin', trainable=False)

            if params['global_step_dqn'] is not None:
                self.global_step_dqn = tf.Variable(int(params['global_step_dqn']), name='global_step_dqn',
                                                   trainable=False)
            else:
                self.global_step_dqn = tf.Variable(0, name='global_step_dqn', trainable=False)

            cond = tf.constant(int(params["dcy_lrl"]))
            self.lr_lin = tf.cond(cond > 0, lambda: tf.train.exponential_decay(params["lr_lin"], self.global_step_lin,
                                                                               params["dcy_lrl_val"]["dcy_stp"],
                                                                               params["dcy_lrl_val"]["dcy_rt"],
                                                                               staircase=True),
                                  lambda: params["lr_lin"])

            # Optimization for linear and DQN model
            self.optim_lin = tf.train.GradientDescentOptimizer(self.lr_lin).minimize(self.cost_lin,
                                                                                     global_step=self.global_step_lin)

            self.optim_dqn = tf.train.AdamOptimizer(params['lr_dqn']).minimize(self.cost_dqn,
                                                                               global_step=self.global_step_dqn)

            # Prediction for Wide Deep Q-Network Agent and td error
            self.y_wdqn = tf.add(self.y_lin, self.y_dqn)
            self.yj_wdqn = tf.add(self.rewards,
                                  tf.multiply(1.0 - self.terminals, tf.multiply(self.discount, self.qt_wdqn)))
            self.Q_pred_wdqn = tf.reduce_sum(tf.multiply(self.y_wdqn, self.actions), reduction_indices=1)
            self.td_error_wdqn = tf.subtract(self.yj_wdqn, self.Q_pred_wdqn)

    def train(self, bat_s_dqn, bat_s_lin, bat_a, bat_t, qt_dqn, qt_lin, qt_wdqn, bat_r, dropout, only_dqn, only_lin,
              weights):
        """
        Charge of training and calculating cost
        Can also predict purely for DQN or Linear Agent
        """
        # WDQN combines train_dqn, train_lin
        if only_dqn:
            _, cnt_dqn, td_error_dqn = self.train_dqn(bat_s_dqn, bat_a, bat_t, qt_dqn, bat_r, dropout, weights)
            return cnt_dqn, td_error_dqn
        elif only_lin:
            _, cnt_lin, td_error_lin = self.train_lin(bat_s_lin, bat_a, bat_t, qt_lin, bat_r, dropout, weights)
            return cnt_lin, td_error_lin
        else:
            _, cnt_wdqn, td_error_wdqn = self.train_wdqn(bat_s_dqn, bat_s_lin, bat_a, bat_t, qt_dqn, qt_lin, qt_wdqn,
                                                         bat_r, dropout, weights)
            return cnt_wdqn, td_error_wdqn

    def train_dqn(self, bat_s_dqn, bat_a, bat_t, qt_dqn, bat_r, dropout, weights):
        """Carry on the training for DQN """
        feed_dict_dqn = {self.x_dqn: bat_s_dqn,
                         self.qt_dqn: qt_dqn,
                         self.actions: bat_a,
                         self.terminals: bat_t,
                         self.rewards: bat_r,
                         self.rate: dropout,
                         self.imp_weights: weights}
        _, cnt_dqn, td_error_dqn = self.sess.run([self.optim_dqn, self.global_step_dqn, self.td_error_dqn],
                                                 feed_dict=feed_dict_dqn)
        return _, cnt_dqn, td_error_dqn

    def train_wdqn(self, bat_s_dqn, bat_s_lin, bat_a, bat_t, qt_dqn, qt_lin, qt_wdqn, bat_r, dropout, weights):
        """Carry on the training for WDQN """
        feed_dict_wdqn = {self.x_dqn: bat_s_dqn,
                          self.x_lin: bat_s_lin,
                          self.qt_lin: qt_lin,
                          self.qt_dqn: qt_dqn,
                          self.qt_wdqn: qt_wdqn,
                          self.actions: bat_a,
                          self.terminals: bat_t,
                          self.rewards: bat_r,
                          self.rate: dropout,
                          self.imp_weights: weights}
        _, _, cnt_wdqn, td_error_wdqn = self.sess.run(
            [self.optim_dqn, self.optim_lin, self.global_step_dqn, self.td_error_wdqn],
            feed_dict=feed_dict_wdqn)
        return _, cnt_wdqn, td_error_wdqn

    def train_lin(self, bat_s_lin, bat_a, bat_t, qt_lin, bat_r, dropout, weights):
        """Carry on the training for linear model """
        feed_dict_lin = {self.x_lin: bat_s_lin,
                         self.qt_lin: qt_lin,
                         self.actions: bat_a,
                         self.terminals: bat_t,
                         self.rewards: bat_r,
                         self.rate: dropout,
                         self.imp_weights: weights}
        _, cnt_lin, td_error_lin = self.sess.run([self.optim_lin, self.global_step_lin, self.td_error_lin],
                                                 feed_dict=feed_dict_lin)
        return _, cnt_lin, td_error_lin

    def rep_network(self, other):
        """
        Replace parameter of network with giving new Param.
        In charge of substitution between Q-Target Network and Q-Network
        """
        mine = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        mine = sorted(mine, key=lambda v: v.name)
        theirs = [t for t in tf.trainable_variables() if t.name.startswith(other.scope)]
        theirs = sorted(theirs, key=lambda v: v.name)
        ops = []
        for p, q in zip(mine, theirs):
            actual = self.sess.run(q)
            op = p.assign(actual)
            ops.append(op)
        self.sess.run(ops)

    def set_session(self, sess):
        """
        Fix the Tensor Flow session
        """
        self.sess = sess

    def predict_wdqn(self, states_dqn, states_lin, dropout):
        """
        Makes predictions for WDQN
        """
        pred = self.sess.run(self.y_wdqn,
                             feed_dict={self.x_dqn: states_dqn, self.x_lin: states_lin, self.rate: dropout})
        return pred

    def predict_dqn(self, states, dropout):
        """
        Makes DQN-predictions
        """
        pred_dqn = self.sess.run(self.y_dqn,
                                 feed_dict={self.x_dqn: states, self.rate: dropout})
        return pred_dqn

    def predict_lin(self, states, dropout):
        """
        Makes Linear predictions
        """
        pred_dqn = self.sess.run(self.y_lin,
                                 feed_dict={self.x_lin: states, self.rate: dropout})
        return pred_dqn
