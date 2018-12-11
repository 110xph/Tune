import sys
import os
import numpy as np
import tensorflow as tf
import datetime

import keras.backend as K

from environment_pq import Database_pq, Environment
from model import ActorCritic

import time
import datetime

argus = {}
arg_key = ['num_trial', 'cur_op', 'num_event', 'p_r_range', 'p_u_index', 'p_i', 'p_d', 'maxlen_mem',
           'maxlen_predict_mem', 'learning_rate', 'train_min_size']

#q_path = '/home/haowen/join-order-benchmark-master/'
q_path = '/home/zxh/join-order-benchmark/'

##########################################################
#                       Sys options
# num_trial:500
# cur_op:['oltp_point_select.lua', 'select_random_ranges.lua', 'oltp_delete.lua', 'oltp_insert.lua', 'bulk_insert.lua', 'oltp_update_index.lua', 'oltp_update_non_index.lua’, ‘oltp_read_write.lua’]
# num_event:1000
# p_r_range:0.6
# p_u_index:0.2
# p_i:0.1
# p_d:p_i
# maxlen_mem:2000
# maxlen_predict_mem:2000
# learning_rate:0.001
# train_min_size:32	(self.batch_size)
##########################################################

def Help():
    pass

def parse_cmd_args():
    global argus

    print("Args: %d" % len(sys.argv))
    if (len(sys.argv) < 1):
        raise Exception('arguments needed')

    # init
    argus['num_trial'] = 500

    argus['cur_op'] = 'oltp_read_write.lua'
    argus['num_event'] = 1000
    argus['p_r_range'] = 0.6
    argus['p_u_index'] = 0.2
    argus['p_i'] = 0.1
    argus['p_d'] = 0.1

    argus['maxlen_mem'] = 2000
    argus['maxlen_predict_mem'] = 2000
    argus['learning_rate'] = 0.001
    argus['train_min_size'] = 32
    argus['job_sql_name'] = '10a'

    # set
    for i in range(1, len(sys.argv)):
        arg = sys.argv[i].split(':')
        if len(arg) == 2 and arg[0] in arg_key:
            if arg[0] in ['cur_op', 'job_sql_name']:
                argus['cur_op'] = arg[1]
            else:
                argus[arg[0]] = int(arg[1])

        else:
            print("Invalid option!")
            Help()
            sys.exit(1)


def main():
    #    try:
    parse_cmd_args()

    sess = tf.Session()
    K.set_session(sess)
    db = Database_pq()

    env = Environment(db, argus)


    actor_critic = ActorCritic(env, sess, learning_rate=argus['learning_rate'], train_min_size=argus['train_min_size'],
                               size_mem=argus['maxlen_mem'], size_predict_mem=argus['maxlen_predict_mem'])

    num_trials = argus['num_trial']  # ?
    # trial_len  = 500   # ?
    # ntp
    env.preheat()

    # First iteration
    cur_state = env._get_obs()  # np.array      (inner_metric + sql)
    cur_state = cur_state.reshape((1, env.state.shape[0]))

    # action = env.action_space.sample()
    action = env.fetch_action()  # np.array
    action_2 = action.reshape((1, env.action_space.shape[0]))  # for memory
    new_state, reward, done, _ = env.step(action, 0, 1)  # apply the action -> to steady state -> return the reward
    new_state = new_state.reshape((1, env.state.shape[0]))
    reward_np = np.array([reward])

    print("0-shape-")
    print(new_state.shape)
    actor_critic.remember(cur_state, action_2, reward_np, new_state, done)
    actor_critic.train()  # len<32, useless

    cur_state = new_state
    for i in range(6):
        for e in ['a', 'b', 'c', 'd', 'e', 'f']:
            f_name = str(i + 1) + e + '.sql'
            if os.path.isfile(q_path + f_name):
                t1 = datetime.datetime.now()
                print("######### Slow Query %d %s  ###########" %(i, e))

                env.parser.SQL_NAMES = [f_name]
                # env.render()
                cur_state = env._get_obs()
                cur_state = cur_state.reshape((1, env.state.shape[0]))
                t2 = datetime.datetime.now()
                print("Time_fetchState: " + str((t2 - t1).microseconds * (1E-3)) + "ms")

                action, isPredicted = actor_critic.act(cur_state)
                print(action)
                action_2 = action.reshape((1, env.action_space.shape[0]))  # for memory
                # action.tolist()                                          # to execute
                new_state, reward, done, _ = env.step(action, isPredicted, i + 1, st = t2)

                t3 = datetime.datetime.now()
                print("Time_total: " + str((t3 - t1).microseconds * (1E-3)) + "ms")
                # print("Total time: " + str(time.mktime(t2.timetuple()) - time.mktime(t1.timetuple())) + "s")
                print("####################################")

                new_state = new_state.reshape((1, env.state.shape[0]))

                reward_np = np.array([reward])
                print("%d-shape-" % i)

                actor_critic.remember(cur_state, action_2, reward_np, new_state, done)
                actor_critic.train()

                cur_state = new_state
    print("End")
    '''
    except:
        print("<>There is an error!<>")
    finally:
        db.close()
    '''

if __name__ == "__main__":
    main()