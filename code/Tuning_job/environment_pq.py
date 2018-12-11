import sys
import os
import time
import datetime
from os import path
import subprocess
import time
from collections import deque
import numpy as np
import random
import tensorflow as tf
import pandas

import psycopg2

import gym
from gym import spaces
from gym.utils import seeding

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras.backend as K
from keras.layers.normalization import BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler

from sql2resource import SqlParser

# q_path = '/home/haowen/join-order-benchmark-master/'
q_path = '/home/zxh/join-order-benchmark/'
tim = datetime.datetime.now()

class Database_pq:
    def __init__(self):
        self.connection = psycopg2.connect("dbname=imdbload user=postgres password=postgres")

        # Choose actions to be considered 59
        self.var_names = ['commit_delay','commit_siblings','deadlock_timeout', 'effective_cache_size', 'effective_io_concurrency', 'from_collapse_limit',
                          'geqo_threshold', 'geqo_selection_bias', 'geqo_pool_size', 'geqo_effort', 'geqo_generations',
                          'join_collapse_limit', 'lock_timeout', 'maintenance_work_mem', 'max_stack_depth', 'statement_timeout',
                          'temp_buffers', 'temp_file_limit', 'vacuum_cost_delay', 'vacuum_cost_limit', 'vacuum_cost_page_dirty',
                          'vacuum_cost_page_hit', 'vacuum_cost_page_miss', 'vacuum_freeze_min_age', 'vacuum_freeze_table_age',
                          'vacuum_multixact_freeze_min_age', 'vacuum_multixact_freeze_table_age', 'work_mem',
                          # 'default_statistics_target', 'extra_float_digits',
                          ]
        self.chosen_action = '('
        for i, a in enumerate(self.var_names):
            if i > 0:
                self.chosen_action = self.chosen_action + ','
            self.chosen_action = self.chosen_action + '\'' + a + '\''
        self.chosen_action = self.chosen_action + ')'
        self.knob_num = len(self.var_names)


        # State list 14
        self.state_names = ["numbackends", "xact_commit", "xact_rollback", "blks_read", "blks_hit", "tup_returned", "tup_fetched", "tup_inserted", "tup_updated",
                            "tup_deleted", "conflicts", "temp_files", "temp_bytes", "deadlocks",]
        self.internal_metric_num = len(self.state_names)

        # Performance
        self.external_metric_num = 2  # [throughput, latency]

    def close(self):
        self.connection.close()

    def fetch_internal_metrics(self):
        with self.connection.cursor() as cursor:
            ######### observation_space
            #         State_status
            # [lock_row_lock_time_max, lock_row_lock_time_avg, buffer_pool_size,
            # buffer_pool_pages_total, buffer_pool_pages_misc, buffer_pool_pages_data, buffer_pool_bytes_data,
            # buffer_pool_pages_dirty, buffer_pool_bytes_dirty, buffer_pool_pages_free, trx_rseg_history_len,
            # file_num_open_files, innodb_page_size]
            #         Cumulative_status
            # [lock_row_lock_current_waits, ]
            '''
            sql = "select count from INNODB_METRICS where name='lock_row_lock_time_max' or name='lock_row_lock_time_avg'\
            or name='buffer_pool_size' or name='buffer_pool_pages_total' or name='buffer_pool_pages_misc' or name='buffer_pool_pages_data'\
            or name='buffer_pool_bytes_data' or name='buffer_pool_pages_dirty' or name='buffer_pool_bytes_dirty' or name='buffer_pool_pages_free'\
            or name='trx_rseg_history_len' or name='file_num_open_files' or name='innodb_page_size'"
            '''
            sql = "select * from pg_stat_database where datname='postgres'"
            cursor.execute(sql)
            result = cursor.fetchall()
            state_list = np.array([])
            for i,s in enumerate(result[0]):
                if i>1 and i<16:
                    state_list = np.append(state_list, s)

            return state_list


    def fetch_knob(self):
        with self.connection.cursor() as cursor:
            ######### action_space
            #         Common part
            # '''
            # sql = "select @@table_open_cache, @@max_connections, @@innodb_buffer_pool_size, @@innodb_buffer_pool_instances,\
            # @@innodb_log_files_in_group, @@innodb_log_file_size, @@innodb_purge_threads, @@innodb_read_io_threads,\
            # @@innodb_write_io_threads, @@binlog_cache_size"
            # '''

            #         Read-only
            # innodb_buffer_pool_instances innodb_log_files_in_group innodb_log_file_size innodb_purge_threads innodb_read_io_threads
            # innodb_write_io_threads

            #         Extended part
            # innodb_adaptive_max_sleep_delay(0,1000000)    innodb_change_buffer_max_size(0,50) innodb_flush_log_at_timeout(1,2700) innodb_flushing_avg_loops(1,1000)
            # innodb_max_purge_lag(0,4294967295)    innodb_old_blocks_pct(5,95) innodb_read_ahead_threshold(0,64)   innodb_replication_delay(0,4294967295)
            # innodb_rollback_segments(1,128)   innodb_adaptive_flushing_lwm(0,70)   innodb_sync_spin_loops (0,4294967295)
            # innodb_lock_wait_timeout(1,1073741824)    innodb_autoextend_increment(1,1000) innodb_concurrency_tickets(1,4294967295)    innodb_max_dirty_pages_pct(0,99)
            # innodb_max_dirty_pages_pct_lwm(0,99)  innodb_io_capacity(100, 2**32-1)    innodb_lru_scan_depth(100, 2**32-1) innodb_old_blocks_time(0, 2**32-1)
            # innodb_purge_batch_size(1,5000)   innodb_spin_wait_delay(0,2**32-1)

            #        Non-dynmic
            # innodb_sync_array_size    metadata_locks_cache_size   metadata_locks_hash_instances   innodb_log_buffer_size  eq_range_index_dive_limit   max_length_for_sort_data
            # read_rnd_buffer_size  table_open_cache_instances  transaction_prealloc_size   binlog_order_commits    query_cache_limit   query_cache_size    query_cache_type    query_prealloc_size
            # join_buffer_size  tmp_table_size  max_seeks_for_key   query_alloc_block_size  sort_buffer_size    thread_cache_size   max_write_lock_count
            ##############################################################################

            # sql = "select setting, max_val from pg_settings where name in " + chosen_action
            sql = "select setting from pg_settings where name in " + self.chosen_action
            # perform_action = "set " + name

            cursor.execute(sql)
            result = cursor.fetchall()
            knob_list = np.array([])

            for s in result:
                knob_list = np.append(knob_list, s[0])

            return knob_list

    def change_knob_nonrestart(self, actions):
        with self.connection.cursor() as cursor:
            for i in range(self.knob_num):
                sql = 'set %s=%d' % (self.var_names[i], actions[i])
                cursor.execute(sql)
                # result = cursor.fetchall()
            # connection.commit()

EXE_TIME = 1
# Define the environment
class Environment(gym.Env):
    global EXE_TIME

    def __init__(self, db, argus):

        self.db = db

        self.parser = SqlParser(db = db, benchmark='job', cur_op=argus['cur_op'], num_event=argus['num_event'], p_r_range=argus['p_r_range'],
                                p_u_index=argus['p_u_index'], p_i=argus['p_i'], p_d=argus['p_d'])

        # self.state_num = db.internal_metric_num
        self.state_num = 31
        self.action_num = db.knob_num  # 4 in fact

        # state_space
        self.o_low = np.array([-10000000000]*self.state_num)
        self.o_high = np.array([10000000000]*self.state_num)
        self.observation_space = spaces.Box(low=self.o_low, high=self.o_high, dtype=np.float32)

        self.state = db.fetch_internal_metrics()

        self.state = np.append(self.parser.predict_sql_resource(), self.state)


        # lowest -- default
        sql = "select setting from pg_settings where name in " + self.db.chosen_action

        cur = self.db.connection.cursor()
        cur.execute(sql)
        result = cur.fetchall()
        a_low = np.array([])
        for s in result:
            a_low = np.append(a_low, s[0])
        print("A_low's size: %d"%a_low.size)

        # highest
        sql = "select max_val from pg_settings where name in " + self.db.chosen_action
        cur = self.db.connection.cursor()
        cur.execute(sql)
        result = cur.fetchall()
        a_high = np.array([])
        for s in result:
            a_high = np.append(a_high, s[0])

        print("A_high's size: %d"%a_high.size)

        self.a_low = a_low.astype('float64')
        self.a_high = a_high.astype('float64')
        self.action_space = spaces.Box(low=self.a_low, high=self.a_high, dtype=np.float32)
        self.default_action = self.a_low

        self.mem = deque(maxlen=argus['maxlen_mem'])  # [throughput, latency]
        self.predicted_mem = deque(maxlen=argus['maxlen_predict_mem'])

        self.seed()
        self.start_time = datetime.datetime.now()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def preheat(self):
        ifs = open(q_path+"timing_basline_job.txt", 'w')
        with self.db.connection.cursor() as cursor:
            for i in range(33):
                for e in ['a', 'b', 'c', 'd', 'e', 'f']:
                    f_name = str(i + 1) + e + '.sql'
                    if os.path.isfile(q_path+f_name):
                        fd = open(q_path+f_name, 'r')
                        sql = fd.read()
                        for i in range(EXE_TIME):
                            t1 = datetime.datetime.now()
                            cursor.execute(sql)
                            t2 = datetime.datetime.now()
                            ifs.write(str(time.mktime(t2.timetuple()) - time.mktime(t1.timetuple())) + "\n")

        ifs.close()
        print("Preheat Finished")

    def fetch_action(self):
        return self.db.fetch_knob()



    # new_state, reward, done,
    def step(self, u, isPredicted, iteration, st=tim):
        # self.db.change_knob_nonrestart(u)

        # 1 run sysbench
        # primary key lookup
        # cmd = "sysbench /home/zxh/sysbench/src/lua/oltp_read_only.lua --threads=4 --events=0 --time=20 --mysql-host=127.0.0.1 --mysql-user='root' --mysql-password='110xph' --mysql-port=3306 --tables=5 --table-size=1000000 --range_selects=off --db-ps-mode=disable --report-interval=1 --mysql-db='sbtest' run >/home/zxh/fl1 2>&1"
        # cmd = "sysbench /home/zxh/sysbench/src/lua/oltp_write_only.lua --threads=10 --time=30 --events=0 --mysql-host=127.0.0.1 --mysql-user='root' --mysql-password='110xph'\
        # --mysql-port=3306 --tables=10 --table-size=500000 --db-ps-mode=disable --report-interval=10 --mysql-db='sbtest_wo_2' run >/home/zxh/fl1 2>&1"
        # self.parser.cmd
        # p = subprocess.check_output(self.parser.cmd + ' cleanup', shell=True)

        t1 = datetime.datetime.now()
        print("Time_recommend: "+str((t1 - st).microseconds * (1E-3))+ "ms")

        with self.db.connection.cursor() as cursor:
            for sql_name in self.parser.SQL_NAMES:
                fd = open(q_path+sql_name, 'r')
                sql = fd.read()
                for i in range(EXE_TIME):
                    # cursor.execute(sql)
                    pass
        t2 = datetime.datetime.now()

        # latency = float((t2 - t1).microseconds)
        latency = float(time.mktime(t2.timetuple()) - time.mktime(t1.timetuple()))
        # print("Time_execute: "+str(latency))

        if latency == 0:
            latency = 0.1
        throughput = 1000/latency

        # print(str(len(self.mem)+1)+"\t"+str(throughput)+"\t"+str(latency))
        cur_time = datetime.datetime.now()
        interval = (cur_time - self.start_time).seconds
        self.mem.append([throughput, latency])
        # 2 refetch state
        self._get_obs()

        # 3 cul reward(T, L)
        if len(self.mem) != 0:
            dt0 = (throughput - self.mem[0][0]) / self.mem[0][0]
            dt1 = (throughput - self.mem[len(self.mem) - 1][0]) / self.mem[len(self.mem) - 1][0]
            if dt0 >= 0:
                rt = ((1 + dt0) ** 2 - 1) * abs(1 + dt1)
            else:
                rt = -((1 - dt0) ** 2 - 1) * abs(1 - dt1)

            dl0 = -(latency - self.mem[0][1]) / self.mem[0][1]
            dl1 = -(latency - self.mem[len(self.mem) - 1][1]) / self.mem[len(self.mem) - 1][1]
            if dl0 >= 0:
                rl = ((1 + dl0) ** 2 - 1) * abs(1 + dl1)
            else:
                rl = -((1 - dl0) ** 2 - 1) * abs(1 - dl1)

        else:  # initial action
            rt = 0
            rl = 0

        reward = 6 * rl + 4 * rt
        '''
        reward = 0
        for i in range(u.shape[0]):
            tmp = u[i] / self.a_high[i]
            reward+=tmp
        '''

        '''
        print("Performance: %d\t%f\t%f\t%f\t%ds" % (len(self.mem) + 1, throughput, latency, reward, interval))
        if isPredicted:
            self.predicted_mem.append([len(self.predicted_mem), throughput, latency, reward])
            if len(self.predicted_mem)%10 == 0:
                print("Predict List")
                print(self.predicted_mem)
       '''

        if isPredicted:
            self.predicted_mem.append([len(self.predicted_mem), throughput, latency, reward])
            # if len(self.predicted_mem)%10 == 0:
            # print("Predict List")
            # print(self.predicted_mem)
            print("Predict %d\t%f\t%f\t%f\t%ds" % (len(self.mem) + 1, throughput, latency, reward, interval))

            self.pfs = open('rw_predict_2', 'a')
            self.pfs.write("%d\t%f\t%f\n" % (iteration, throughput, latency))
            self.pfs.close()
        else:
            print("Random %d\t%f\t%f\t%f\t%ds" % (len(self.mem) + 1, throughput, latency, reward, interval))

            self.rfs = open('rw_random_2', 'a')
            self.rfs.write("%d\t%f\t%f\n" % (iteration, throughput, latency))
            self.rfs.close()

        return self.state, reward, False, {}

    def _get_obs(self):
        self.state = self.db.fetch_internal_metrics()
        self.state = np.append(self.parser.predict_sql_resource(), self.state)
        return self.state