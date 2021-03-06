import numpy as np
import psycopg2
import os.path
import time
import datetime

# size = 38
vector_dict = {'ProjectSet':0,
               'ModifyTable':1,
               'Append':2,'Merge Append':3,
               'Recursive Union':4, 
               'BitmapAnd':5,'BitmapOr':6,
               'Nested Loop':7, 
               'Merge Join':8, 'Hash Join':9, 'Seq Scan':10, 'Sample Scan':11,
               'Gather':12, 'Gather Merge':13,
               'Index Scan':14, 'Index Only Scan':15, 'Bitmap Index Scan':16, 'Bitmap Heap Scan':17, 'Tid Scan':18,'Subquery Scan':19, 'Function Scan':20, 'Table Function Scan':21, 'Values Scan':22, 'CTE Scan':23, 'Named Tuplestore Scan':24, 'WorkTable Scan':25, 'Foreign Scan':26, 'Custom Scan':27, 
               'Materialize':28,
               'Sort':29, 
               'Group':30,
               'Aggregate':31,
               'WindowAgg':32, 
               'Unique':33, 
               'SetOp':34,
               'LockRows':35,
               'Limit':36,
               'Hash':37, 'TotalCost':38}

V_LEN = len(vector_dict)
sql_vector = [0]*V_LEN
REPEAT_TIME = 2
LASTOP = 38
q_path = '/home/xuanhe/benchmark/tpch/tpch_2_17_3/dbgen/queries/'  # the path to sql statements
v_path = q_path  # the path to generated feature vectors
t_path = q_path
vfs = open(v_path+'sql_vector.txt', 'a')
tfs = open(t_path+"sql_time.txt", 'a')

class Database:
    def __init__(self):
        self.connection = psycopg2.connect("dbname=tpch user=postgres password=postgres")                                                                                   #####
        # 26
        self.var_names = ['deadlock_timeout', 'effective_cache_size', 'effective_io_concurrency', 'from_collapse_limit',\
                          'geqo_threshold', 'geqo_selection_bias', 'geqo_pool_size', 'geqo_effort', 'geqo_generations',\
                          'join_collapse_limit', 'lock_timeout', 'maintenance_work_mem', 'max_stack_depth', 'statement_timeout',\
                          'temp_buffers', 'temp_file_limit', 'vacuum_cost_delay', 'vacuum_cost_limit', 'vacuum_cost_page_dirty',\
                          'vacuum_cost_page_hit', 'vacuum_cost_page_miss', 'vacuum_freeze_min_age', 'vacuum_freeze_table_age',\
                          'vacuum_multixact_freeze_min_age', 'vacuum_multixact_freeze_table_age', 'work_mem'
                          ]
        self.chosen_action = '('
        for i, a in enumerate(self.var_names):
            if i > 0:
                self.chosen_action = self.chosen_action + ','
            self.chosen_action = self.chosen_action + '\'' + a + '\''
        self.chosen_action = self.chosen_action + ')'


        # size = 16
        self.state_names = ["numbackends", "xact_commit", "xact_rollback", "blks_read", "blks_hit", "tup_returned", "tup_fetched", "tup_inserted", "tup_updated",\
                            "tup_deleted", "conflicts", "temp_files", "temp_bytes", "deadlocks", "blk_read_time", "blk_write_time"]

        self.knob_num = len(self.var_names)
        self.internal_metric_num = len(self.state_names)
        self.external_metric_num = 2  # [throughput, latency]

        with self.connection.cursor() as cursor:
            cursor.execute("set effective_cache_size=1534288")
            self.connection.commit()	
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
            sql = "select * from pg_stat_database where datname='tpch'"
            cursor.execute(sql)
            result = cursor.fetchall()
            state_list = np.array([])
            for i,s in enumerate(result[0]):
                if i>1 and i!=18 and i!=21:
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


def search_plan(dict_p, head):
    global sql_vector

    op_name = dict_p['Node Type']
    op_id = vector_dict[op_name]
    # print("### Op %d" % op_id)
    cost = int(dict_p['Total Cost']) - int(dict_p['Startup Cost'])
    sql_vector[op_id] += cost
    if head == 1:
        sql_vector[LASTOP] = int(dict_p['Total Cost'])		
        head = 0	

    if 'Plans' in dict_p.keys():
        for d in dict_p['Plans']:
            search_plan(d, head)
 
    return head

def sql2v(sql):
    conn = psycopg2.connect("dbname=tpch user=postgres password=postgres")
    with conn.cursor() as cursor:
        cursor.execute('explain (format json) ' + sql)
        res = cursor
        head = 1
        for s in res:
            plan = s[0][0]
            head = search_plan(plan['Plan'], head)
            # print(sql_vector)                                                                   ### Write into file
            for s in sql_vector:
                vfs.write(str(s)+'\t')


    #	print(res)
    conn.close()

db = Database()

for id in range(22):

	s1 = db.fetch_internal_metrics()

	f_name = str(id+1)+".sql"
	q_fs = open(f_name, 'r')
	sql = q_fs.read()
	vfs.write("[sql] %d\t"%(id+1))
	sql2v(sql)

	t1 = datetime.datetime.now()
	with db.connection.cursor() as cursor:
		cursor.execute(open(str(id+1)+".sql", "r").read())
	t2 = datetime.datetime.now()
	print("[ok] %d\t"%(id+1)+str(t2-t1))
	tfs.write("[sql] %d\t"%(id+1) + str(t2-t1) + "\n")

	db.close()
	db = Database()
	s2 = db.fetch_internal_metrics()
	
	for s in s2-s1:
		vfs.write(str(s)+'\t')
	vfs.write('\n')

	sql_vector = [0]*V_LEN
	q_fs.close()

vfs.close()
tfs.close()


