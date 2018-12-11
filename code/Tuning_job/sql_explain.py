import numpy as np
import psycopg2
import os.path

# Size: 38
vector_dict = {'ProjectSet': 0,
               'ModifyTable': 1,

               'Append': 2, 'Merge Append': 3,

               'Recursive Union': 4,
               'BitmapAnd': 5, 'BitmapOr': 6,
               'Nested Loop': 7,
               'Merge Join': 8, 'Hash Join': 9, 'Seq Scan': 10, 'Sample Scan': 11,
               'Gather': 12, 'Gather Merge': 13,
               'Index Scan': 14, 'Index Only Scan': 15, 'Bitmap Index Scan': 16, 'Bitmap Heap Scan': 17, 'Tid Scan': 18,
               'Subquery Scan': 19, 'Function Scan': 20, 'Table Function Scan': 21, 'Values Scan': 22, 'CTE Scan': 23,
               'Named Tuplestore Scan': 24, 'WorkTable Scan': 25, 'Foreign Scan': 26, 'Custom Scan': 27,
               'Materialize': 28,
               'Sort': 29,
               'Group': 30,
               'Aggregate': 31,
               'WindowAgg': 32,
               'Unique': 33,
               'SetOp': 34,
               'LockRows': 35,
               'Limit': 36,
               'Hash': 37}

# project   modify
# union BitmapAnd   BitmapOr
# Nested_loop
# (join) Hash_Join Merge_Join
# (Dataset) append   merge_append
# (scan) Seq_Scan  Sample_scan Index_scan  Index_only_scan Bitmap_index_scan   bitmap_heap_scan    Tid_scan    Subquery_scan   Function_scan   Table_function_scan Value_scan  CTE_scan    Named_tuplestore_scan   WorkTable_scan  Foreign_scan    Custom_scan
# Meterialize
# Sort
# (aggregation) Group Aggregate   WindowAgg
# Unique
# Setop
# LockRows
# Limit
# Hash



repeat_time = 1
v_path = '/home/haowen/join-order-benchmark-master/'  # the path to generated feature vectors
# v_path = '/home/zxh/join-order-benchmark/'  # the path to generated feature vectors

q_path = '/home/haowen/join-order-benchmark-master/'
# q_path = '/home/zxh/join-order-benchmark/'


class sql_explainer:
    def __init__(self, sql_name, db):
        self.db = db
        self.sql_name = sql_name

        self.v_len = len(vector_dict)
        self.sql_vector = [0] * self.v_len

    # CIDR
    def search_plan(self, dict_p):

        op_name = dict_p['Node Type']
        op_id = vector_dict[op_name]
        # print("### Op %d" % op_id)
        cost = int(dict_p['Total Cost']) - int(dict_p['Startup Cost'])
        self.sql_vector[op_id] += cost

        if 'Plans' in dict_p.keys():
            for d in dict_p['Plans']:
                self.search_plan(d)


    def sql2v(self, sql):
        conn = psycopg2.connect("dbname=imdbload user=postgres password=postgres")
        with conn.cursor() as cursor:
            cursor.execute('explain (format json) ' + sql)
            res = cursor
            for s in res:
                plan = s[0][0]
                self.search_plan(plan['Plan'])
                # print(sql_vector)                                                                   ### Write into file

                return self.sql_vector

        #	print(res)
        conn.close()

    def explain(self):
        if os.path.isfile(q_path + self.sql_name):
            self.sql_vector = [1] * self.v_len
            # return self.sql_vector
            q_fs = open(q_path + self.sql_name, 'r')
            sql = q_fs.read()  ### sql
            self.sql2v(sql)
        else:
            print("!!! No such a sql file name!")
            return []