import threading
import datetime
import os
import psycopg2
import heapq
import time


############################################################################
#### aim: to measure the 'explain' time under different concurrency
#### param: QUERY_NUM: the batch of queries to be explained together\
####        THREAD_NUM: concurrency
#### environment: postgreSQL + tpch benchmark            
############################################################################


QUERY_NUM = 22
THREAD_NUM = 4
q_path = '/home/xuanhe/benchmark/tpch/tpch_2_17_3/dbgen/queries/' 
#q_path = '/home/zxh/Desktop/neural-network-knob/benchmark/join-order-benchmark/'

H = []                         # the heap of query ids
for i in range(QUERY_NUM):
    H.append(i+1)


class database():
    def __init__(self):
        self.conn = psycopg2.connect("dbname=tpch user=postgres password=postgres")
        self.cur = self.conn.cursor()

def thread_job(sql_id):

    f_name = str(sql_id)+".sql"
    q_fs = open(q_path+f_name, 'r')
    sql = q_fs.read()
    
    db = database()    
    db.cur.execute("explain "+sql)		# the "create view ..." queries may fail
    db.conn.commit()
    db.conn.close()				# it's necesary to close the cursor for each explain    
    
# batch executing queries
def func_batch():
    td = []
    for i in range(THREAD_NUM):
        if len(H)!=0:
            ind = heapq.heappop(H)
            t = threading.Thread(target=thread_job, args=(ind+1,))
            td.append(t)
        else:
            break

    for t in td:
        t.start()
        t.join()				# main exits until all the threads finish
    

if __name__  == "__main__":

    for i in range(1,10):
        THREAD_NUM = i+1
    
        s_time = datetime.datetime.now()
        while len(H)!=0:    
            func_batch()
                    
        print("End[%d]:%s"%(THREAD_NUM, str(datetime.datetime.now()-s_time)))


