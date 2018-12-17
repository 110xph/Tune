## DATE:2018/12/17
### Prepare
1. How to get the sql statement's excution plan?\
You can use the keyword explain before the sql statement.
Which will show you the explain order of the sql statement.\
Like SQL:\
EXPLAIN SELECT * FROM table WHERE id1 = ? AND id2 = ?;\
The result of this query will tell you the order of the where clause excution.
2. The optimizer is the core part of the MySQL explain. Only the optimizer determined the join order and where order. And these two orders are my research's core.

### 调研主体
1. INDEX 与 NON INDEX 的优先级问题
2. INDEX 内部分布的优先级问题
3. 各种判断操作的优先级问题

### 方法
1. 判断代价的方式：
   1. 统计数据(ANALYZE TABLE <tablename>)
   2. 最后查询消耗(SHOW STATUS LIKE 'last_query_cost';)
2. 判断索引情况和表的情况：
   1. 查询索引情况：(SHOW INDEX FROM <tablename>;)      重要索引信息：cardinality
   2. 查询表的字段情况：(SHOW FULL COLUMNS FROM <tablename>;)
3. 利用explain查询执行顺序和执行信息
4. 利用explian extended后show warnings显示改写后语句
5. select * from information_schema.optimizer_trace\G;显示全部的优化执行过程
6. 添加强制索引FORCE INDEX或忽略索引IGNORE INDEX选项等多种强制性控制
7. show table status;显示数据库的表的基本信息。
8. set profiling=1;开启分析的摘要器。

### 记录信息
1. INDEX统计中的cardinality仅有表中的元素数量相关，重复元素重复计次
2. 最后查询消耗显示数值为上次查询中优化器计划消耗的查询代价。用于计算查询执行顺序使用。
3. 对于show full columns对于键值的判断仅有PRI和MUL和UNI和空值
4. 重要：Analyze table对于数据的分布情况和查询情况进行了统计分析，经过analyze之代价分布会有比较明显的变化。实验过程中直接插入数据时无法获知正确的查询代价，代价仅仅由表的条目数量决定。仅有当经过analyze之后，index的代价会发生改变。此时才能真正体现出数据库优化器的分析能力。在建表结束到初次analyze之前的数据分析都是简单由表的条目数量决定的。最为明显的变化是show index中的cardinality会基于重复条目数量进行更新（降低至不重复条目数量）。
5. 查询优化的过程中，没有建立索引的项不会进行索引优化（这句话是显然的）
6. 之后几项仅对比where中=的判断顺序
   1. 首先，优先执行的顺序与分布相关，字段分布较少的部分会被优先执行。（但是这个分布判断不是非常准确，可以利用特殊构造手段使优化器迷惑，从而无法判断或者难以选择最小的分布项）
   2. 一般情况下，能选择相对分布较小的分布项优先进行判断筛选。
   3. 同时，优先执行的顺序与基数cardinality相关，一般会选择占比cardinality较小的进行执行，也就是说对于范围相同的情况，会优先选择cardinality较大的进行执行
   4. 查询顺序是对于上述二者的权衡，难于确定最优解，仅能得出相对较优的解（优化器自身数据分析视角下的较优解）
   5. 整体来看，优化器优先选择的执行顺序是自身预期的能是数据范围收缩最快的方式进行执行的，它预期自身生成的执行顺序满足了最优化的数据范围下降的速度，即在最开始的数次判断中实现大量的数据筛选工作
7. 整体算法实现的方式：
   1. 选择可能用于优化的key的范围
   2. 利用统计信息（主要是分布信息和cardinality信息）选择keys(set)
   3. 其中搜集信息的过程使用了自定义的hanlder类

### MYSQL优化
1. 子查询优化（MYSQL只能进行如下3项的优化）
   1. where语句优化（一般优化index选择数量和顺序，对于<>和LIKE%%等会执行全表查询（放弃index））
   2. join的弱优化
   3. IN,NOT IN,>ALL,<ALL,=ALL;这些会进行语义转化优化
2. 视图重写：将视图取消并转化为查询语句，对于查询语句结合原本的选择信息同时优化
3. 等价谓词重写：基本上是通过重写建立索引实现优化，共计7条规则
   1. LIKE（单向或无向）规则
   2. BETWEEN-AND规则
   3. IN转OR规则
   4. OR转ANY规则
   5. ALL/ANY转聚集函数（MIN/MAX）
   6. NOT等价重写
   7. OR重写为UNION ALL（并集规则）

### Example

### Quote
#### Useful
1. sql整体执行过程：https://www.cnblogs.com/xpchild/p/3769977.html;https://cloud.tencent.com/developer/article/1103154;https://www.jianshu.com/p/d7665192aaaf;
2. sql优化器源码分析（重点强调optimizer过程：）https://blog.csdn.net/flighting_sky/article/details/11856485（非常有用）
3. 优化理论：https://www.cnblogs.com/chiangchou/p/mysql-5.html（还有6和8）（非常有用，6中介绍了MYSQL的优化方式）
4. 全面调优：https://dbaplus.cn/news-11-687-1.html
5. 各种类型的强制性控制：https://blog.csdn.net/hsd2012/article/details/51526733

### Less Useful
1. sql语句index退化为全表索引的禁忌：https://blog.csdn.net/tzs_1041218129/article/details/70991149
2. 索引简介：https://blog.csdn.net/fly910905/article/details/77466729
3. explain使用方法：https://www.jianshu.com/p/b22fffa09f36；https://juejin.im/post/5a52386d51882573443c852a；
4. 