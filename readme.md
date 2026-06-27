对于一个算法类Algorithm，假设由ABC三个method依次执行，总的预算为1000，每次进化只给method_sample=50次sample的机会，种群大小pop_size=5。A,B,C指的是对应模块种群
1. A,B,C <- a0,b0,c0 #a0,b0,c0就是response里初始的method代码和分数
2. 按照eoh进化A，但是初始种群里有a0，当sample数量到method_sample的时候，停止进化，此时种群里会出现一下几种情况：
    a. 只有a0（也就是进化是没有取到合法样本）或者种群个体大于1，但是a0分数最好，那么就保持原本algorithm_str不变，
    b. 种群个体等于5且无a0，就按照当前最好的种群个体去取代algorithm_str对应item，更新全局分数。
    不管是上面哪一种，保留当前种群A1_population
3. 此时相当于A1+B+C，再进化B，和之前eoh进化A一样，结束后保留当前种群B1_population，然后进化C，在这个过程中，分数只会要么变好，要么保持不变。
4. 假设根据分数下一个要进化的是A模块，那么把A1_population中取代algorithm_str的那一个个体分数替换为当前整体框架的分数，其他个体保持不变，然后作为初始种群用eoh进化，进化结束后，按情况去替换
5. 也就是说每次选取新的method的时候，都把该method上一次进化时，种群中用来替换的个体的分数换成当前全局分数，其他的个体不变，再用这个更新过的种群拿入eoh去进化
        