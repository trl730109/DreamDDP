import matplotlib.pyplot as plt
import glob
import itertools
import numpy as np
import pandas as pd
import numpy as np

markers=['.','x','o','v','^','<','>','1','2','3','4','8','s','p','*']
#markers=[None]
colors = ['b', 'g', 'r', 'm', 'y', 'k', 'orange', 'purple', 'olive']
markeriter = itertools.cycle(markers)
coloriter = itertools.cycle(colors)

def main():

    df = pd.read_csv("results.csv", header=0)
    df = df.sort_values(by = ['msg_size', 'job_num'])
    print(df)
    msg_sizes = df['msg_size'].unique()
    job_nums = df['job_num'].unique()
    node_nums = df['node_num'].unique()

    fig, ax = plt.subplots(figsize = (12, 8))
    #for msg_size in msg_sizes:
    for n in node_nums:
        
        throughput = []
        for job_num in job_nums:
            #latency.append(parse_log("logs/job_n%d_s%d" % (job_num, msg_size)))
            #throughput.append(sum([msg_size / l for l in parse_log("logs/job_n%d_s%d" % (job_num, msg_size))]))
            #latency = df[(df.job_num == job_num) & (df.msg_size == msg_size)].avg_latency
            latency = df[(df.job_num == job_num) & (df.node_num == n)].avg_latency
            #throughput.append(104857600.0 / list(latency)[0])
            throughput.append(np.sum(104857600.0 / np.array(list(latency))))
            #throughput.append(list(latency)[0] / 1e3)

        #plt.plot(job_num, latency)
        #throughput = [msg_size / l for l in latency]
        marker = next(markeriter)
        color = next(coloriter)
        ax.plot(throughput, label="node=%d" % (n), marker=marker, markerfacecolor='none', color=color)

    #marker = markeriter.next()
    #color = coloriter.next()
    #ax.plot(ther_max, label="theoretical Full-Occupied", marker=marker, markerfacecolor='none', color=color)
        
    ax.set_xlabel("Job Number", size=18)
    ax.set_ylabel("Throughput/MB", size=18)
    #ax.set_ylabel("Latency/ms", size=18)
    ax.yaxis.set_tick_params(labelsize=18)
    #ax.set_ylim(top=25)
    ax.set_xticklabels([0, 2, 3, 4, 5, 6, 7, 8], size=18)
    ax.grid(linestyle=':')
    ax.legend(fontsize=18, loc='upper left')
    plt.show()

if __name__ == '__main__':
    main()
