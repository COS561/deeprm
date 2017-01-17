import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join
import os
rnn = True

onlyfiles = [f for f in listdir(os.getcwd()) if isfile(join(os.getcwd(), f))]

#if rnn:
slowdowns_rnn = [f for f in onlyfiles if f.endswith("_job_slow_down_rnn.npy")]

slowdowns_rnn = zip([f.strip("_job_slow_down_rnn.npy") for f in slowdowns_rnn],
    [np.load(f).item() for f in slowdowns_rnn])
#else:
slowdowns = [f for f in onlyfiles if f.endswith("_job_slow_down.npy")]

slowdowns = zip([f.strip("_job_slow_down.npy") for f in slowdowns],
               [np.load(f).item() for f in slowdowns])

amps =[]
ratios = []
halluc = []
halluc_rnn = []
reg_rnn = []
reg = []

slowdowns_rnn = [(int(name.split("_")[0]), amp_sds) for name, amp_sds in slowdowns_rnn]

slowdowns = [(int(name.split("_")[0]), amp_sds) for name, amp_sds in slowdowns]

slowdowns_rnn.sort()

slowdowns.sort()

for amp, amp_sds in slowdowns_rnn:
    print(name)
    r = np.mean([np.mean(s) for s in amp_sds['PG', 'Regular']])
    print r, ": regular"
    h = np.mean([np.mean(s) for s in amp_sds['PG', 'Hallucinated']])
    print h, ": halluc"
    print h / r, ": ratio"

    amps.append(amp)
    halluc_rnn.append(h)
    reg_rnn.append(r)
    ratios.append(h / r)


for amp, amp_sds in slowdowns:
    print(name)
    r = np.mean([np.mean(s) for s in amp_sds['PG', 'Regular']])
    print r, ": regular"
    h = np.mean([np.mean(s) for s in amp_sds['PG', 'Hallucinated']])
    print h, ": halluc"
    print h / r, ": ratio"

    #amps.append(amp)
    halluc.append(h)
    reg.append(r)
    ratios.append(h / r)

print(ratios)
print(amps)

plt.figure()
plt.plot(amps, reg, amps, halluc, amps, reg_rnn, amps, halluc_rnn, linewidth=2)
plt.legend(['FFN (GT)', 'FFN + Ground Truth SFP', 'FFN (RNN)', 'FFN + RNN SFP'])
plt.xlabel('Max Job Duration')
plt.ylabel('Mean Job Slowdown')
plt.show()


'''
sd4 = np.load(slowdowns[3])
sd6 = np.load(slowdowns[4])
sd8 = np.load(slowdowns[5])
sd10 = np.load(slowdowns[0])
sd12 = np.load(slowdowns[1])
sd14 = np.load(slowdowns[2])

jobs_slow_down = np.load('jobs_slow_down.npy').item()
#-- matplotlib colormap no overlap --
plot = True
if plot:
    traj_types = ["Regular", "Hallucinated"]
    num_colors = len(traj_types)
    cm = plt.get_cmap('gist_rainbow')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_color_cycle([cm(1. * i / num_colors) for i in range(num_colors)])

    for test_type in ['PG']:
        for traj_type in traj_types:
            slow_down_cdf = np.sort(np.concatenate(jobs_slow_down[test_type, traj_type]))
            slow_down_yvals = np.arange(len(slow_down_cdf))/float(len(slow_down_cdf))
            ax.plot(slow_down_cdf, slow_down_yvals, linewidth=2, label=traj_type)

    plt.legend(loc=4)
    plt.xlabel("job slowdown", fontsize=20)
    plt.ylabel("CDF", fontsize=20)
    # plt.show()
    plt.savefig("blah_slowdown_fig" + ".pdf")
'''