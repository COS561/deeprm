import numpy as np
import matplotlib.pyplot as plt

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