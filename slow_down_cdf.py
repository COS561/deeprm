import numpy as np
import cPickle
import matplotlib.pyplot as plt

import environment
import parameters
import pg_network
import other_agents

import copy

def discount(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    out = np.zeros(len(x))
    out[-1] = x[-1]
    for i in reversed(xrange(len(x)-1)):
        out[i] = x[i] + gamma*out[i+1]
    assert x.ndim >= 1
    # More efficient version:
    # scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
    return out


def categorical_sample(prob_n):
    """
    Sample from categorical distribution,
    specified by a vector of class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np.random.rand()).argmax()


def get_traj(test_type, pa, env, episode_max_length, pg_resume=None, render=False):
    """
    Run agent-environment loop for one whole episode (trajectory)
    Return dictionary of results
    """

    if test_type == 'PG':  # load trained parameters

        pg_learner = pg_network.PGLearner(pa)

        net_handle = open(pg_resume, 'rb')
        net_params = cPickle.load(net_handle)
        pg_learner.set_net_params(net_params)

    env.reset()
    rews = []

    ob = env.observe()

    for _ in xrange(episode_max_length):

        if test_type == 'PG':
            a = pg_learner.choose_action(ob)

        elif test_type == 'Tetris':
            a = other_agents.get_packer_action(env.machine, env.job_slot)

        elif test_type == 'SJF':
            a = other_agents.get_sjf_action(env.machine, env.job_slot)

        elif test_type == 'Random':
            a = other_agents.get_random_action(env.job_slot)

        ob, rew, done, info = env.step(a, repeat=True)

        rews.append(rew)

        if done: break
        if render: env.render()
        # env.render()

    return np.array(rews), info


def get_traj_halluc(test_type, pa, env, episode_max_length, pg_resume=None, render=False):
    """
    Run agent-environment loop for one whole episode (trajectory)
    Return dictionary of results
    """

    if test_type == 'PG':  # load trained parameters

        pg_learner = pg_network.PGLearner(pa)

        net_handle = open(pg_resume, 'rb')
        net_params = cPickle.load(net_handle)
        pg_learner.set_net_params(net_params)

    env.reset()
    rews = []

    rnn_tmp = env.rnn

    for te in xrange(episode_max_length):

        env.rnn = None
        ori_env = copy.deepcopy(env)
        actions = []
        future = min(episode_max_length - te, pa.simu_len)
        rews_hals = np.zeros((pa.num_hal, future), dtype=float)

        if pa.rnn:
            rnn_tmp.forecast_from_history()

        for h in range(pa.num_hal):
            new_env = copy.deepcopy(ori_env)
            new_env.rnn = rnn_tmp
            ob = new_env.observe()

            for th in range(future):

                if test_type == 'PG':
                    a = pg_learner.choose_action(ob)

                elif test_type == 'Tetris':
                    a = other_agents.get_packer_action(new_env.machine, new_env.job_slot)

                elif test_type == 'SJF':
                    a = other_agents.get_sjf_action(new_env.machine, new_env.job_slot)

                elif test_type == 'Random':
                    a = other_agents.get_random_action(new_env.job_slot)

                if th == 0:
                    actions.append(a)

                if not pa.rnn:
                    ob, rew, done, info = new_env.step(a, repeat=True)
                else:
                    ob, rew, done, info = new_env.forecast_step(a, repeat=True)


                if done: break

                rews_hals[h][th] = rew
        
        sum_rews = rews_hals.sum(axis=1, dtype=float)

        a_best = actions[np.argmax(sum_rews)]
        working_env = copy.deepcopy(ori_env)
        working_env.rnn = rnn_tmp

        if pa.rnn:
            ob, rew, done, info, new_job_list = working_env.step(a_best, repeat=True, return_raw_jobs=True)

            for new_job in new_job_list:
                working_env.rnn.update_history(new_job)

        else:
            ob, rew, done, info = working_env.step(a_best, repeat=True)

        rews.append(rew)

        if done: break
        if render: working_env.render()
        # env.render()

    env.rnn = rnn_tmp
    return np.array(rews), info


def launch(pa, pg_resume=None, render=False, plot=False, repre='image', end='no_new_job'):

    # ---- Parameters ----

    test_types = ['Tetris', 'SJF', 'Random']

    if pg_resume is not None:
        test_types = ['PG'] + test_types

    env = environment.Env(pa, render, repre=repre, end=end)

    all_discount_rews = {}
    jobs_slow_down = {}
    work_complete = {}
    work_remain = {}
    job_len_remain = {}
    num_job_remain = {}
    job_remain_delay = {}

    for test_type in test_types:
        all_discount_rews[test_type] = []
        jobs_slow_down[test_type] = []
        work_complete[test_type] = []
        work_remain[test_type] = []
        job_len_remain[test_type] = []
        num_job_remain[test_type] = []
        job_remain_delay[test_type] = []

    for seq_idx in xrange(pa.num_ex):
        print('\n\n')
        print("=============== " + str(seq_idx) + " ===============")

        test_types = ['PG']
        for test_type in test_types:

            print "Regular version"
            rews, info = get_traj(test_type, pa, env, pa.episode_max_length, pg_resume)

            print "---------- " + test_type + " -----------"

            print "total discount reward : \t %s" % (discount(rews, pa.discount)[0])

            print " "
            print "Hallucinated version"
            rews, info = get_traj_halluc(test_type, pa, env, pa.episode_max_length, pg_resume)

            print "---------- " + test_type + " -----------"

            print "total discount reward : \t %s" % (discount(rews, pa.discount)[0])




            all_discount_rews[test_type].append(
                discount(rews, pa.discount)[0]
            )

            # ------------------------
            # ---- per job stat ----
            # ------------------------

            enter_time = np.array([info.record[i].enter_time for i in xrange(len(info.record))])
            finish_time = np.array([info.record[i].finish_time for i in xrange(len(info.record))])
            job_len = np.array([info.record[i].len for i in xrange(len(info.record))])
            job_total_size = np.array([np.sum(info.record[i].res_vec) for i in xrange(len(info.record))])

            finished_idx = (finish_time >= 0)
            unfinished_idx = (finish_time < 0)

            jobs_slow_down[test_type].append(
                (finish_time[finished_idx] - enter_time[finished_idx]) / job_len[finished_idx]
            )
            work_complete[test_type].append(
                np.sum(job_len[finished_idx] * job_total_size[finished_idx])
            )
            work_remain[test_type].append(
                np.sum(job_len[unfinished_idx] * job_total_size[unfinished_idx])
            )
            job_len_remain[test_type].append(
                np.sum(job_len[unfinished_idx])
            )
            num_job_remain[test_type].append(
                len(job_len[unfinished_idx])
            )
            job_remain_delay[test_type].append(
                np.sum(pa.episode_max_length - enter_time[unfinished_idx])
            )

        env.seq_no = (env.seq_no + 1) % env.pa.num_ex

    # -- matplotlib colormap no overlap --
    if plot:
        num_colors = len(test_types)
        cm = plt.get_cmap('gist_rainbow')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_color_cycle([cm(1. * i / num_colors) for i in range(num_colors)])

        for test_type in test_types:
            slow_down_cdf = np.sort(np.concatenate(jobs_slow_down[test_type]))
            slow_down_yvals = np.arange(len(slow_down_cdf))/float(len(slow_down_cdf))
            ax.plot(slow_down_cdf, slow_down_yvals, linewidth=2, label=test_type)

        plt.legend(loc=4)
        plt.xlabel("job slowdown", fontsize=20)
        plt.ylabel("CDF", fontsize=20)
        # plt.show()
        plt.savefig(pg_resume + "_slowdown_fig" + ".pdf")

    return all_discount_rews, jobs_slow_down


def main():
    pa = parameters.Parameters()

    pa.simu_len = 200  # 5000  # 1000
    pa.num_ex = 10  # 100
    pa.num_nw = 10
    pa.num_seq_per_batch = 20
    # pa.max_nw_size = 5
    # pa.job_len = 5
    pa.new_job_rate = 0.3
    pa.discount = 1

    pa.episode_max_length = 20000  # 2000

    pa.compute_dependent_parameters()

    render = False

    plot = True  # plot slowdown cdf

    pg_resume = None
    pg_resume = 'data/pg_re_discount_1_rate_0.3_simu_len_200_num_seq_per_batch_20_ex_10_nw_10_1450.pkl'
    # pg_resume = 'data/pg_re_1000_discount_1_5990.pkl'

    pa.unseen = True

    launch(pa, pg_resume, render, plot, repre='image', end='all_done')


if __name__ == '__main__':
    main()
