import numpy as np
import math


class Dist:
    def __init__(self, num_res, max_nw_size, job_len):
        self.num_res = num_res
        self.max_nw_size = max_nw_size
        self.job_len = job_len

        self.job_small_chance = 0.8
        self.job_period = 5

        self.job_len_big_lower = job_len * 2 / 3
        self.job_len_big_upper = job_len

        self.job_len_small_lower = 1
        self.job_len_small_upper = job_len / 5

        self.dominant_res_lower = max_nw_size / 2
        self.dominant_res_upper = max_nw_size

        self.other_res_lower = 1
        self.other_res_upper = max_nw_size / 5

        self.normal = 0
        self.bimodal = 0
        self.periodic = 1
        self.noise = False

        self.switch_chance = 0.5

    def normal_dist(self):

        # new work duration
        nw_len = np.random.randint(1, self.job_len + 1)  # same length in every dimension

        nw_size = np.zeros(self.num_res)

        for i in range(self.num_res):
            nw_size[i] = np.random.randint(1, self.max_nw_size + 1)

        return nw_len, nw_size

    def bi_model_dist(self):

        # -- job length --
        if np.random.rand() < self.job_small_chance:  # small job
            nw_len = np.random.randint(self.job_len_small_lower,
                                       self.job_len_small_upper + 1)
        else:  # big job
            nw_len = np.random.randint(self.job_len_big_lower,
                                       self.job_len_big_upper + 1)

        nw_size = np.zeros(self.num_res)

        # -- job resource request --
        dominant_res = np.random.randint(0, self.num_res)
        for i in range(self.num_res):
            if i == dominant_res:
                nw_size[i] = np.random.randint(self.dominant_res_lower,
                                               self.dominant_res_upper + 1)
            else:
                nw_size[i] = np.random.randint(self.other_res_lower,
                                               self.other_res_upper + 1)

        return nw_len, nw_size


def generate_sequence_work(pa, seed=42):
    np.random.seed(seed)

    simu_len = pa.simu_len * pa.num_ex

    nw_dist = pa.dist.bi_model_dist

    nw_len_seq = np.zeros((pa.num_ex, pa.simu_len), dtype=int)
    nw_size_seq = np.zeros((pa.num_ex, pa.simu_len, pa.num_res), dtype=int)

    if pa.dist.normal:

        for i in range(simu_len):

            if np.random.rand() < pa.new_job_rate:
                nw_len_seq[i], nw_size_seq[i, :] = nw_dist()

        nw_len_seq = np.reshape(nw_len_seq, [pa.num_ex, pa.simu_len])
        nw_size_seq = np.reshape(nw_size_seq, [pa.num_ex, pa.simu_len, pa.num_res])

    else:

        for i in range(pa.num_ex):
            # set parameters of length dist for cycle i:

            if pa.dist.bimodal:

                if np.random.rand() < pa.switch_chance:
                    pa.dist.job_small_chance = 1 - pa.dist.job_small_chance

            elif pa.dist.periodic:
                pa.dist.job_period = np.random.randint(3, 12)
                pa.dist.job_amplitude = np.random.randint(4, 14)
                pa.dist.job_phase = np.random.randint(0, 12)
                pa.dist.size_periods = [pa.dist.job_period + np.random.randint(-2, 2) for _ in range(pa.num_res)]
                pa.dist.size_phases = [np.random.randint(3, 12) for _ in range(pa.num_res)]

            for j in range(pa.simu_len):
                # generate length, size attributes of sequence j in cycle i:

                if np.random.rand() < pa.new_job_rate:

                    if pa.dist.bimodal:

                        nw_len_seq[i, j], nw_size_seq[i, j, :] = pa.dist.bi_model_dist()

                    elif pa.dist.periodic:

                        # nw_len_seq[i,j] = round(4*(math.sin(0.5*j) + math.cos(0.25*j))+8)
                        if pa.dist.noise:
                            offset = np.random.randint(-2, 2)
                        else:
                            offset = 0
                        nw_len_seq[i, j] = round(0.5 * pa.dist.job_amplitude * (math.sin((j + offset + pa.dist.job_phase) / 
                            float(pa.dist.job_period)))) + (0.5 * pa.dist.job_amplitude) + 1

                        for k in range(pa.num_res):
                            if pa.dist.noise:
                                offset = np.random.randint(-2, 2)
                            else:
                                offset = 0
                            nw_size_seq[i, j, k] = round(np.floor(
                                (pa.dist.max_nw_size / 2.0) + (pa.dist.max_nw_size / 2.0) *
                                (math.sin((j + offset + pa.dist.size_phases[k]) / pa.dist.size_periods[k])))) + 1

        if not(np.all(nw_len_seq >= 1)):
            print(nw_len_seq[np.where(nw_len_seq < 1)])
        assert(np.all(nw_size_seq >= 1))

    # print nw_len_seq
    return nw_len_seq, nw_size_seq