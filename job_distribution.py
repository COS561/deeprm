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
        self.bimodal = 1
        self.periodic = 0

        #self.switch_chance = 0.8

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

        nw_len_seq = np.reshape(nw_len_seq,[pa.num_ex, pa.simu_len])
        nw_size_seq = np.reshape(nw_size_seq,[pa.num_ex, pa.simu_len, pa.num_res])

    else:

        for i in range(pa.num_ex):
            #set parameters of length dist for cycle i:

            if pa.dist.bimodal:

                if np.random.rand() < 0.5:
                    pa.dist.job_small_chance = 1 - pa.dist.job_small_chance

            elif pa.dist.periodic:
                    pa.dist.job_period = np.random.randint(2, 10)

            for j in range(pa.simu_len):
                #generate length, size attributes of sequence j in cycle i:
                pa.new_job_rate = 0.3
                if np.random.rand() < pa.new_job_rate:

                    if pa.dist.bimodal:

                        nw_len_seq[i, j], nw_size_seq[i, j, :] = pa.dist.bi_model_dist()
        
                    elif pa.dist.periodic:

                        #nw_len_seq[i,j] = round(4*(math.sin(0.5*j) + math.cos(0.25*j))+8)
                        offset = np.random.randint(-2, 2)
                        nw_len_seq[i,j] = round(7*(math.sin((j+offset)/pa.dist.job_period))+8)

                        for k in range(pa.num_res):
                            nw_size_seq[i,j,k] = np.random.randint(1, pa.dist.max_nw_size + 1)

    # for i in range(simu_len):

    #     if np.random.rand() < pa.new_job_rate:  # a new job comes

    #         periodic_dist = 0;

    #         if periodic_dist:

    #             nw_len_seq[i] = round(4*(math.sin(0.5*i) + math.cos(0.25*i))+8)

    #             for j in range(pa.num_res):
    #                 nw_size_seq[i,j] = np.random.randint(1, pa.dist.max_nw_size + 1)

    #         else:

    #             nw_len_seq[i], nw_size_seq[i, :] = nw_dist()

    # nw_len_seq = np.reshape(nw_len_seq,
    #                         [pa.num_ex, pa.simu_len])
    # nw_size_seq = np.reshape(nw_size_seq,
    #                          [pa.num_ex, pa.simu_len, pa.num_res])

    print nw_len_seq
    return nw_len_seq, nw_size_seq

def generate_sequence_for_rnn(pa, seed=42):

    np.random.seed(seed)

    simu_len = pa.simu_len

    nw_dist = pa.dist.bi_model_dist

    nw_seq = np.zeros((simu_len, pa.num_res + 1), dtype=int)
    # print nw_seq

    for i in range(simu_len):

        if np.random.rand() < pa.new_job_rate:  # a new job comes

            if pa.nonStationary:
                if np.random.rand() < pa.dist.switch_chance: # switch duration distribution
                    pa.dist.job_small_chance = 1 - pa.dist.job_small_chance

            nw_seq[i, 0], nw_seq[i, 1:] = nw_dist()

    #print nw_seq

    return nw_seq
