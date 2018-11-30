from queue import PriorityQueue
from basic_lsh import BasicLSH
import numpy as np
from operator import add

class PerturbSet:
    def __init__(self, ps, score):
        self.ps = ps
        self.score = score

    def __gt__(self, other):
        return self.score > other.score

    def __str__(self):
        return str(self.score) + ' ' + str(self.ps)

    def __repr__(self):
        return self.__str__()

class MultiprobeLSH(BasicLSH):

    def __init__(self, dim, l, m, w, t):
        """

        :param dim: diminsionality of the data
        :param l: number of hash
        :param m: number of hash values to concatenate to form the key to the hash tables
        :param w: bucket size
        :param t: number of perturbation vectors that will be applied to each query
        """
        super().__init__(dim, l, m, w)
        self.t = t

        self.scores = [0 for _ in range(2*m)]
        self.perturb_sets = []
        self.perturb_vecs = []

        self.hash_tables = [{} for _ in range(self.l)]
        self.init_probe_sequence()

    def init_probe_sequence(self):
        m = self.m
        score = self.scores

        for j in range(1, m+1):
            score[j-1] = j * (j+1) / (4 * (m+1) * (m+2))

        for j in range(m+1, 2*m + 1):
            score[j-1] = 1 - (2*m+1-j) / (m+1) + (2*m+1-j) * (2*m+2-j) / (4 * (m+1) * (m+2))

        self.scores = score

        self.gen_perturbing_sets()
        self.gen_perturbing_vectors()

    @staticmethod
    def expand(ps):
        mx = 0
        for i in ps:
            mx = max(mx, i)

        nx = ps.copy()
        nx[mx + 1] = True
        return nx

    @staticmethod
    def shift(ps):
        mx = 0
        for i in ps:
            mx = max(mx, i)

        nx = ps.copy()
        nx.pop(mx)
        nx[mx + 1] = True
        return nx

    def is_valid(self, perturbset, m):
        ps = perturbset.ps
        for key in ps:
            if 2 * m + 1 - key in ps:
                return False
            if key > 2 * m:
                return False
        return True

    def gen_perturbing_sets(self):
        pq = PriorityQueue()
        start = {1: True}
        pq.put(PerturbSet(start, self.get_score(start)))
        #print(pq.queue[0])

        for i in range(self.t):
            counter = 0
            while True:
                top = pq.get()
                next_shift = self.shift(top.ps)
                pq.put(PerturbSet(next_shift, self.get_score(next_shift)))

                next_expand = self.expand(top.ps)
                pq.put(PerturbSet(next_expand, self.get_score(next_expand)))

                if self.is_valid(pq.queue[0], self.m): # top
                    self.perturb_sets.append(pq.queue[0])
                    break

                if counter >= 2 * self.m:
                    raise RuntimeError('too many iterations, probably infinite loop!')
                counter += 1

    def gen_perturbing_vectors(self):
        perms = []

        for i in range(self.l):
            perm = np.random.permutation(self.m).tolist()
            perms.append(perm + perm[::-1])


        for pso in self.perturb_sets:
            #print(pso)
            perturb_vec = []
            for j in range(self.l):
                vec = [0 for _ in range(self.m)]
                for k in pso.ps:
                    mapped_ind = perms[j][k-1]
                    if k > self.m:
                        # If it is -1
                        vec[mapped_ind] = -1
                    else:
                        # If it is +1
                        vec[mapped_ind] = 1
                perturb_vec.append(vec)
            self.perturb_vecs.append(perturb_vec)
        #print(self.perturb_vecs)

    def get_score(self, ps):
        score = 0
        for j in ps:
            score += self.scores[j-1]
        return score

    def hash2(self, planes, input_point):
        try:
            input_point = np.array(input_point)  # for faster dot product
            projections = np.dot(planes, input_point)
        except TypeError as e:
            print("""The input point needs to be an array-like object with
                  numbers only elements""")
            raise
        except ValueError as e:
            print("""The input point needs to be of the same dimension as
                  `input_dim` when initializing this LSHash instance""", e)
            raise
        else:
            return projections

    def query(self, q, num_results=None, distance_func=None):

        base_key = self.hash(q)
        #print('base_key', base_key)
        #print('per_vec', self.perturb_vecs)

        candidates = set()
        for i in range(len(self.perturb_vecs) + 1):
            perturbed_table_keys = base_key
            if i != 0:
                perturbed_table_keys = self.perturb(base_key, self.perturb_vecs[i-1])

            #print("new_vec", perturbed_table_keys)

            results = self.query_helper(perturbed_table_keys, candidates)

            for result in results:
                candidates.add(result)

            if num_results and len(candidates) >= num_results:
                return list(candidates)[:num_results]
        return list(candidates)

    @staticmethod
    def str_intersection(str1, str2):
        count = 0
        for i, c in enumerate(str1):
            if str1[i] == str2[i]:
                count += 1
        return count

    def query_helper(self, table_keys, results):
        hvs = self.input_to_hash(table_keys)
        seen = set()
        for i, table in enumerate(self.hash_tables):
            if hvs[i] in table:
                for candidate in table[hvs[i]]:
                    seen.add(candidate)

        return list(seen)

    def perturb(self, base_key, perturbation):
        if len(base_key) != len(perturbation):
            raise ValueError("Number tables does not match with number of perturb vecs")
        perturbed_table_key = []
        for i, p in enumerate(perturbation):
            perturbed_table_key.append((np.array(base_key[i]) + p).tolist())
        return perturbed_table_key

# test code: try different t

"""
mp = MultiprobeLSH(dim=10, l=7, m=3, w=5.0, t=1)

points = np.random.rand(10,10) * 32.0
#print(points)

for i, p in enumerate(points):
    mp.insert(p, i)
    mp.insert(p, i+10)

mp.insert(list(np.array(points[1]) + [1, 0, 0, 2, 0, 3, 0, 4, 0, 0]), 456)

new = list(np.array(points[1]) + [1, 0, 0, 1, 0, 1, 0, 0, 0, 0])

print(mp.query(new, 3))
"""




