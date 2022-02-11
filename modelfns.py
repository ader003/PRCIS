# This code is from the Supporting Website page for
# Chin-Chia Michael Yeh, et al.: Error-bounded Approximate Time Series Joins using Compact Dictionary Representations of Time Series. SDM to appear.
# https://sites.google.com/view/appjoin ("Implementation details for the proposed method")


import numpy as np
import matrixprofile


_EPS = 1e-6

class MatproDict:
    def __init__(self, win_len, ctx_factor=None, n_pattern=None,
                 save_factor=0.5, n_job=4, verbose=False):
        self.win_len = win_len
        if save_factor is not None:
            self.n_pattern = None
        else:
            self.n_pattern = n_pattern
        if ctx_factor is None:
            self.pattern_len = win_len
        else:
            self.pattern_len = int(win_len * ctx_factor)
        self.save_factor = save_factor
        self.n_job = n_job
        self.verbose = verbose
        

    def check_valid(self, data):
        is_valid = np.all(np.isfinite(data)) and np.std(data) > _EPS
        return is_valid

    def fit(self, data):
        data_valid = data.copy()
        data_len = data_valid.shape[0]
        if self.save_factor == 0:
            self.capture_pattern(data_valid)
            return

        n_sub = data_len - self.win_len + 1
        is_valid = np.zeros(n_sub, dtype=bool)
        for i in range(n_sub):
            data_i = data[i:i + self.win_len]
            is_valid[i] = self.check_valid(data_i)
        data_valid[np.logical_not(np.isfinite(data_valid))] = 0
        prefix = (self.pattern_len - self.win_len) // 2
        suffix = self.pattern_len - self.win_len - prefix

        total_valid = np.zeros(data_len)
        for i in range(n_sub):
            if is_valid[i]:
                total_valid[i:i + self.win_len] = 1
        total_valid = np.sum(total_valid)

        if self.verbose:
            print('compute self-join ', end='')
        matpro = matrixprofile.algorithms.mpx(
            data_valid, self.win_len, cross_correlation=True,
            n_jobs=self.n_job)
        if self.verbose:
            print('done')
        mp = matpro['mp']
        ez = int(np.ceil(self.win_len / 4.0))
        ez_idx = []
        dp = None

        if self.n_pattern is None:
            n_iter = n_sub
        else:
            n_iter = self.n_pattern

        mask = np.ones(data.shape[0], dtype=bool)
        for i in range(n_iter):
            if self.verbose:
                print('captureing element {0:d} '.format(i), end='')
            if i == 0:
                mp_i = mp
            else:
                mp_i = mp - dp

            for idx in ez_idx:
                ez_start = idx - ez
                ez_start = max(ez_start, 0)
                ez_end = idx + ez
                ez_end = min(ez_end, mp.shape[0])
                mp_i[ez_start:ez_end] = -np.inf

            mp_i[np.logical_not(is_valid)] = -np.inf
            idx = np.argmax(mp_i)
            if self.verbose:
                print('index={0:d}, value={1:0.4f} '.format(
                    idx, mp_i[idx]), end='')
            if not np.isfinite(mp_i[idx]):
                break
            ez_idx.append(idx)

            idx_start = idx
            idx_end = idx_start + self.win_len
            idx_end = min(idx_end, data.shape[0])
            idx_start = idx_end - self.win_len

            if idx_start - prefix < 0:
                idx_start_ = 0
                idx_end_ = idx_start_ + self.pattern_len
            elif idx_end + suffix > data_len:
                idx_end_ = data_len
                idx_start_ = idx_end_ - self.pattern_len
            else:
                idx_start_ = idx_start - prefix
                idx_end_ = idx_end + suffix

            mask_old = mask.copy()
            mask[idx_start_:idx_end_] = False

            if self.save_factor is not None:
                n_select = np.sum(np.logical_not(mask))
                save_factor = 1 - n_select / total_valid
                if self.verbose:
                    print('factor={0:0.2f}, {1:d}/{2:d} '.format(
                        save_factor, int(n_select), int(total_valid)),
                        end='')
                if len(ez_idx) > 1 and save_factor < self.save_factor:
                    mask = mask_old
                    if self.verbose:
                        print('{0:0.2f} < {1:0.2f} terminal'.format(
                            save_factor, self.save_factor))
                    break

            if self.verbose:
                print('compute mass ', end='')
            query = data[idx_start:idx_end]
            dp_i = matrixprofile.algorithms.mass2(data_valid, query)
            dp_i = 1 - (np.abs(dp_i) ** 2) / (2 * query.shape[0])
            dp_i[dp_i > 1] = 1
            dp_i[dp_i < -1] = -1

            if i == 0:
                dp = dp_i
            else:
                dp = np.maximum(dp, dp_i)
            if self.verbose:
                print('done')

        data_masked = data.copy()
        data_masked[mask] = np.inf
        data_masked = np.concatenate(
            (data_masked, [np.inf, ], ), axis=0)
        self.capture_pattern(data_masked)

    def capture_pattern(self, data):
        data_valid = data.copy()
        n_sub = data_valid.shape[0] - self.win_len + 1
        is_valid = np.zeros(
            n_sub, dtype=bool)
        for i in range(n_sub):
            data_i = data[i:i + self.win_len]
            is_valid[i] = self.check_valid(data_i)

        pattern_idx = []
        last_status = None
        cur_idx = -np.ones(2, dtype=int)
        for i in range(is_valid.shape[0]):
            if cur_idx[0] == -1 and last_status != True and is_valid[i]:
                cur_idx[0] = i
            if cur_idx[0] > -1 and is_valid[i]:
                cur_idx[1] = i + self.win_len
            if cur_idx[0] > -1 and last_status == True and not is_valid[i]:
                pattern_idx.append(cur_idx)
                cur_idx = -np.ones(2, dtype=int)
            last_status = is_valid[i]

        if cur_idx[1] > -1:
            pattern_idx.append(cur_idx)

        self.pattern = []
        self.pattern_idxs = []
        for idx in pattern_idx:
            if idx[1] - idx[0] < self.win_len:
                continue
            self.pattern.append(data_valid[idx[0]:idx[1]])
            self.pattern_idxs.append((idx[0], idx[1]))
        self.n_pattern = len(self.pattern)

    def get_pattern(self):
        return self.pattern.copy(), self.pattern_idxs.copy()

    def get_pattern_size(self):
        pattern_size = 0
        for i in range(self.n_pattern):
            pattern_i = self.pattern[i]
            pattern_size += pattern_i.shape[0]
        return pattern_size

    def join(self, query):
        query = query.copy()
        pro_len = query.shape[0] - self.win_len + 1
        is_valid = np.zeros(pro_len)
        for i in range(pro_len):
            is_valid[i] = self.check_valid(
                query[i:i + self.win_len])
        query[np.logical_not(np.isfinite(query))] = 0

        pro = []
        for i in range(self.n_pattern):
            pattern_i = self.pattern[i]
            pro_i = matrixprofile.algorithms.mpx(
                query, self.win_len, query=pattern_i, n_jobs=self.n_job)
            pro_i = pro_i['mp']
            pro.append(pro_i)
        pro = np.array(pro)
        pro = np.min(pro, axis=0)
        pro[np.logical_not(is_valid)] = np.inf
        return pro

