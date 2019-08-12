import numpy as np
import queue
import pandas as pd

class TimeSeriesAnalysis(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cusum = []
        self.changepoints = []
        self.changepoint_intervals = []

    def ChangePointAnalysis(self, num_bootstraps=1000, confidence=.95, max_num_changepoints=50):
        self.cusum = cumulative_sum(self.y)
        changepoint, S_diff = find_change_point(self.cusum)
        cl = bootstrap_analysis(self.y, S_diff, num_bootstraps)


        if cl > confidence:
            self.changepoints.append((changepoint, cl))

            q = queue.Queue()
            cp1 = TimeSeriesAnalysis(self.x[:changepoint], self.y[:changepoint])
            cp2 = TimeSeriesAnalysis(self.x[changepoint:], self.y[changepoint:])
            q.put(cp1)
            q.put(cp2)
            while (not q.empty()):
                if (len(self.changepoints) == max_num_changepoints):
                    return
                ts_object = q.get()
                interval_cusum = cumulative_sum(ts_object.y)
                interval_cp, interval_diff = find_change_point(interval_cusum)
                interval_changepoint = ts_object.x[interval_cp]
                interval_changepoint_idx = self.x.index(interval_changepoint)
                if (is_new_changepoint(interval_changepoint_idx, self.changepoints)):
                    interval_cl = bootstrap_analysis(ts_object.y, interval_diff, num_bootstraps)
                    if interval_cl > confidence:
                        self.changepoints.append((interval_changepoint_idx, interval_cl))

                        interval_cp1 = TimeSeriesAnalysis(ts_object.x[:changepoint], ts_object.y[:changepoint])
                        if len(ts_object.x[:changepoint]) > 1:
                            q.put(interval_cp1)

                        interval_cp2 = TimeSeriesAnalysis(ts_object.x[changepoint:], ts_object.y[changepoint:])
                        if len(ts_object.x[changepoint:]) > 1:
                            q.put(interval_cp2)


    '''
        events: array containing timestamps at which the actions happen
    '''
    def AutoCorrelationFunction(self, events):
        '''
            Build a vector which has same size as x. +1 indicates a change point,
            -1 indicates an event. Perform ACF at lag={0,1,2,...,N} where N is the
            size of x.
        '''

        T = np.zeros(len(self.x))
        dates = [int(x) for x in self.x]
        #events = [int(e) for e in events]

        for cp in self.changepoints:
            T[cp[0]] = 1

        for event in events:
            idx = np.searchsorted(self.x, event)
            T[idx] = -1

        results = []
        for l in range(len(self.x)/2):
            C = pd.Series(T).autocorr(l)
            results.append((l, C))

        # Sort by correlation value
        return sorted(results, key=lambda x: abs(x[1]), reverse=True)



    def CumulativeSum(self):
        average = np.mean(self.y)
        cusum = np.zeros(len(self.y))
        for i in range(1, len(self.y)):
            cusum[i] = cusum[i-1] + (self.y[i] - average)
        self.cusum = cusum
   
def is_new_changepoint(cp, changepoints_list):
    for (changepoint, confidence_level) in changepoints_list:
        if cp == changepoint:
            return False
    return True


def cumulative_sum(y):
    average = np.mean(y)
    cusum = np.zeros(len(y))
    for i in range(1, len(y)):
        cusum[i] = cusum[i-1] + (y[i] - average)
    return cusum

def find_change_point(cusum):
    max_point = max(cusum)
    min_point = min(cusum)
    if abs(max_point) > abs(min_point):
        return np.argmax(cusum), max_point - min_point
    else:
        return np.argmin(cusum), max_point - min_point

def bootstrap_analysis(y, S_diff, num_bootstraps):
    X = 0
    for i in range(num_bootstraps):
        bootstrap_sample = np.random.permutation(y)
        bootstrap_cusum = cumulative_sum(bootstrap_sample)
        _, S_diff_bootstrap = find_change_point(bootstrap_cusum)
        if S_diff > S_diff_bootstrap:
            X += 1

    return X/float(num_bootstraps)
