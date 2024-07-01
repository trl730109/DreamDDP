import random
import time
import numpy as np
from comm_core import rank, size, Communicator
import bisect


num_init_points = 0
num_trials = 10
target_time = None

class Tuner(object):
    """Tuning parameter x to minimize the iteration time with Bayesian Optimization."""
    def __init__(self, x=32, bound=(1.0, 100.0), max_num_steps=num_trials):      
        self._current_point = x
        self._bound = bound
        self._l  = bound[0]
        self._r  = bound[1]
        self._mid = 0
        self._iter_time_mid = 0
        self._iter_time_mid_1 = 0
        self._iter_time_mid_2 = 0
        self._max_num_steps = max_num_steps

        self._opt_point = None
        self._opt_iter_time = None
        # random search
        self._init_points = [random.random() * (bound[1]-bound[0]) + bound[0] 
                for _ in range(num_init_points)]
        # grid search
        self._init_points = [i * (bound[1]-bound[0]) / (num_init_points-1) + bound[0] 
                for i in range(num_init_points)]
        
        # empty init points
        #self._init_points = []
      
        self._num_steps = 0
        self._interval = 5
        self._timestamps = []
        self._warmup_record = True

        # self._utility = UtilityFunction(kind='ei', kappa=0.0, xi=0.1)
        # self._opt = BayesianOptimization(f=None, pbounds={'x': bound})
        # self._opt = BinarySearch(self._bound)
        #self._bo_cost = 0
        self._bs_cost = []



    def _record(self):
        self._timestamps.append(time.time())
        if len(self._timestamps) < self._interval:
            return None
      
        if self._warmup_record: # discard warmup time
            self._warmup_record = False
            return None
      
        durations = [self._timestamps[i] - self._timestamps[i-1] 
                     for i in range(3, len(self._timestamps))]
        self._timestamps = []
        return np.mean(durations)
      
    def opt_point(self):
        return self._opt_point, self._opt_iter_time
  
    def step(self):
        """Return new point for fine-tuning when it is ready, else return None."""

        if self._num_steps > self._max_num_steps:
            return None

        if self._num_steps == self._max_num_steps:
            self._num_steps += 1
            if rank() == 0:
                print("BS Tuning optimal param: %.4f, optimal iteration time %.4f" 
                        % (self._opt_point, self._opt_iter_time))
                #print("BO Tuning cost: %.4f" % (self._bo_cost / self._max_num_steps))
                print("BS Tuning cost:", np.mean(self._bs_cost))
                # print("BS Tuning cost of 10 steps:", self._bs_cost)

            if self._current_point != self._opt_point:
                return self._opt_point
            else:
                return None

        
        # record time
        iter_time = self._record()
        if iter_time is None:
            return None

        if rank() == 0:
            print("BS Tuning step [%d], param: %.4f, iteration time: %.4f" % 
                    (self._num_steps, self._current_point, iter_time))
                    
        # store the best result
        if self._opt_point is None or iter_time < self._opt_iter_time:
            self._opt_point = self._current_point
            self._opt_iter_time = iter_time

        stime = time.time() 
        if  self._num_steps % 3 == 0:
            self._iter_time_mid = iter_time
            self._mid = self._current_point
            next_point = self._mid - self._bound[1]/10
            self._current_point = next_point
            self._num_steps += 1
            self._bs_cost.append(time.time() - stime)
            return next_point 
        
        if  self._num_steps % 3 == 1:
            # iter_time_mid_1 = 0           
            self._iter_time_mid_1 =  iter_time
            next_point = self._mid + self._bound[1]/10
            self._current_point  = next_point
            self._num_steps += 1
            self._bs_cost.append(time.time() - stime)
            return next_point 

        if  self._num_steps % 3 == 2:
            self._iter_time_mid_2 =  iter_time
            if  self._iter_time_mid > self._iter_time_mid_1 and self._iter_time_mid < self._iter_time_mid_2:
                self._r = self._mid
                if  self._l > self._r or self._l == self._r:   
                    if rank() == 0:
                        print("Optimal param: %.4f, Optimal iteration time: %.4f" % (self._current_point, iter_time))
                    self._bs_cost.append(time.time() - stime)      
                    return self._current_point 
                next_point = self._l + (self._r - self._l) // 2
                if rank() == 0:
                    print('next_point:',next_point)
                self._current_point  = next_point
                self._num_steps += 1
                self._bs_cost.append(time.time() - stime)
                return next_point        
            elif self._iter_time_mid > self._iter_time_mid_2 and self._iter_time_mid < self._iter_time_mid_1:
                self._l = self._mid
                if  self._l > self._r or self._l == self._r:
                    if rank() == 0:
                        print("Optimal param: %.4f, Optimal iteration time: %.4f" % (self._current_point, iter_time))
                    self._bs_cost.append(time.time() - stime)      
                    return self._current_point 
                next_point = self._l + (self._r - self._l) // 2
                if rank() == 0:
                    print('next_point:',next_point)
                self._current_point  = next_point
                self._num_steps += 1
                self._bs_cost.append(time.time() - stime)
                return next_point
            elif self._iter_time_mid > self._iter_time_mid_1 and self._iter_time_mid > self._iter_time_mid_2:
                if self._iter_time_mid_1 > self._iter_time_mid_2:
                    self._l = self._mid
                    if  self._l > self._r or self._l == self._r:  
                        if rank() == 0:
                            print("Optimal param: %.4f, Optimal iteration time: %.4f" % (self._current_point, iter_time))  
                        self._bs_cost.append(time.time() - stime)
                        return self._current_point 
                    next_point = self._l + (self._r - self._l) // 2
                    self._current_point  = next_point
                    if rank() == 0:
                        print('next_point:',next_point)
                    self._num_steps += 1
                    self._bs_cost.append(time.time() - stime)
                    return next_point
                elif self._iter_time_mid_1 < self._iter_time_mid_2: 
                    self._r = self._mid
                    if  self._l > self._r or self._l == self._r:  
                        if rank() == 0:
                            print("Optimal param: %.4f, Optimal iteration time: %.4f" % (self._current_point, iter_time))  
                        self._bs_cost.append(time.time() - stime)
                        return self._current_point
                    next_point = self._l + (self._r - self._l) // 2
                    if rank() == 0:
                        print('next_point:',next_point)
                    self._current_point  = next_point
                    self._num_steps += 1
                    self._bs_cost.append(time.time() - stime)
                    return next_point
                else:
                    self._l = self._mid
                    if  self._l > self._r or self._l == self._r:   
                        if rank() == 0:
                            print("Optimal param: %.4f, Optimal iteration time: %.4f" % (self._current_point, iter_time))  
                        self._bs_cost.append(time.time() - stime)
                        return self._current_point 
                    next_point = self._l + (self._r - self._l) // 2
                    if rank() == 0:
                        print('next_point:',next_point)
                    self._current_point  = next_point
                    self._num_steps += 1
                    self._bs_cost.append(time.time() - stime)
                    return next_point                                   
            elif self._iter_time_mid < self._iter_time_mid_1 and self._iter_time_mid < self._iter_time_mid_2:
                if rank() == 0:
                    print("Optimal param: %.4f, Optimal iteration time: %.4f" % (self._mid, self._iter_time_mid))
                # return None
                next_point = self._mid
                self._current_point  = next_point
                self._num_steps += 1
                self._bs_cost.append(time.time() - stime)               
                return next_point    

