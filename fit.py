# %%
import os
from random import uniform
import numpy as np
from IPython import embed
import scipy.special as sp
import scipy.optimize as opt
import matplotlib.pyplot as plt

import copy


# %%
class PDFunction:

    def __init__(self, *args) -> None:
        self.init_params = args
        self.params = [*args]

    def update(self, *args):
        self.params = [*args]

    def __call__(self, t):
        raise NotImplementedError

    def em_step(self, arr, prob):
        raise NotImplementedError


class GammaDistribution(PDFunction):

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def __call__(self, t):
        a, b = self.params
        return b**a / (sp.gamma(a)) * np.e**(-b * t) * t**(a - 1) 

    def em_step(self, arr, prob):
        target = np.log((prob * arr).sum() / prob.sum()) - (prob * np.log(arr)).sum() / prob.sum()
        coef = prob.sum() / np.maximum((prob * arr).sum(), 1e-8)
        func = lambda x: np.log(x+1e-5) - sp.digamma(x+1e-5) - target
        jac = lambda x: 1 / x - sp.gamma(x)
        root = opt.root(func, self.params[0], jac=jac)
        self.params[0] = root.x[0]
        self.params[1] = self.params[0] * coef


class PoissonDistribution(PDFunction):
    ...

# %%
def visualize_pdf(func: PDFunction, boundary, nstep=1000, color='green'):
    low, high = boundary
    x = np.arange(nstep) / nstep * (high - low) + low
    y = func(x)
    plt.plot(x, y, color=color, alpha=0.75)


def error_pdf(func, data_arr, steps=50000):
    y = np.histogram(data_arr, bins=steps, density=True)[0]
    x = np.arange(steps) / steps * (data_arr.max() - data_arr.min()) + data_arr.min()
    z = func(x)
    return np.abs(y - z).mean()


# %%


class FitRunner:

    def __init__(self, distribution, arr, init_weight=0.5) -> None:
        self.data_arr = arr
        self.weight = init_weight
        dist_cls_a, args_a = distribution[0]
        self.dist_cls_a = dist_cls_a
        self.dist_a: PDFunction = dist_cls_a(*args_a)
        dist_cls_b, args_b = distribution[1]
        self.dist_cls_b = dist_cls_b
        self.dist_b: PDFunction = dist_cls_b(*args_b)
        self.best_err = float('inf')
        self.opt_params_a = copy.deepcopy(args_a)
        self.opt_params_b = copy.deepcopy(args_b)
        self.opt_weight = init_weight

    def fit(self, step=10, visualize=False, quiet=False, save=None, opt=True):
        for i in range(step):
            calc = lambda x: self.weight * self.dist_a(x) + (1 - self.weight) * self.dist_b(x)
            if not quiet:
                print(f"Step #{i}")
                print(self)
                print(f"Error: {error_pdf(calc, self.data_arr)}")
            if visualize:
                self.visualize(save)
            pdf_a = self.dist_a(self.data_arr)
            pdf_b = self.dist_b(self.data_arr)
            pdf_sum = self.weight * pdf_a + (1 - self.weight) * pdf_b
            prob_a = self.weight * pdf_a / pdf_sum
            prob_b = (1 - self.weight) * pdf_b / pdf_sum
            self.weight = prob_a.sum() / len(prob_a)
            self.dist_a.em_step(self.data_arr, prob_a)
            self.dist_b.em_step(self.data_arr, prob_b)
            error = self.error()
            if error < self.best_err:
                self.best_err = error
                self.opt_params_a = copy.deepcopy(self.dist_a.params)
                self.opt_params_b = copy.deepcopy(self.dist_b.params)
                self.opt_weight = self.weight
        if opt:
            self.dist_a.update(*self.opt_params_a)
            self.dist_b.update(*self.opt_params_b)
            self.weight = self.opt_weight

    def error(self, steps=50000):
        y = np.histogram(self.data_arr, bins=steps, density=True)[0]
        x = np.arange(steps) / steps * (self.data_arr.max() - self.data_arr.min()) + self.data_arr.min()
        z = self.dist_a(x) * self.weight + self.dist_b(x) * (1 - self.weight)
        return np.abs(y - z).mean()

    def visualize(self, save=None):
        data_arr = self.data_arr
        plt.hist(data_arr, color='g', bins=500, alpha=0.5, density=True)
        calc = lambda x: (self.weight * self.dist_a(x) + (1 - self.weight) * self.dist_b(x))
        visualize_pdf(calc, (data_arr.min(), data_arr.max()))
        visualize_pdf(lambda x: self.weight * self.dist_a(x), (data_arr.min(), data_arr.max()), color='red')
        visualize_pdf(lambda x: (1 - self.weight) * self.dist_b(x), (data_arr.min(), data_arr.max()),
                      color='blue')
        if save is None:
            plt.show()
            plt.cla()
        else:
            try:
                os.remove(save)
            except:
                pass
            print("Saving...")
            plt.savefig(save)
            plt.cla()

    def judge(self, arr):
        return self.weight * self.dist_a(arr) > (1 - self.weight) * self.dist_b(arr)

    def judge2(self, arr, init=0.01):
        root = opt.root(lambda x: self.weight * self.dist_a(x) - (1 - self.weight) * self.dist_b(x), init).x[0]
        # root = opt.newton(lambda x: self.weight * self.dist_a(x) - (1 - self.weight) * self.dist_b(x), init)
        return arr < root

    def __str__(self) -> str:
        return (f'Distribution 1 params: {self.dist_a.params}\n') + (
            f'Distribution 2 params: {self.dist_b.params}\n') + (f'Weight: {self.weight}')

def fit_gamma(arr, a1=2, b1=200, a2=50, b2=50, weight=0.5, step=10, save=None, quiet=True):
    arr = np.abs(arr)
    # a1, b1 = 2, 200
    # a2, b2 = 50, 50
    # weight = 0.5
    dist_cls = GammaDistribution
    # bins = 50
    # plt.hist(arr, bins=bins, alpha=0.5, density=True, stacked=True)
    dist_a, dist_b = dist_cls(a1, b1), dist_cls(a2, b2)
    # visualize_pdf(lambda x: (1 - weight) * dist_b(x), (arr.min(), arr.max()), color='blue')
    # visualize_pdf(lambda x: weight * dist_a(x) + (1 - weight) * dist_b(x), (arr.min(), arr.max()), color='green')
    runner = FitRunner([(dist_cls, (a1, b1)), (dist_cls, (a2, b2))], arr)
    runner.fit(step=step, quiet=quiet)
    
    if save is not None:
        runner.visualize(save=save)
    mask_label = []
    for each in arr:
        if weight * dist_a(each) >= (1 - weight) * dist_b(each):
            mask_label.append(False)
        else:
            mask_label.append(True)
    return mask_label

# %%
# Data preparation
# arr = np.load('test.npy')
# arr = np.abs(arr)
if __name__ == '__main__':
    a1, b1 = 2, 10
    a2, b2 = 8, 12
    weight = 0.4
    arr = np.array([(np.random.gamma(a1, 1/b1) if np.random.uniform(0, 1) < weight else np.random.gamma(a2, 1/b2))  for _ in range(50000)])

    # %%
    # Initial params
    a1, b1 = 0.5, 1.0
    a2, b2 = 5.0, 5.0
    weight = 0.5
    dist_cls = GammaDistribution
    bins = 500

    # %%
    # Visualize data
    plt.hist(arr, bins=bins, alpha=0.5, density=True, stacked=True)
    dist_a, dist_b = dist_cls(a1, b1), dist_cls(a2, b2)
    visualize_pdf(lambda x: weight * dist_a(x), (arr.min(), arr.max()), color='green')
    visualize_pdf(lambda x: (1 - weight) * dist_b(x), (arr.min(), arr.max()), color='blue')
    plt.show()
    # %%
    # Fitting
    runner = FitRunner([(dist_cls, (a1, b1)), (dist_cls, (a2, b2))], arr)
    runner.visualize()
    runner.fit(step=50, quiet=True, opt=True)
    print(runner.error())
    
    plt.hist(arr, range=(arr.min(), arr.max()), bins=bins, alpha=0.5, density=False, stacked=True, color='red')
    plt.show()
    plt.hist(arr[runner.judge(arr)], range=(arr.min(), arr.max()), bins=bins, alpha=0.5, density=False, stacked=True, color='green')
    init_a = (runner.dist_a.params[0] - 1) / (runner.dist_a.params[1])
    init_b = (runner.dist_b.params[0] - 1) / (runner.dist_b.params[1])
    print('init: ', (init_a + init_b) / 2)
    plt.show()
    plt.hist(arr[runner.judge2(arr, (init_a + init_b) / 2)], range=(arr.min(), arr.max()), bins=bins, alpha=0.5, density=False, stacked=True, color='blue')
    plt.show()
    runner.visualize()
    print(runner)

# %%
