import argparse
import os

import matplotlib
import matplotlib.pyplot as plt

from jax.experimental.ode import build_odeint
import jax.numpy as np
import jax.random as random
from jax.random import PRNGKey

import numpyro
from numpyro.infer.util import initialize_model
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

matplotlib.use('Agg')  # noqa: E402


def dy_dt(y, t, theta):
    return np.stack([
        y[1],
        -y[0] - theta * y[1],
    ])

sho_int = build_odeint(dy_dt, rtol=1e-5, atol=1e-3, mxstep=500)

def model(ts, y=None):
    """
    :param int N: number of measurement times
    :param numpy.ndarray y: measured populations with shape (N, 2)
    """
    # initial population
    y0 = numpyro.sample("y0", dist.Uniform(-1., 5.), sample_shape=(2,))

    theta = numpyro.sample("theta", dist.Uniform(0., 2.))

    # integrate dy/dt
    z = sho_int(y0, ts, theta)

    # measurement errors
    sigma = numpyro.sample("sigma",
                           dist.TruncatedNormal(low=0., loc=0., scale=0.1)
                           )

    # measured populations
    numpyro.sample("y", dist.Normal(z[:, 0], sigma), obs=y)


def main(args):
    key = PRNGKey(0)
    true_y0 = np.array([0.1, 0.1])
    true_theta = 0.1
    true_sigma = 0.01
    ts = np.linspace(0., 50., 1000)
    data = sho_int(true_y0, ts, true_theta)[:, 0]
    data += true_sigma * random.normal(key=key, shape=data.shape)
    plt.plot(ts, data)
    plt.savefig('data.pdf')

    mcmc = MCMC(NUTS(model),
                args.num_warmup, args.num_samples, num_chains=args.num_chains,
                progress_bar=True)
    if args.num_chains == 1:
        init_params={
                 'y0': np.array([1.5, 0.6]),
                 'theta': 0.15,
                 'sigma': 0.3,
              }
    else:
        init_params={
                 'y0': np.array([[1.5, 0.6]]*args.num_chains),
                 'theta': np.array([0.15] * args.num_chains),
                 'sigma': np.array([0.3] * args.num_chains),
              }
    mcmc.run(key, ts=ts, y=data,
             #init_params=init_params
             )
    mcmc.print_summary()


if __name__ == '__main__':
    assert numpyro.__version__.startswith('0.2.4')
    parser = argparse.ArgumentParser(description='SHO Model')
    parser.add_argument('-n', '--num-samples', nargs='?', default=200, type=int)
    parser.add_argument('--num-warmup', nargs='?', default=200, type=int)
    parser.add_argument("--num-chains", nargs='?', default=4, type=int)
    parser.add_argument('--device', default='cpu', type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)
