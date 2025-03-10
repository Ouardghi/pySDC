import numpy as np
from matplotlib.colors import TABLEAU_COLORS

cmap = TABLEAU_COLORS


def merge_descriptions(descA, descB):
    """
    Merge two dictionaries that may contain dictionaries, which happens when merging descriptions, for instance.

    Keys that occur in both dictionaries will be overwritten by the ones from `descB` and `descA` will be modified, not
    copied!

    Args:
        descA (dict): Dictionary that you want to merge into
        descB (dict): Dictionary you want to merge from

    Returns:
        dict: decsA with updated parameters
    """
    for key in descB.keys():
        if type(descB[key]) == dict:
            descA[key] = merge_descriptions(descA.get(key, {}), descB[key])
        else:
            descA[key] = descB[key]
    return descA


class Strategy:
    '''
    Abstract class for resilience strategies
    '''

    def __init__(self, useMPI=False, skip_residual_computation='none', stop_at_nan=True, **kwargs):
        '''
        Initialization routine
        '''
        self.useMPI = useMPI
        self.max_steps = 1e5

        # set default values for plotting
        self.linestyle = '-'
        self.marker = '.'
        self.name = ''
        self.bar_plot_x_label = ''
        self.color = list(cmap.values())[0]

        # parameters for computational efficiency
        if skip_residual_computation == 'all':
            self.skip_residual_computation = ('IT_CHECK', 'IT_DOWN', 'IT_UP', 'IT_FINE', 'IT_COARSE')
        elif skip_residual_computation == 'most':
            self.skip_residual_computation = ('IT_DOWN', 'IT_UP', 'IT_FINE', 'IT_COARSE')
        elif skip_residual_computation == 'none':
            self.skip_residual_computation = ()
        else:
            raise NotImplementedError(
                f'Don\'t know when to skip residual computation with rule \"{skip_residual_computation}\"'
            )

        self.stop_at_nan = stop_at_nan

        # setup custom descriptions
        self.custom_description = {}
        self.custom_description['sweeper_params'] = {'skip_residual_computation': self.skip_residual_computation}
        self.custom_description['level_params'] = {}
        self.custom_description['problem_params'] = {}
        self.custom_description['step_params'] = {}
        self.custom_description['convergence_controllers'] = {}

        # prepare parameters for masks to identify faults that cannot be fixed by this strategy
        self.fixable = []
        self.fixable += [
            {
                'key': 'node',
                'op': 'gt',
                'val': 0,
            }
        ]
        self.fixable += [
            {
                'key': 'error',
                'op': 'isfinite',
            }
        ]

        # stuff for work-precision diagrams
        self.precision_parameter = None
        self.precision_parameter_loc = []

    def __str__(self):
        return self.name

    def get_controller_params(self, **kwargs):
        return {'all_to_done': False}

    def get_description_for_tolerance(self, problem, param, **kwargs):
        return {}

    def get_fixable_params(self, **kwargs):
        """
        Return a list containing dictionaries which can be passed to `FaultStats.get_mask` as keyword arguments to
        obtain a mask of faults that can be fixed

        Returns:
            list: Dictionary of parameters
        """
        return self.fixable

    def get_fault_args(self, problem, num_procs):
        '''
        Routine to get arguments for the faults that are exempt from randomization

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            dict: Arguments for the faults that are exempt from randomization
        '''
        args = {}
        args['target'] = 0

        if problem.__name__ == "run_vdp":
            args['time'] = 5.5  # 25
        elif problem.__name__ == "run_Schroedinger":
            args['time'] = 0.3
        elif problem.__name__ == "run_quench":
            args['time'] = 41.0
        elif problem.__name__ == "run_Lorenz":
            args['time'] = 10
        elif problem.__name__ == "run_AC":
            args['time'] = 1e-2
        elif problem.__name__ == "run_RBC":
            args['time'] = 20.19
        elif problem.__name__ == "run_GS":
            args['time'] = 100.0

        return args

    def get_random_params(self, problem, num_procs):
        '''
        Routine to get parameters for the randomization of faults

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            dict: Randomization parameters
        '''
        base_params = self.get_base_parameters(problem, num_procs)

        rnd_params = {}
        rnd_params['iteration'] = base_params['step_params']['maxiter']
        rnd_params['rank'] = num_procs

        if problem.__name__ in ['run_Schroedinger', 'run_quench', 'run_AC', 'run_RBC', 'run_GS']:
            rnd_params['min_node'] = 1

        if problem.__name__ == "run_quench":
            rnd_params['iteration'] = 5
        elif problem.__name__ == 'run_Lorenz':
            rnd_params['iteration'] = 5
        elif problem.__name__ == 'run_RBC':
            rnd_params['problem_pos'] = [3, 16, 16]
        elif problem.__name__ == 'run_vdp':
            rnd_params['iteration'] = 5

        return rnd_params

    @property
    def style(self):
        """
        Get the plotting parameters for the strategy.
        Supply them to a plotting function using `**`

        Returns:
            (dict): The plotting parameters as a dictionary
        """
        return {
            'marker': self.marker,
            'label': self.label,
            'color': self.color,
            'ls': self.linestyle,
        }

    @property
    def label(self):
        """
        Get a label for plotting
        """
        return self.name

    @classmethod
    def get_Tend(cls, problem, num_procs=1, resilience_experiment=False):
        '''
        Get the final time of runs for fault stats based on the problem

        Args:
            problem (function): A problem to run
            num_procs (int): Number of processes

        Returns:
            float: Tend to put into the run
        '''
        if problem.__name__ == "run_vdp":
            if resilience_experiment:
                return 11.5
            else:
                return 20
        elif problem.__name__ == "run_piline":
            return 20.0
        elif problem.__name__ == "run_Lorenz":
            return 20
        elif problem.__name__ == "run_Schroedinger":
            return 1.0
        elif problem.__name__ == "run_quench":
            return 500.0
        elif problem.__name__ == "run_AC":
            return 0.025
        elif problem.__name__ == "run_RBC":
            return 21
        elif problem.__name__ == "run_GS":
            return 500
        else:
            raise NotImplementedError('I don\'t have a final time for your problem!')

    def get_base_parameters(self, problem, num_procs=1):
        '''
        Get a base parameters for the problems independent of the strategy.

        Args:
            problem (function): A problem to run
            num_procs (int): Number of processes

        Returns:
            dict: Custom description
        '''
        from pySDC.implementations.convergence_controller_classes.step_size_limiter import StepSizeLimiter

        custom_description = {}
        custom_description['step_params'] = {}
        custom_description['level_params'] = {}
        custom_description['problem_params'] = {}

        if problem.__name__ == "run_vdp":
            custom_description['step_params'] = {'maxiter': 3}
            custom_description['problem_params'] = {
                'u0': np.array([1.1, 0], dtype=np.float64),
                'mu': 1000,
                'crash_at_maxiter': False,
                'newton_tol': 1e-11,
                'stop_at_nan': False,
            }
            custom_description['level_params'] = {'dt': 1e-4}

        elif problem.__name__ == "run_Lorenz":
            custom_description['step_params'] = {'maxiter': 5}
            custom_description['level_params'] = {'dt': 1e-3}
            custom_description['problem_params'] = {'stop_at_nan': False}
        elif problem.__name__ == "run_Schroedinger":
            custom_description['step_params'] = {'maxiter': 5}
            custom_description['level_params'] = {'dt': 1e-2, 'restol': -1}
            custom_description['problem_params']['nvars'] = (256, 256)
        elif problem.__name__ == "run_quench":
            custom_description['level_params'] = {'restol': -1, 'dt': 8.0}
            custom_description['step_params'] = {'maxiter': 5}
            custom_description['problem_params'] = {
                'newton_maxiter': 29,
                'newton_tol': 1e-7,
                'nvars': 2**6,
                'direct_solver': False,
                'lintol': 1e-8,
                'liniter': 29,
                'order': 6,
            }
        elif problem.__name__ == "run_AC":
            eps = 4e-2
            custom_description['step_params'] = {'maxiter': 5}
            custom_description['problem_params'] = {
                'nvars': (128,) * 2,
                'init_type': 'circle',
                'eps': eps,
                'radius': 0.25,
                'nu': 2,
            }
            custom_description['level_params'] = {'restol': -1, 'dt': 0.1 * eps**2}
        elif problem.__name__ == 'run_RBC':
            custom_description['level_params']['dt'] = 2.5e-2 if num_procs == 4 else 5e-2
            custom_description['step_params'] = {'maxiter': 5}
        elif problem.__name__ == 'run_GS':
            custom_description['level_params']['dt'] = 1.0
            custom_description['step_params'] = {'maxiter': 5}

        custom_description['convergence_controllers'] = {
            # StepSizeLimiter: {'dt_min': self.get_Tend(problem=problem, num_procs=num_procs) / self.max_steps}
        }

        if self.stop_at_nan:
            from pySDC.implementations.convergence_controller_classes.crash import StopAtNan

            custom_description['convergence_controllers'][StopAtNan] = {'thresh': 1e20}

        from pySDC.implementations.convergence_controller_classes.crash import StopAtMaxRuntime

        max_runtime = {
            'run_vdp': 1000,
            'run_Lorenz': 500,
            'run_Schroedinger': 150,
            'run_quench': 150,
            'run_AC': 150,
            'run_RBC': 1000,
            'run_GS': 100,
        }

        custom_description['convergence_controllers'][StopAtMaxRuntime] = {
            'max_runtime': max_runtime.get(problem.__name__, 100)
        }
        return custom_description

    def get_custom_description(self, problem, num_procs=1):
        '''
        Get a custom description based on the problem

        Args:
            problem (function): A problem to run
            num_procs (int): Number of processes

        Returns:
            dict: Custom description
        '''
        custom_description = self.get_base_parameters(problem, num_procs)
        return merge_descriptions(custom_description, self.custom_description)

    def get_custom_description_for_faults(self, problem, *args, **kwargs):
        '''
        Get a custom description based on the problem to run the fault stuff

        Returns:
            dict: Custom description
        '''
        custom_description = self.get_custom_description(problem, *args, **kwargs)
        if problem.__name__ == "run_vdp":
            custom_description['step_params'] = {'maxiter': 5}
            custom_description['problem_params'] = {
                'u0': np.array([2.0, 0], dtype=np.float64),
                'mu': 5,
                'crash_at_maxiter': False,
                'newton_tol': 1e-11,
                'stop_at_nan': False,
            }
            custom_description['level_params'] = {'dt': 1e-2}
        return custom_description

    def get_reference_value(self, problem, key, op, num_procs=1):
        """
        Get a reference value for a given problem for testing in CI.

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            key (str): The name of the variable you want to compare
            op (function): The operation you want to apply to the data
            num_procs (int): Number of processes

        Returns:
            The reference value
        """
        raise NotImplementedError(
            f'The reference value you are looking for is not implemented for {type(self).__name__} strategy!'
        )


class InexactBaseStrategy(Strategy):
    """
    Base class for inexact strategies.
    """

    def __init__(
        self, double_adaptivity=False, newton_inexactness=True, linear_inexactness=True, SDC_maxiter=16, **kwargs
    ):
        kwargs = {**kwargs, 'skip_residual_computation': 'most'}
        super().__init__(**kwargs)
        self.double_adaptivity = double_adaptivity
        self.newton_inexactness = newton_inexactness
        self.linear_inexactness = linear_inexactness
        self.SDC_maxiter = SDC_maxiter

    def get_controller_params(self, **kwargs):
        return {'all_to_done': True}

    def get_custom_description(self, problem, num_procs=1):
        from pySDC.implementations.convergence_controller_classes.inexactness import NewtonInexactness

        preconditioner = {
            'run_Lorenz': 'MIN-SR-NS',
        }.get(problem.__name__, 'MIN-SR-S')

        desc = {}
        desc['sweeper_params'] = {'QI': preconditioner}
        desc['step_params'] = {'maxiter': self.SDC_maxiter}
        desc['problem_params'] = {}
        desc['level_params'] = {'restol': 1e-8, 'residual_type': 'last_abs'}
        desc['convergence_controllers'] = {}

        inexactness_params = {
            'min_tol': 1e-12,
            'ratio': 1e-2,
            'max_tol': 1e-4,
            'use_e_tol': False,
            'maxiter': 15,
        }

        if self.newton_inexactness and problem.__name__ not in ['run_Schroedinger', 'run_AC', 'run_RBC', 'run_GS']:
            if problem.__name__ == 'run_quench':
                inexactness_params['ratio'] = 1e-1
                inexactness_params['min_tol'] = 1e-11
                inexactness_params['maxiter'] = 5
            elif problem.__name__ == "run_vdp":
                inexactness_params['ratio'] = 1e-5
                inexactness_params['min_tol'] = 1e-15
                inexactness_params['maxiter'] = 9
            desc['convergence_controllers'][NewtonInexactness] = inexactness_params

        if problem.__name__ in ['run_vdp']:
            desc['problem_params']['stop_at_nan'] = False

        if self.linear_inexactness and problem.__name__ in ['run_quench']:
            desc['problem_params']['inexact_linear_ratio'] = 1e-1
            if problem.__name__ in ['run_quench']:
                desc['problem_params']['direct_solver'] = False
                desc['problem_params']['liniter'] = 9
                desc['problem_params']['min_lintol'] = 1e-11

        from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestarting

        desc['convergence_controllers'][BasicRestarting.get_implementation(useMPI=self.useMPI)] = {
            'max_restarts': 29,
            'crash_after_max_restarts': True,
        }
        return merge_descriptions(super().get_custom_description(problem, num_procs), desc)


class BaseStrategy(Strategy):
    '''
    Do a fixed iteration count
    '''

    def __init__(self, skip_residual_computation='all', **kwargs):
        '''
        Initialization routine
        '''
        super().__init__(skip_residual_computation=skip_residual_computation, **kwargs)
        self.color = list(cmap.values())[0]
        self.marker = 'o'
        self.name = 'base'
        self.bar_plot_x_label = 'base'
        self.precision_parameter = 'dt'
        self.precision_parameter_loc = ['level_params', 'dt']

    @property
    def label(self):
        return r'fixed'

    def get_custom_description(self, problem, num_procs):
        desc = super().get_custom_description(problem, num_procs)
        if problem.__name__ == "run_AC":
            desc['level_params']['dt'] = 1e-2 * desc['problem_params']['eps'] ** 2
        return desc

    def get_custom_description_for_faults(self, problem, num_procs, *args, **kwargs):
        desc = self.get_custom_description(problem, num_procs, *args, **kwargs)
        if problem.__name__ == "run_quench":
            desc['level_params']['dt'] = 5.0
        elif problem.__name__ == "run_AC":
            desc['level_params']['dt'] = 4e-5 if num_procs == 4 else 8e-5
        elif problem.__name__ == "run_GS":
            desc['level_params']['dt'] = 4e-1
        elif problem.__name__ == "run_vdp":
            desc['step_params'] = {'maxiter': 5}
            desc['problem_params'] = {
                'u0': np.array([2.0, 0], dtype=np.float64),
                'mu': 5,
                'crash_at_maxiter': False,
                'newton_tol': 1e-11,
                'stop_at_nan': False,
            }
            desc['level_params'] = {'dt': 4.5e-2}
        return desc

    def get_reference_value(self, problem, key, op, num_procs=1):
        """
        Get a reference value for a given problem for testing in CI.

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            key (str): The name of the variable you want to compare
            op (function): The operation you want to apply to the data
            num_procs (int): Number of processes

        Returns:
            The reference value
        """
        if problem.__name__ == "run_Lorenz":
            if key == 'work_newton' and op == sum:
                return 12350
            elif key == 'e_global_post_run' and op == max:
                return 1.3527453646133836e-07

        super().get_reference_value(problem, key, op, num_procs)


class AdaptivityStrategy(Strategy):
    '''
    Adaptivity as a resilience strategy
    '''

    def __init__(self, **kwargs):
        '''
        Initialization routine
        '''
        from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity

        kwargs['skip_residual_computation'] = 'all'
        super().__init__(**kwargs)
        self.color = list(cmap.values())[1]
        self.marker = '*'
        self.name = 'adaptivity'
        self.bar_plot_x_label = 'adaptivity'
        self.precision_parameter = 'e_tol'
        self.precision_parameter_loc = ['convergence_controllers', Adaptivity, 'e_tol']

    @property
    def label(self):
        return r'$\Delta t$-adaptivity'

    # def get_fixable_params(self, maxiter, **kwargs):
    #     """
    #     Here faults occurring in the last iteration cannot be fixed.

    #     Args:
    #         maxiter (int): Max. iterations until convergence is declared

    #     Returns:
    #         (list): Contains dictionaries of keyword arguments for `FaultStats.get_mask`
    #     """
    #     self.fixable += [
    #         {
    #             'key': 'iteration',
    #             'op': 'lt',
    #             'val': maxiter,
    #         }
    #     ]
    #     return self.fixable

    def get_custom_description(self, problem, num_procs):
        '''
        Routine to get a custom description that adds adaptivity

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            The custom descriptions you can supply to the problem when running it
        '''
        from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity
        from pySDC.implementations.convergence_controller_classes.step_size_limiter import StepSizeLimiter

        base_params = super().get_custom_description(problem, num_procs)
        custom_description = {}
        custom_description['convergence_controllers'] = {}

        dt_max = np.inf
        dt_slope_max = np.inf
        dt_slope_min = 0
        beta = 0.9

        if problem.__name__ == "run_piline":
            e_tol = 1e-7
        elif problem.__name__ == "run_vdp":
            e_tol = 2e-5
        elif problem.__name__ == "run_Lorenz":
            e_tol = 1e-6 if num_procs == 4 else 1e-7
        elif problem.__name__ == "run_Schroedinger":
            e_tol = 4e-7
        elif problem.__name__ == "run_quench":
            e_tol = 1e-8
            custom_description['problem_params'] = {
                'newton_tol': 1e-10,
                'lintol': 1e-11,
            }

            from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestarting

            custom_description['convergence_controllers'][BasicRestarting.get_implementation(useMPI=self.useMPI)] = {
                'max_restarts': 99,
            }
        elif problem.__name__ == "run_AC":
            e_tol = 1e-7
            # dt_max = 0.1 * base_params['problem_params']['eps'] ** 2
        elif problem.__name__ == 'run_RBC':
            if num_procs == 4:
                e_tol = 2e-2
            else:
                e_tol = 1e-4
            dt_slope_min = 1
            beta = 0.5
        elif problem.__name__ == 'run_GS':
            e_tol = 1e-5

        else:
            raise NotImplementedError(
                'I don\'t have a tolerance for adaptivity for your problem. Please add one to the\
 strategy'
            )

        custom_description['convergence_controllers'][Adaptivity] = {
            'e_tol': e_tol,
            'dt_slope_max': dt_slope_max,
            'dt_rel_min_slope': dt_slope_min,
            'beta': beta,
        }
        custom_description['convergence_controllers'][StepSizeLimiter] = {
            'dt_max': dt_max,
        }
        return merge_descriptions(base_params, custom_description)

    def get_reference_value(self, problem, key, op, num_procs=1):
        """
        Get a reference value for a given problem for testing in CI.

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            key (str): The name of the variable you want to compare
            op (function): The operation you want to apply to the data
            num_procs (int): Number of processes

        Returns:
            The reference value
        """
        if problem.__name__ == 'run_Lorenz':
            if key == 'work_newton' and op == sum:
                return 2989
            elif key == 'e_global_post_run' and op == max:
                return 5.636767497207984e-08

        super().get_reference_value(problem, key, op, num_procs)

    def get_custom_description_for_faults(self, problem, num_procs, *args, **kwargs):
        from pySDC.implementations.convergence_controller_classes.step_size_limiter import StepSizeLimiter
        from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity
        from pySDC.implementations.convergence_controller_classes.step_size_limiter import StepSizeSlopeLimiter
        from pySDC.projects.Resilience.RBC import ReachTendExactly

        desc = self.get_custom_description(problem, num_procs, *args, **kwargs)
        if problem.__name__ == "run_quench":
            desc['level_params']['dt'] = 1.1e1
            desc['convergence_controllers'][Adaptivity]['e_tol'] = 1e-6
        elif problem.__name__ == "run_AC":
            desc['convergence_controllers'][Adaptivity]['e_tol'] = 1e-5
        elif problem.__name__ == "run_GS":
            desc['convergence_controllers'][Adaptivity]['e_tol'] = 2e-6
        elif problem.__name__ == "run_vdp":
            desc['step_params'] = {'maxiter': 5}
            desc['sweeper_params'] = {'num_nodes': 3, 'QI': 'LU'}
            desc['problem_params'] = {
                'u0': np.array([2.0, 0], dtype=np.float64),
                'mu': 5,
                'crash_at_maxiter': True,
                'newton_tol': 1e-8,
                'stop_at_nan': True,
                'relative_tolerance': False,
            }
            desc['level_params'] = {'dt': 8e-3}
            desc['convergence_controllers'][Adaptivity]['e_tol'] = 2e-7
            desc['convergence_controllers'][ReachTendExactly] = {'Tend': 11.5}
            # desc['convergence_controllers'][StepSizeSlopeLimiter] = {'dt_slope_min': 1/4, 'dt_slope_max': 4}
        return desc


class AdaptivityRestartFirstStep(AdaptivityStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.color = 'teal'
        self.name = 'adaptivityRestartFirstStep'

    def get_custom_description(self, problem, num_procs):
        '''
        Add the other version of basic restarting.

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            The custom descriptions you can supply to the problem when running it
        '''
        custom_description = super().get_custom_description(problem, num_procs)
        from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestarting

        custom_description['convergence_controllers'][BasicRestarting.get_implementation(useMPI=self.useMPI)] = {
            'max_restarts': 15,
            'restart_from_first_step': True,
        }
        return custom_description

    @property
    def label(self):
        return f'{super().label} restart from first step'


class AdaptiveHotRodStrategy(Strategy):
    '''
    Adaptivity + Hot Rod as a resilience strategy
    '''

    def __init__(self, **kwargs):
        '''
        Initialization routine
        '''
        from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity

        kwargs['skip_residual_computation'] = 'all'
        super().__init__(**kwargs)
        self.color = list(cmap.values())[4]
        self.marker = '.'
        self.name = 'adaptive Hot Rod'
        self.bar_plot_x_label = 'adaptive\nHot Rod'
        self.precision_parameter = 'e_tol'
        self.precision_parameter_loc = ['convergence_controllers', Adaptivity, 'e_tol']

    def get_custom_description(self, problem, num_procs):
        '''
        Routine to get a custom description that adds adaptivity and Hot Rod

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            The custom description you can supply to the problem when running it
        '''
        from pySDC.implementations.convergence_controller_classes.hotrod import HotRod
        from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity

        if problem.__name__ == "run_vdp":
            e_tol = 3e-7
            dt_min = 1e-3
            maxiter = 4
            HotRod_tol = 2e-6
        elif problem.__name__ == "run_Lorenz":
            e_tol = 3e-7
            dt_min = 1e-3
            maxiter = 4
            HotRod_tol = 2e-6
        else:
            raise NotImplementedError(
                'I don\'t have a tolerance for adaptive Hot Rod for your problem. Please add one \
to the strategy'
            )

        no_storage = num_procs > 1

        custom_description = {
            'convergence_controllers': {
                HotRod: {'HotRod_tol': HotRod_tol, 'no_storage': no_storage},
                Adaptivity: {'e_tol': e_tol, 'dt_min': dt_min, 'embedded_error_flavor': 'linearized'},
            },
            'step_params': {'maxiter': maxiter},
        }

        return merge_descriptions(super().get_custom_description(problem, num_procs), custom_description)

    def get_reference_value(self, problem, key, op, num_procs=1):
        """
        Get a reference value for a given problem for testing in CI.

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            key (str): The name of the variable you want to compare
            op (function): The operation you want to apply to the data
            num_procs (int): Number of processes

        Returns:
            The reference value
        """
        if problem.__name__ == "run_Lorenz":
            if key == 'work_newton' and op == sum:
                return 5092
            elif key == 'e_global_post_run' and op == max:
                return 4.107116318152748e-06

        super().get_reference_value(problem, key, op, num_procs)


class IterateStrategy(Strategy):
    '''
    Iterate for as much as you want
    '''

    def __init__(self, **kwargs):
        '''
        Initialization routine
        '''
        kwargs['skip_residual_computation'] = 'most'
        super().__init__(**kwargs)
        self.color = list(cmap.values())[2]
        self.marker = 'v'
        self.name = 'iterate'
        self.bar_plot_x_label = 'iterate'
        self.precision_parameter = 'restol'
        self.precision_parameter_loc = ['level_params', 'restol']

    @property
    def label(self):
        return r'$k$-adaptivity'

    def get_custom_description(self, problem, num_procs):
        '''
        Routine to get a custom description that allows for adaptive iteration counts

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            The custom description you can supply to the problem when running it
        '''
        restol = -1
        e_tol = -1

        if problem.__name__ == "run_piline":
            restol = 2.3e-8
        elif problem.__name__ == "run_vdp":
            restol = 9e-7
        elif problem.__name__ == "run_Lorenz":
            restol = 16e-7
        elif problem.__name__ == "run_Schroedinger":
            restol = 6.5e-7
        elif problem.__name__ == "run_quench":
            restol = 1e-7
        elif problem.__name__ == "run_AC":
            restol = 1e-11
        elif problem.__name__ == "run_RBC":
            restol = 1e-4
        elif problem.__name__ == "run_GS":
            restol = 1e-4
        else:
            raise NotImplementedError(
                'I don\'t have a residual tolerance for your problem. Please add one to the \
strategy'
            )

        custom_description = {
            'step_params': {'maxiter': 99},
            'level_params': {'restol': restol, 'e_tol': e_tol},
        }

        if problem.__name__ == "run_quench":
            custom_description['level_params']['dt'] = 1.0

        return merge_descriptions(super().get_custom_description(problem, num_procs), custom_description)

    def get_random_params(self, problem, num_procs):
        '''
        Routine to get parameters for the randomization of faults

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            dict: Randomization parameters
        '''

        rnd_params = super().get_random_params(problem, num_procs)
        if problem.__name__ == "run_quench":
            rnd_params['iteration'] = 1
        return rnd_params

    def get_reference_value(self, problem, key, op, num_procs=1):
        """
        Get a reference value for a given problem for testing in CI.

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            key (str): The name of the variable you want to compare
            op (function): The operation you want to apply to the data
            num_procs (int): Number of processes

        Returns:
            The reference value
        """
        if problem.__name__ == "run_Lorenz":
            if key == 'work_newton' and op == sum:
                return 9200
            elif key == 'e_global_post_run' and op == max:
                return 2.139863344829962e-05

        super().get_reference_value(problem, key, op, num_procs)


class kAdaptivityStrategy(IterateStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.precision_parameter = 'dt'
        self.precision_parameter_loc = ['level_params', 'dt']

    def get_custom_description(self, problem, num_procs, *args, **kwargs):
        desc = super().get_custom_description(problem, num_procs, *args, **kwargs)
        desc['level_params']['restol'] = 1e-9
        if problem.__name__ == "run_quench":
            desc['problem_params']['newton_tol'] = 1e-9
            desc['problem_params']['lintol'] = 1e-9
            desc['level_params']['dt'] = 2.5
        elif problem.__name__ == "run_AC":
            desc['level_params']['restol'] = 1e-11
            desc['level_params']['dt'] = 0.4 * desc['problem_params']['eps'] ** 2 / 8.0
        elif problem.__name__ == "run_RBC":
            desc['level_params']['dt'] = 7e-2
            desc['level_params']['restol'] = 1e-6
            desc['level_params']['e_tol'] = 1e-7
        elif problem.__name__ == "run_GS":
            desc['level_params']['dt'] = 1.0
            desc['level_params']['restol'] = 1e-9
        return desc

    def get_custom_description_for_faults(self, problem, num_procs, *args, **kwargs):
        desc = self.get_custom_description(problem, num_procs, *args, **kwargs)
        if problem.__name__ == 'run_quench':
            desc['level_params']['dt'] = 5.0
        elif problem.__name__ == 'run_AC':
            desc['level_params']['dt'] = 4e-4 if num_procs == 4 else 5e-4
            desc['level_params']['restol'] = 1e-5 if num_procs == 4 else 1e-11
        elif problem.__name__ == 'run_RBC':
            desc['level_params']['restol'] = 1e-3 if num_procs == 4 else 1e-6
        elif problem.__name__ == 'run_Lorenz':
            desc['level_params']['dt'] = 8e-3
        elif problem.__name__ == "run_vdp":
            desc['sweeper_params'] = {'num_nodes': 3}
            desc['problem_params'] = {
                'u0': np.array([2.0, 0], dtype=np.float64),
                'mu': 5,
                'crash_at_maxiter': False,
                'newton_tol': 1e-11,
                'stop_at_nan': False,
            }
            desc['level_params'] = {'dt': 4.0e-2, 'restol': 1e-7}
        return desc

    def get_reference_value(self, problem, key, op, num_procs=1):
        """
        Get a reference value for a given problem for testing in CI.

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            key (str): The name of the variable you want to compare
            op (function): The operation you want to apply to the data
            num_procs (int): Number of processes

        Returns:
            The reference value
        """
        if problem.__name__ == "run_Lorenz":
            if key == 'work_newton' and op == sum:
                return 12350
            elif key == 'e_global_post_run' and op == max:
                return 1.3527453646133836e-07

        super().get_reference_value(problem, key, op, num_procs)


class HotRodStrategy(Strategy):
    '''
    Hot Rod as a resilience strategy
    '''

    def __init__(self, **kwargs):
        '''
        Initialization routine
        '''
        kwargs['skip_residual_computation'] = 'all'
        super().__init__(**kwargs)
        self.color = list(cmap.values())[3]
        self.marker = '^'
        self.name = 'Hot Rod'
        self.bar_plot_x_label = 'Hot Rod'
        self.precision_parameter = 'dt'
        self.precision_parameter_loc = ['level_params', 'dt']

    def get_custom_description(self, problem, num_procs):
        '''
        Routine to get a custom description that adds Hot Rod

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            The custom description you can supply to the problem when running it
        '''
        from pySDC.implementations.convergence_controller_classes.hotrod import HotRod
        from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestartingNonMPI

        base_params = super().get_custom_description(problem, num_procs)
        if problem.__name__ == "run_vdp":
            # if num_procs == 4:
            #     HotRod_tol = 1.800804e-04
            # elif num_procs == 5:
            #     HotRod_tol = 9.329361e-05
            # else:  # 1 process
            #     HotRod_tol = 1.347949e-06
            # HotRod_tol = 7e-6 if num_procs > 1 else 5e-7
            HotRod_tol = 7.2e-05
            maxiter = 6
        elif problem.__name__ == "run_Lorenz":
            if num_procs == 5:
                HotRod_tol = 9.539348e-06
            elif num_procs == 4:
                HotRod_tol = 3.201e-6
            else:
                HotRod_tol = 7.720589e-07
            maxiter = 6
        elif problem.__name__ == "run_Schroedinger":
            if num_procs == 5:
                HotRod_tol = 2.497697e-06
            elif num_procs == 4:
                HotRod_tol = 1.910405e-06
            else:
                HotRod_tol = 4.476790e-07
            maxiter = 6
        elif problem.__name__ == "run_quench":
            if num_procs == 5:
                HotRod_tol = 1.017534e-03
            elif num_procs == 4:
                HotRod_tol = 1.017534e-03
            else:
                HotRod_tol = 5.198620e-04
            maxiter = 6
        elif problem.__name__ == 'run_AC':
            HotRod_tol = 9.564437e-06
            maxiter = 6
        elif problem.__name__ == 'run_RBC':
            HotRod_tol = 3e-4 if num_procs == 4 else 6.34e-6
            maxiter = 6
        elif problem.__name__ == 'run_GS':
            HotRod_tol = 3.22e-5
            maxiter = 6
        else:
            raise NotImplementedError(
                'I don\'t have a tolerance for Hot Rod for your problem. Please add one to the\
 strategy'
            )

        no_storage = False  # num_procs > 1

        custom_description = {
            'convergence_controllers': {
                HotRod: {'HotRod_tol': HotRod_tol, 'no_storage': no_storage},
                BasicRestartingNonMPI: {
                    'max_restarts': 2,
                    'crash_after_max_restarts': False,
                    'restart_from_first_step': True,
                },
            },
            'step_params': {'maxiter': maxiter},
            'level_params': {},
        }
        if problem.__name__ == "run_AC":
            custom_description['level_params']['dt'] = 8e-5
        return merge_descriptions(base_params, custom_description)

    def get_custom_description_for_faults(self, problem, *args, **kwargs):
        desc = self.get_custom_description(problem, *args, **kwargs)
        if problem.__name__ == "run_quench":
            desc['level_params']['dt'] = 5.0
        elif problem.__name__ == "run_AC":
            desc['level_params']['dt'] = 8e-5
        elif problem.__name__ == "run_GS":
            desc['level_params']['dt'] = 4e-1
        elif problem.__name__ == "run_vdp":
            desc['step_params'] = {'maxiter': 6}
            desc['problem_params'] = {
                'u0': np.array([2.0, 0], dtype=np.float64),
                'mu': 5,
                'crash_at_maxiter': False,
                'newton_tol': 1e-11,
                'stop_at_nan': False,
            }
            desc['level_params'] = {'dt': 4.5e-2}
        return desc

    def get_reference_value(self, problem, key, op, num_procs=1):
        """
        Get a reference value for a given problem for testing in CI.

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            key (str): The name of the variable you want to compare
            op (function): The operation you want to apply to the data
            num_procs (int): Number of processes

        Returns:
            The reference value
        """
        if problem.__name__ == "run_Lorenz":
            if key == 'work_newton' and op == sum:
                return 12350
            elif key == 'e_global_post_run' and op == max:
                return 1.3527453646133836e-07

        super().get_reference_value(problem, key, op, num_procs)


class AdaptivityCollocationStrategy(InexactBaseStrategy):
    '''
    Adaptivity based on collocation as a resilience strategy
    '''

    def __init__(self, **kwargs):
        '''
        Initialization routine
        '''
        kwargs = {
            'skip_residual_computation': 'most',
            **kwargs,
        }

        from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityCollocation

        self.restol = None
        super().__init__(**kwargs)
        self.color = list(cmap.values())[1]
        self.marker = '*'
        self.name = 'adaptivity_coll'
        self.bar_plot_x_label = 'adaptivity collocation'
        self.precision_parameter = 'e_tol'
        self.adaptive_coll_params = {}
        self.precision_parameter_loc = ['convergence_controllers', AdaptivityCollocation, 'e_tol']

    def get_custom_description(self, problem, num_procs):
        '''
        Routine to get a custom description that adds adaptivity

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            The custom descriptions you can supply to the problem when running it
        '''
        from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityCollocation

        custom_description = {}

        dt_max = np.inf
        dt_min = 1e-5

        if problem.__name__ == "run_piline":
            e_tol = 1e-7
            dt_min = 1e-2
        elif problem.__name__ == "run_vdp":
            e_tol = 2e-5
            dt_min = 1e-3
        elif problem.__name__ == "run_Lorenz":
            e_tol = 2e-5
            dt_min = 1e-3
        elif problem.__name__ == "run_Schroedinger":
            e_tol = 4e-6
            dt_min = 1e-3
        elif problem.__name__ == "run_quench":
            e_tol = 1e-5
            dt_min = 1e-3
            dt_max = 1e2
        elif problem.__name__ == "run_AC":
            e_tol = 1e-4
        else:
            raise NotImplementedError(
                'I don\'t have a tolerance for adaptivity for your problem. Please add one to the\
 strategy'
            )

        custom_description['convergence_controllers'] = {
            AdaptivityCollocation: {
                'e_tol': e_tol,
                'dt_min': dt_min,
                'dt_max': dt_max,
                'adaptive_coll_params': self.adaptive_coll_params,
                'restol_rel': 1e-2,
            }
        }
        return merge_descriptions(super().get_custom_description(problem, num_procs), custom_description)


class AdaptivityCollocationTypeStrategy(AdaptivityCollocationStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.color = list(cmap.values())[4]
        self.marker = '.'
        self.adaptive_coll_params = {
            'quad_type': ['RADAU-RIGHT', 'GAUSS'],
            'do_coll_update': [False, True],
        }

    @property
    def label(self):
        return 'adaptivity type'

    def get_reference_value(self, problem, key, op, num_procs=1):
        """
        Get a reference value for a given problem for testing in CI.

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            key (str): The name of the variable you want to compare
            op (function): The operation you want to apply to the data
            num_procs (int): Number of processes

        Returns:
            The reference value
        """
        if problem.__name__ == "run_Lorenz":
            if key == 'work_newton' and op == sum:
                return 1025
            elif key == 'e_global_post_run' and op == max:
                return 4.266975256683736e-06

        super().get_reference_value(problem, key, op, num_procs)


class AdaptivityCollocationRefinementStrategy(AdaptivityCollocationStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.color = list(cmap.values())[5]
        self.marker = '^'
        self.adaptive_coll_params = {
            'num_nodes': [2, 3],
            'quad_type': ['GAUSS', 'RADAU-RIGHT'],
            'do_coll_update': [True, False],
        }

    @property
    def label(self):
        return 'adaptivity refinement'

    def get_reference_value(self, problem, key, op, num_procs=1):
        """
        Get a reference value for a given problem for testing in CI.

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            key (str): The name of the variable you want to compare
            op (function): The operation you want to apply to the data
            num_procs (int): Number of processes

        Returns:
            The reference value
        """
        if problem.__name__ == "run_Lorenz":
            if key == 'work_newton' and op == sum:
                return 917
            elif key == 'e_global_post_run' and op == max:
                return 1.0874929465387595e-05

        super().get_reference_value(problem, key, op, num_procs)


class AdaptivityCollocationDerefinementStrategy(AdaptivityCollocationStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.color = list(cmap.values())[6]
        self.marker = '^'
        self.adaptive_coll_params = {'num_nodes': [4, 3]}

    @property
    def label(self):
        return 'adaptivity de-refinement'

    def get_reference_value(self, problem, key, op, num_procs=1):
        """
        Get a reference value for a given problem for testing in CI.

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            key (str): The name of the variable you want to compare
            op (function): The operation you want to apply to the data
            num_procs (int): Number of processes

        Returns:
            The reference value
        """
        if problem.__name__ == 'run_Lorenz':
            if key == 'work_newton' and op == sum:
                return 1338
            elif key == 'e_global_post_run' and op == max:
                return 0.0001013999955041811

        super().get_reference_value(problem, key, op, num_procs)


class DIRKStrategy(AdaptivityStrategy):
    '''
    DIRK4(3)
    '''

    def __init__(self, **kwargs):
        '''
        Initialization routine
        '''
        from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityRK

        super().__init__(**kwargs)
        self.color = list(cmap.values())[7]
        self.marker = '^'
        self.name = 'DIRK'
        self.bar_plot_x_label = 'DIRK4(3)'
        self.precision_parameter = 'e_tol'
        self.precision_parameter_loc = ['convergence_controllers', AdaptivityRK, 'e_tol']
        self.max_steps = 1e5

    @property
    def label(self):
        return 'DIRK4(3)'

    def get_custom_description(self, problem, num_procs):
        '''
        Routine to get a custom description that adds adaptivity

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            The custom descriptions you can supply to the problem when running it
        '''
        from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityRK, Adaptivity
        from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestarting
        from pySDC.implementations.sweeper_classes.Runge_Kutta import DIRK43

        adaptivity_description = super().get_custom_description(problem, num_procs)

        e_tol = adaptivity_description['convergence_controllers'][Adaptivity]['e_tol']
        adaptivity_description['convergence_controllers'].pop(Adaptivity, None)
        adaptivity_description.pop('sweeper_params', None)

        rk_params = {
            'step_params': {'maxiter': 1},
            'sweeper_class': DIRK43,
            'convergence_controllers': {
                AdaptivityRK: {'e_tol': e_tol},
                BasicRestarting.get_implementation(useMPI=self.useMPI): {
                    'max_restarts': 49,
                    'crash_after_max_restarts': False,
                },
            },
        }

        custom_description = merge_descriptions(adaptivity_description, rk_params)

        return custom_description

    def get_reference_value(self, problem, key, op, num_procs=1):
        """
        Get a reference value for a given problem for testing in CI.

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            key (str): The name of the variable you want to compare
            op (function): The operation you want to apply to the data
            num_procs (int): Number of processes

        Returns:
            The reference value
        """
        if problem.__name__ == "run_Lorenz":
            if key == 'work_newton' and op == sum:
                return 5467
            elif key == 'e_global_post_run' and op == max:
                return 7.049480537091313e-07

        super().get_reference_value(problem, key, op, num_procs)

    def get_random_params(self, problem, num_procs):
        '''
        Routine to get parameters for the randomization of faults

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            dict: Randomization parameters
        '''
        rnd_params = super().get_random_params(problem, num_procs)
        rnd_params['iteration'] = 1
        rnd_params['min_node'] = 5

        return rnd_params


class ARKStrategy(AdaptivityStrategy):
    '''
    ARK5(4)
    '''

    def __init__(self, **kwargs):
        '''
        Initialization routine
        '''
        from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityRK

        super().__init__(**kwargs)
        self.color = list(cmap.values())[7]
        self.marker = 'P'
        self.name = 'ARK'
        self.bar_plot_x_label = 'ARK5(4)'
        self.precision_parameter = 'e_tol'
        self.precision_parameter_loc = ['convergence_controllers', AdaptivityRK, 'e_tol']
        self.max_steps = 1e5

    @property
    def label(self):
        return 'ARK5(4)'

    def get_custom_description(self, problem, num_procs):
        '''
        Routine to get a custom description that adds adaptivity

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            The custom descriptions you can supply to the problem when running it
        '''
        from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityRK, Adaptivity
        from pySDC.implementations.convergence_controller_classes.step_size_limiter import StepSizeSlopeLimiter
        from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestarting
        from pySDC.implementations.sweeper_classes.Runge_Kutta import ARK548L2SA

        adaptivity_description = super().get_custom_description(problem, num_procs)

        e_tol = adaptivity_description['convergence_controllers'][Adaptivity]['e_tol'] / 20.0
        adaptivity_description['convergence_controllers'].pop(Adaptivity, None)
        adaptivity_description.pop('sweeper_params', None)

        rk_params = {
            'step_params': {'maxiter': 1},
            'sweeper_class': ARK548L2SA,
            'convergence_controllers': {
                AdaptivityRK: {'e_tol': e_tol},
                BasicRestarting.get_implementation(useMPI=self.useMPI): {
                    'max_restarts': 49,
                    'crash_after_max_restarts': False,
                },
            },
        }

        if problem.__name__ == "run_RBC":
            rk_params['convergence_controllers'][StepSizeSlopeLimiter] = {'dt_rel_min_slope': 0.25}

        custom_description = merge_descriptions(adaptivity_description, rk_params)

        return custom_description

    def get_reference_value(self, problem, key, op, num_procs=1):
        """
        Get a reference value for a given problem for testing in CI.

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            key (str): The name of the variable you want to compare
            op (function): The operation you want to apply to the data
            num_procs (int): Number of processes

        Returns:
            The reference value
        """
        if problem.__name__ == "run_Schroedinger":
            if key == 'work_newton' and op == sum:
                return 0
            elif key == 'e_global_post_run' and op == max:
                return 3.1786601531890356e-08

        super().get_reference_value(problem, key, op, num_procs)


class ARK3_CFL_Strategy(BaseStrategy):
    """
    This is special for RBC with CFL number for accuracy
    """

    def __init__(self, **kwargs):
        '''
        Initialization routine
        '''
        from pySDC.implementations.problem_classes.RayleighBenard import CFLLimit

        super().__init__(**kwargs)
        self.color = 'maroon'
        self.marker = '<'
        self.name = 'ARK3'
        self.bar_plot_x_label = 'ARK3'
        self.precision_parameter = 'cfl'
        self.precision_parameter_loc = ['convergence_controllers', CFLLimit, 'cfl']
        self.max_steps = 1e5

    @property
    def label(self):
        return 'ARK3'

    def get_custom_description(self, problem, num_procs):
        '''
        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            The custom descriptions you can supply to the problem when running it
        '''
        from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestarting
        from pySDC.implementations.sweeper_classes.Runge_Kutta import ARK3
        from pySDC.implementations.problem_classes.RayleighBenard import CFLLimit
        from pySDC.implementations.convergence_controller_classes.step_size_limiter import StepSizeSlopeLimiter

        desc = super().get_custom_description(problem, num_procs)

        rk_params = {
            'step_params': {'maxiter': 1},
            'sweeper_class': ARK3,
            'convergence_controllers': {
                CFLLimit: {
                    'cfl': 0.5,
                    'dt_max': 1.0,
                },
                StepSizeSlopeLimiter: {'dt_rel_min_slope': 0.2},
            },
        }

        custom_description = merge_descriptions(desc, rk_params)

        return custom_description


class ESDIRKStrategy(AdaptivityStrategy):
    '''
    ESDIRK5(3)
    '''

    def __init__(self, **kwargs):
        '''
        Initialization routine
        '''
        from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityRK

        super().__init__(**kwargs)
        self.color = 'violet'
        self.marker = '^'
        self.name = 'ESDIRK'
        self.bar_plot_x_label = 'ESDIRK5(3)'
        self.precision_parameter = 'e_tol'
        self.precision_parameter_loc = ['convergence_controllers', AdaptivityRK, 'e_tol']
        self.max_steps = 1e5

    @property
    def label(self):
        return 'ESDIRK5(3)'

    def get_description_for_tolerance(self, problem, param, **kwargs):
        desc = {}
        if problem.__name__ == 'run_Schroedinger':
            desc['problem_params'] = {'lintol': param}
        return desc

    def get_custom_description(self, problem, num_procs):
        '''
        Routine to get a custom description that adds adaptivity

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            The custom descriptions you can supply to the problem when running it
        '''
        from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityRK, Adaptivity
        from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestarting
        from pySDC.implementations.sweeper_classes.Runge_Kutta import ESDIRK53

        adaptivity_description = super().get_custom_description(problem, num_procs)

        e_tol = adaptivity_description['convergence_controllers'][Adaptivity]['e_tol']
        adaptivity_description['convergence_controllers'].pop(Adaptivity, None)
        adaptivity_description.pop('sweeper_params', None)

        mod = 1e1 if problem.__name__ == 'run_quench' else 1.0

        rk_params = {
            'step_params': {'maxiter': 1},
            'sweeper_class': ESDIRK53,
            'convergence_controllers': {
                AdaptivityRK: {'e_tol': e_tol * mod},
                BasicRestarting.get_implementation(useMPI=self.useMPI): {
                    'max_restarts': 49,
                    'crash_after_max_restarts': False,
                },
            },
        }

        custom_description = merge_descriptions(adaptivity_description, rk_params)

        return custom_description

    def get_reference_value(self, problem, key, op, num_procs=1):
        """
        Get a reference value for a given problem for testing in CI.

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            key (str): The name of the variable you want to compare
            op (function): The operation you want to apply to the data
            num_procs (int): Number of processes

        Returns:
            The reference value
        """
        if problem.__name__ == "run_Lorenz":
            if key == 'work_newton' and op == sum:
                return 2963
            elif key == 'e_global_post_run' and op == max:
                return 4.126039954144289e-09

        super().get_reference_value(problem, key, op, num_procs)

    def get_random_params(self, problem, num_procs):
        '''
        Routine to get parameters for the randomization of faults

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            dict: Randomization parameters
        '''
        rnd_params = super().get_random_params(problem, num_procs)
        rnd_params['iteration'] = 1
        rnd_params['min_node'] = 6

        return rnd_params


class ERKStrategy(DIRKStrategy):
    """
    Explicit embedded RK using Cash-Karp's method
    """

    def __init__(self, **kwargs):
        '''
        Initialization routine
        '''
        super().__init__(**kwargs)
        self.color = list(cmap.values())[8]
        self.marker = 'x'
        self.name = 'ERK'
        self.bar_plot_x_label = 'ERK5(4)'

    def get_description_for_tolerance(self, problem, param, **kwargs):
        desc = {}
        if problem.__name__ == 'run_Schroedinger':
            desc['problem_params'] = {'lintol': param}

        return desc

    @property
    def label(self):
        return 'CK5(4)'

    def get_random_params(self, problem, num_procs):
        '''
        Routine to get parameters for the randomization of faults

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            dict: Randomization parameters
        '''
        rnd_params = super().get_random_params(problem, num_procs)
        rnd_params['min_node'] = 7

        return rnd_params

    def get_custom_description(self, problem, num_procs=1):
        from pySDC.implementations.sweeper_classes.Runge_Kutta import Cash_Karp

        desc = super().get_custom_description(problem, num_procs)
        desc['sweeper_class'] = Cash_Karp

        if problem.__name__ == "run_AC":
            desc['level_params']['dt'] = 2e-5
        return desc

    def get_reference_value(self, problem, key, op, num_procs=1):
        """
        Get a reference value for a given problem for testing in CI.

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            key (str): The name of the variable you want to compare
            op (function): The operation you want to apply to the data
            num_procs (int): Number of processes

        Returns:
            The reference value
        """
        if problem.__name__ == "run_Lorenz":
            if key == 'work_newton' and op == sum:
                return 0
            elif key == 'e_global_post_run' and op == max:
                return 1.509206128957885e-07

        super().get_reference_value(problem, key, op, num_procs)


class DoubleAdaptivityStrategy(AdaptivityStrategy):
    '''
    Adaptivity based both on embedded estimate and on residual
    '''

    def __init__(self, **kwargs):
        '''
        Initialization routine
        '''
        from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity

        kwargs['skip_residual_computation'] = 'all'
        super().__init__(**kwargs)
        self.color = list(cmap.values())[7]
        self.marker = '^'
        self.name = 'double_adaptivity'
        self.bar_plot_x_label = 'double adaptivity'
        self.precision_parameter = 'e_tol'
        self.precision_parameter_loc = ['convergence_controllers', Adaptivity, 'e_tol']
        self.residual_e_tol_ratio = 1.0
        self.residual_e_tol_abs = None

    @property
    def label(self):
        return 'double adaptivity'

    def get_custom_description(self, problem, num_procs):
        '''
        Routine to get a custom description that adds adaptivity

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            The custom descriptions you can supply to the problem when running it
        '''
        from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityResidual, Adaptivity
        from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestarting

        custom_description = super().get_custom_description(problem, num_procs)

        if self.residual_e_tol_abs:
            e_tol = self.residual_e_tol_abs
        else:
            e_tol = custom_description['convergence_controllers'][Adaptivity]['e_tol'] * self.residual_e_tol_ratio
        custom_description['convergence_controllers'][AdaptivityResidual] = {
            'e_tol': e_tol,
            'allowed_modifications': ['decrease'],
        }

        custom_description['convergence_controllers'][BasicRestarting.get_implementation(useMPI=self.useMPI)] = {
            'max_restarts': 15
        }

        return custom_description

    def get_reference_value(self, problem, key, op, num_procs=1):
        """
        Get a reference value for a given problem for testing in CI.

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            key (str): The name of the variable you want to compare
            op (function): The operation you want to apply to the data
            num_procs (int): Number of processes

        Returns:
            The reference value
        """
        if problem.__name__ == 'run_Lorenz':
            if key == 'work_newton' and op == sum:
                return 2989
            elif key == 'e_global_post_run' and op == max:
                return 5.636763944494305e-08

        super().get_reference_value(problem, key, op, num_procs)


class AdaptivityAvoidRestartsStrategy(AdaptivityStrategy):
    """
    Adaptivity with the avoid restarts option
    """

    @property
    def label(self):
        return 'adaptivity (avoid restarts)'

    def get_custom_description(self, problem, num_procs):
        '''
        Routine to get a custom description that adds adaptivity

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            The custom descriptions you can supply to the problem when running it
        '''
        from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity
        from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestarting

        custom_description = super().get_custom_description(problem, num_procs)

        custom_description['convergence_controllers'][Adaptivity]['avoid_restarts'] = True

        custom_description['convergence_controllers'][BasicRestarting.get_implementation(useMPI=self.useMPI)] = {
            'max_restarts': 15
        }

        return custom_description

    def get_reference_value(self, problem, key, op, num_procs=1):
        """
        Get a reference value for a given problem for testing in CI.

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            key (str): The name of the variable you want to compare
            op (function): The operation you want to apply to the data
            num_procs (int): Number of processes

        Returns:
            The reference value
        """
        if problem.__name__ == "run_Lorenz":
            if key == 'work_newton' and op == sum:
                return 2989
            elif key == 'e_global_post_run' and op == max:
                return 5.636763944494305e-08

        super().get_reference_value(problem, key, op, num_procs)


class AdaptivityInterpolationStrategy(AdaptivityStrategy):
    """
    Adaptivity with interpolation between restarts
    """

    @property
    def label(self):
        return 'adaptivity+interpolation'

    def get_custom_description(self, problem, num_procs):
        '''
        Routine to get a custom description that adds adaptivity

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            The custom descriptions you can supply to the problem when running it
        '''
        from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity
        from pySDC.implementations.convergence_controller_classes.interpolate_between_restarts import (
            InterpolateBetweenRestarts,
        )
        from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestarting

        custom_description = super().get_custom_description(problem, num_procs)

        custom_description['convergence_controllers'][Adaptivity]['avoid_restarts'] = False
        custom_description['convergence_controllers'][InterpolateBetweenRestarts] = {}

        custom_description['convergence_controllers'][BasicRestarting.get_implementation(useMPI=self.useMPI)] = {
            'max_restarts': 15
        }

        return custom_description

    def get_reference_value(self, problem, key, op, num_procs=1):
        """
        Get a reference value for a given problem for testing in CI.

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            key (str): The name of the variable you want to compare
            op (function): The operation you want to apply to the data
            num_procs (int): Number of processes

        Returns:
            The reference value
        """
        if problem.__name__ == "run_Lorenz":
            if key == 'work_newton' and op == sum:
                return 6659
            elif key == 'e_global_post_run' and op == max:
                return 2.9780002756552015e-06

        super().get_reference_value(problem, key, op, num_procs)


class AdaptivityExtrapolationWithinQStrategy(InexactBaseStrategy):
    '''
    Adaptivity based on extrapolation between collocation nodes as a resilience strategy
    '''

    def __init__(self, **kwargs):
        '''
        Initialization routine
        '''
        from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityExtrapolationWithinQ

        self.restol = None
        super().__init__(**kwargs)
        self.color = list(cmap.values())[8]
        self.marker = '*'
        self.name = 'adaptivity_extraQ'
        self.bar_plot_x_label = 'adaptivity Q'
        self.precision_parameter = 'e_tol'
        self.adaptive_coll_params = {}
        self.precision_parameter_loc = ['convergence_controllers', AdaptivityExtrapolationWithinQ, 'e_tol']

    def get_custom_description(self, problem, num_procs):
        '''
        Routine to get a custom description that adds adaptivity

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            The custom descriptions you can supply to the problem when running it
        '''
        from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityExtrapolationWithinQ

        custom_description = {}

        dt_max = np.inf
        dt_min = 1e-5

        if problem.__name__ == "run_vdp":
            e_tol = 2e-5
            dt_min = 1e-3
        elif problem.__name__ == "run_piline":
            e_tol = 1e-7
            dt_min = 1e-2
        elif problem.__name__ == "run_Lorenz":
            e_tol = 2e-5
            dt_min = 1e-3
        elif problem.__name__ == "run_Schroedinger":
            e_tol = 4e-6
            dt_min = 1e-3
        elif problem.__name__ == "run_quench":
            e_tol = 1e-5
            dt_min = 1e-3
            dt_max = 1e2
        elif problem.__name__ == "run_AC":
            e_tol = 1e-4
        else:
            raise NotImplementedError(
                'I don\'t have a tolerance for adaptivity for your problem. Please add one to the\
 strategy'
            )

        custom_description['convergence_controllers'] = {
            AdaptivityExtrapolationWithinQ: {
                'e_tol': e_tol,
                'dt_min': dt_min,
                'dt_max': dt_max,
                'restol_rel': 1e-2,
                'restart_at_maxiter': True,
            }
        }
        return merge_descriptions(super().get_custom_description(problem, num_procs), custom_description)

    def get_reference_value(self, problem, key, op, num_procs=1):
        """
        Get a reference value for a given problem for testing in CI.

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            key (str): The name of the variable you want to compare
            op (function): The operation you want to apply to the data
            num_procs (int): Number of processes

        Returns:
            The reference value
        """
        if problem.__name__ == "run_Lorenz":
            if key == 'work_newton' and op == sum:
                return 2198
            elif key == 'e_global_post_run' and op == max:
                return 5.412657451131508e-07

        super().get_reference_value(problem, key, op, num_procs)


class AdaptivityPolynomialError(InexactBaseStrategy):
    '''
    Adaptivity based on extrapolation between collocation nodes as a resilience strategy
    '''

    def __init__(self, interpolate_between_restarts=False, use_restol_rel=True, max_slope=4, **kwargs):
        '''
        Initialization routine
        '''
        from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityPolynomialError

        self.restol = None
        super().__init__(**kwargs)
        self.color = list(cmap.values())[9]
        self.marker = '+'
        self.name = 'adaptivity-inter'
        self.bar_plot_x_label = 'adaptivity Q'
        self.precision_parameter = 'e_tol'
        self.adaptive_coll_params = {}
        self.precision_parameter_loc = ['convergence_controllers', AdaptivityPolynomialError, 'e_tol']
        self.interpolate_between_restarts = interpolate_between_restarts
        self.use_restol_rel = use_restol_rel
        self.max_slope = max_slope

    def get_custom_description(self, problem, num_procs):
        '''
        Routine to get a custom description that adds adaptivity

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            The custom descriptions you can supply to the problem when running it
        '''
        from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityPolynomialError
        from pySDC.implementations.convergence_controller_classes.step_size_limiter import StepSizeLimiter
        from pySDC.implementations.convergence_controller_classes.check_convergence import CheckConvergence

        base_params = super().get_custom_description(problem, num_procs)
        custom_description = {}

        dt_max = np.inf
        restol_rel = 1e-4
        restol_min = 1e-12
        restol_max = 1e-5
        dt_slope_min = 0
        dt_min = 0
        abort_at_growing_residual = True
        level_params = {}
        problem_params = {}
        beta = 0.9

        if problem.__name__ == "run_vdp":
            e_tol = 6e-4
            level_params['dt'] = 0.1
            restol_rel = 1e-5
            restol_min = 1e-12
            dt_min = 1e-7
            problem_params['newton_tol'] = 1e-14
        elif problem.__name__ == "run_piline":
            e_tol = 1e-7
        elif problem.__name__ == "run_Lorenz":
            e_tol = 2e-4
        elif problem.__name__ == "run_Schroedinger":
            e_tol = 3e-5
        elif problem.__name__ == "run_quench":
            e_tol = 1e-7
            level_params['dt'] = 50.0
            restol_min = 1e-11
            restol_rel = 1e-1
        elif problem.__name__ == "run_AC":
            e_tol = 1.0e-4
            restol_rel = 1e-3 if num_procs == 4 else 1e-3
            # dt_max = 0.1 * base_params['problem_params']['eps'] ** 2
        elif problem.__name__ == "run_RBC":
            e_tol = 5e-2 if num_procs == 4 else 5e-3
            dt_slope_min = 1.0
            abort_at_growing_residual = False
            restol_rel = 1e-2 if num_procs == 4 else 1e-4
            restol_max = 1e-1
            restol_min = 5e-8
            self.max_slope = 4
            beta = 0.5
            level_params['e_tol'] = 1e-5
        elif problem.__name__ == 'run_GS':
            e_tol = 1e-4
            restol_rel = 4e-3
            restol_max = 1e-4
            restol_min = 1e-9
        else:
            raise NotImplementedError(
                'I don\'t have a tolerance for adaptivity for your problem. Please add one to the\
 strategy'
            )

        custom_description['convergence_controllers'] = {
            AdaptivityPolynomialError: {
                'e_tol': e_tol,
                'restol_rel': restol_rel if self.use_restol_rel else 1e-11,
                'restol_min': restol_min if self.use_restol_rel else 1e-12,
                'restol_max': restol_max if self.use_restol_rel else 1e-5,
                'restart_at_maxiter': True,
                'factor_if_not_converged': self.max_slope,
                'interpolate_between_restarts': self.interpolate_between_restarts,
                'abort_at_growing_residual': abort_at_growing_residual,
                'beta': beta,
            },
            StepSizeLimiter: {
                'dt_max': dt_max,
                'dt_slope_max': self.max_slope,
                'dt_min': dt_min,
                'dt_rel_min_slope': dt_slope_min,
            },
        }
        custom_description['level_params'] = level_params
        custom_description['problem_params'] = problem_params

        return merge_descriptions(base_params, custom_description)

    def get_custom_description_for_faults(self, problem, num_procs, *args, **kwargs):
        desc = self.get_custom_description(problem, num_procs, *args, **kwargs)
        if problem.__name__ == "run_quench":
            from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityPolynomialError

            desc['convergence_controllers'][AdaptivityPolynomialError]['e_tol'] = 1e-7 * 11
            desc['level_params']['dt'] = 4.0
        elif problem.__name__ == "run_AC":
            from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityPolynomialError

            desc['convergence_controllers'][AdaptivityPolynomialError]['e_tol'] = 6e-3 if num_procs == 4 else 1e-3
            if num_procs == 4:
                desc['step_params'] = {'maxiter': 50}
        elif problem.__name__ == "run_Lorenz":
            from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityPolynomialError

            desc['convergence_controllers'][AdaptivityPolynomialError]['e_tol'] = 2e-4
            desc['convergence_controllers'][AdaptivityPolynomialError]['restol_min'] = 1e-11
            desc['convergence_controllers'][AdaptivityPolynomialError]['restol_rel'] = 1e-11
        elif problem.__name__ == "run_vdp":
            from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityPolynomialError
            from pySDC.implementations.convergence_controller_classes.inexactness import NewtonInexactness

            desc['step_params'] = {'maxiter': 16}
            desc['problem_params'] = {
                'u0': np.array([2.0, 0], dtype=np.float64),
                'mu': 5,
                'crash_at_maxiter': False,
                'newton_tol': 1e-11,
                'stop_at_nan': False,
            }
            desc['convergence_controllers'][AdaptivityPolynomialError]['e_tol'] = 5e-4
            desc['convergence_controllers'][AdaptivityPolynomialError]['restol_rel'] = 8e-5
            desc['convergence_controllers'].pop(NewtonInexactness)
            desc['level_params'] = {'dt': 4.5e-2}
        return desc

    def get_random_params(self, problem, num_procs):
        '''
        Routine to get parameters for the randomization of faults

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            dict: Randomization parameters
        '''

        rnd_params = super().get_random_params(problem, num_procs)
        if problem.__name__ == "run_quench":
            rnd_params['iteration'] = 1
        elif problem.__name__ == 'run_Lorenz':
            rnd_params['iteration'] = 5
        return rnd_params

    def get_reference_value(self, problem, key, op, num_procs=1):
        """
        Get a reference value for a given problem for testing in CI.

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            key (str): The name of the variable you want to compare
            op (function): The operation you want to apply to the data
            num_procs (int): Number of processes

        Returns:
            The reference value
        """
        if problem.__name__ == "run_Lorenz":
            if key == 'work_newton' and op == sum:
                return 2123
            elif key == 'e_global_post_run' and op == max:
                return 7.931560830343187e-08

        super().get_reference_value(problem, key, op, num_procs)

    @property
    def label(self):
        return r'$\Delta t$-$k$-adaptivity'
