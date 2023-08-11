import imp
import os
import sys

from natsort import natsorted
import torch
import numpy as np
import json
from ignite.engine import (Engine,
                           Events)
from ignite.handlers import ModelCheckpoint

from .radam import RAdam
from .trackers import progress_report

class experiment(object):
    """
    Parse and set up arguments, set up the model and optimizer, and prepare 
    the folder into which to save the experiment.

    In args, expecting `model_from` and `path`.
    """
    def __init__(self, args):
        self.args = args
        self._epoch = [0]
        self.experiment_path = args.path
        
        # Make experiment directory, if necessary.
        if not os.path.exists(args.path):
            os.makedirs(args.path)

        # Get model config.
        # 
        # NOTE: assuming that this always exists. Currently, dispatch
        # reloads all initial arguments on resume, so `model_from` is
        # defined even on resume. Taking `model_from` always from args
        # instead of from saved state allows it to be overriden on resume
        # by passing the argument anew.
        # 
        # NOTE: the model config file has to exist in its original location
        # when resuming an experiment.
        with open(args.model_from) as f:
            self.model_as_str = f.read()

        # Initialize model and optimizer.
        model, optimizer = self._init_state(
                                     optimizer_name=args.optimizer,
                                     learning_rate=args.learning_rate,
                                     opt_kwargs=args.opt_kwargs,
                                     weight_decay=args.weight_decay,
                                     model_kwargs=args.model_kwargs)
            
        # Does the experiment directory already contain state files?
        state_file_list = natsorted([fn for fn in os.listdir(args.path)
                                     if fn.startswith('state_dict_')
                                     and fn.endswith('.pth')])
        
        # If yes, resume; else, initialize a new experiment.
        if os.path.exists(args.path) and len(state_file_list):
            # RESUME old experiment
            state_file = state_file_list[-1]
            state_from = os.path.join(args.path, state_file)
            print("Resuming from {}.".format(state_from))
            self._load_state(load_from=state_from,
                             model=model, optimizer=optimizer)
        else:
            # INIT new experiment
            with open(os.path.join(args.path, "args.txt"), 'w') as f:
                f.write('\n'.join(sys.argv))
            if args.weights_from is not None:
                # Load weights from specified checkpoint.
                self._load_state(load_from=args.weights_from, model=model)
                self._epoch[0] = 0
            with open(os.path.join(args.path, "config.py"), 'w') as f:
                f.write(self.model_as_str)
        
        # Initialization complete. Store objects, etc.
        self.model = model
        self.optimizer = optimizer
        print("Number of parameters\n"+
              "\n".join([" {} : {}".format(key, count_params(self.model[key]))
                         for key in self.model.keys()
                         if hasattr(self.model[key], 'parameters')]))
    
    def setup_engine(self, function,
                     append=True, prefix=None, epoch_length=None):
        engine = Engine(function)
        fn = "log.txt" if prefix is None else "{}_log.txt".format(prefix)
        progress = progress_report(
            prefix=prefix,
            append=append,
            epoch_length=epoch_length,
            log_path=os.path.join(self.experiment_path, fn))
        progress.attach(engine)
        return engine
    
    def setup_checkpoints(self, trainer, evaluator, score_function, n_saved=2):
        # Checkpoint at every epoch and increment epoch in
        # `self.model_save_dict`.
        checkpoint_last_handler = ModelCheckpoint(
                                    dirname=self.experiment_path,
                                    filename_prefix='state',
                                    n_saved=n_saved,
                                    save_interval=1,
                                    atomic=True,
                                    create_dir=True,
                                    require_empty=False,
                                    save_as_state_dict=False)
        
        # Checkpoint for best model performance.
        checkpoint_best_handler = ModelCheckpoint(
                                    dirname=self.experiment_path,
                                    filename_prefix='best_state',
                                    n_saved=1,
                                    score_function=score_function,
                                    atomic=True,
                                    create_dir=True,
                                    require_empty=False,
                                    save_as_state_dict=False)
        
        # Save checkpoint histories in a lightweight checkpoint.
        # If experiment state is resumed, so are these histories.
        checkpoint_hist_handler = ModelCheckpoint(
                                    dirname=self.experiment_path,
                                    filename_prefix='checkpoint_history',
                                    n_saved=n_saved,
                                    save_interval=1,
                                    atomic=True,
                                    create_dir=True,
                                    require_empty=False,
                                    save_as_state_dict=False)
        
        # The lists of saved checkpoints is stored in the experiment object.
        # If experiment state is resumed, so are these lists.
        # 
        # Monkey-patch the checkpoint handlers to use _these_ lists rather
        # than their own which are always initialized empty, even when
        # resuming an experiment.
        checkpoint_last_handler._saved = self._checkpoint_saved_last
        checkpoint_best_handler._saved = self._checkpoint_saved_best
        
        # Track epoch in the experiment object, the training engine state,
        # and the checkpoint handlers.
        # 
        # This allows the correct epoch to be reported, training to stop after
        # the correct number of epochs, resumed experiments to start on the
        # correct epoch, and saved states to be enumerated with the correct
        # epoch.
        trainer.add_event_handler(Events.STARTED,
                                  lambda engine: setattr(engine.state,
                                                         "epoch",
                                                         self._epoch[0]))
        checkpoint_last_handler._iteration = self._epoch[0]
        checkpoint_best_handler._iteration = self._epoch[0]
        checkpoint_hist_handler._iteration = self._epoch[0]
        
        # Function to update state dicts, call checkpoint handler, and
        # increment epoch count. Runs at the end of each epoch.
        # 
        # Also stores the '_saved' list of previous checkpoints from all
        # checkpoint handlers, so that previous checkpoints could be cleaned
        # up when an experiment is resumed.
        # 
        # To support saving some problematic layers (eg. layer normalization),
        # `state_dict()` is called every time a state dict is needed (every
        # time that a state is saved).
        # 
        # NOTE: `checkpoint_handler` only uses `engine` by applying
        # `score_function` to it -- so only the `checkpoint_best_handler`
        # needs an enginer (the `evaluator` engine). This engine can safely
        # be passed to the `checkpoint_last_handler`, too, since it will not
        # be used there.
        def call_checkpoint_handlers(engine):
            self._epoch[0] = trainer.state.epoch  # Update the epoch count.
            model_save_dict = {'dict': {
                'epoch'        : self._epoch,
                'model_as_str' : self.model_as_str
                }}
            for key in self.model.keys():
                if hasattr(self.model[key], 'parameters'):
                    _optimizer_state = self.optimizer[key].state_dict()
                else:
                    _optimizer_state = None
                model_save_dict['dict'][key] = {
                    'model_state'     : self.model[key].state_dict(),
                    'optimizer_state' : _optimizer_state}
            checkpoint_last_handler(engine, model_save_dict)                
            checkpoint_best_handler(engine, model_save_dict)
            hist_save_dict = {'dict': {
                'last_checkpoint' : checkpoint_last_handler._saved,
                'best_checkpoint' : checkpoint_best_handler._saved
                }}
            checkpoint_hist_handler(engine, hist_save_dict)
        evaluator.add_event_handler(Events.EPOCH_COMPLETED,
                                    call_checkpoint_handlers)
        
    
    def get_epoch(self):
        return self._epoch[0]
    
    def _init_state(self, optimizer_name, learning_rate=0.,
                    opt_kwargs=None, weight_decay=0., model_kwargs=None):
        '''
        Initialize the model, its state, and the optimizer's state.
        
        Requires the model to be defined in `self.model_as_str`.
        '''
        
        # Build the model.
        if model_kwargs is None:
            model_kwargs = {}
        module = imp.new_module('module')
        exec(self.model_as_str, module.__dict__)
        model = getattr(module, 'build_model')(**model_kwargs)
        if not isinstance(model, dict):
            model = {'model': model}
        
        # If optimizer_name is in JSON format, convert string to dict.
        try:
            optimizer_name = json.loads(optimizer_name)
        except ValueError:
            # Not in JSON format; keep as a string.
            optimizer_name = optimizer_name
        
        # Setup the optimizer.
        optimizer = {}
        for key in model.keys():
            if not hasattr(model[key], 'parameters'):
                continue    # Model has no parameters and cannot be optimized.
            def parse(arg):
                # Helper function for args when passed as dict, with
                # model names as keys.
                if isinstance(arg, dict) and key in arg:
                    return arg[key]
                return arg
            model[key].cuda()
            optimizer[key] = self._get_optimizer(
                                            name=parse(optimizer_name),
                                            params=model[key].parameters(),
                                            lr=parse(learning_rate),
                                            opt_kwargs=parse(opt_kwargs),
                                            weight_decay=weight_decay)
        
        # Store the lists of past checkpoints in the experiment object.
        # (checkpoint handler internals -- used for resuming handler state).
        self._checkpoint_saved_last = []
        self._checkpoint_saved_best = []
        
        return model, optimizer
    
    def _load_state(self, load_from, model, optimizer=None):
        '''
        Restore the model, its state, and the optimizer's state.
        '''
        saved_dict = torch.load(load_from)
        for key in model.keys():
            model[key].load_state_dict(saved_dict[key]['model_state'])
            if optimizer is not None and hasattr(model[key], 'parameters'):
                optimizer[key].load_state_dict(
                                   saved_dict[key]['optimizer_state'])

        # Check if the model config was the same as what is initialized now.
        if saved_dict['model_as_str'] != self.model_as_str:
            print("NOTE : model configuration differs from the one used with "
                  "the last saved state. Using the new configuration.")
        
        # Experiment metadata.
        self._epoch[0] = saved_dict['epoch'][0]
        
        # Store the lists of past checkpoints in the experiment object.
        # (checkpoint handler internals -- used for resuming handler state).
        def clean(checkpoint_list):
            # Keep only the paths that exist.
            valid_checkpoint_list = []
            for priority, paths in checkpoint_list:
                if np.all([os.path.exists(p) for p in paths]):
                    valid_checkpoint_list.append((priority, paths))   
            return valid_checkpoint_list
        hist_file_list = natsorted(
            [fn for fn in os.listdir(self.experiment_path)
             if fn.startswith('checkpoint_history_dict_')
             and fn.endswith('.pth')])
        if len(hist_file_list):
            hist_path = os.path.join(self.experiment_path, hist_file_list[-1])
            hist_dict = torch.load(hist_path)
            self._checkpoint_saved_last = clean(hist_dict['last_checkpoint'])
            self._checkpoint_saved_best = clean(hist_dict['best_checkpoint'])
    
    def load_last_state(self):
        state_file = natsorted([fn for fn in os.listdir(self.experiment_path)
                                if fn.startswith('state_dict_')
                                and fn.endswith('.pth')])[-1]
        state_from = os.path.join(self.experiment_path, state_file)
        self._load_state(load_from=state_from,
                         model=self.model,
                         optimizer=self.optimizer)
    
    def load_best_state(self):
        state_file = natsorted([fn for fn in os.listdir(self.experiment_path)
                                if fn.startswith('best_state_dict_')
                                and fn.endswith('.pth')])[-1]
        state_from = os.path.join(self.experiment_path, state_file)
        self._load_state(load_from=state_from,
                         model=self.model,
                         optimizer=self.optimizer)

    def _get_optimizer(self, name, params, lr=0., opt_kwargs=None, 
                       weight_decay=0.):
        kwargs = {'params'       : [p for p in params if p.requires_grad],
                  'lr'           : lr,
                  'weight_decay' : weight_decay}
        optimizer = None
        if name=='adam' or name=='amsgrad':
            if opt_kwargs is None:
                opt_kwargs = {'betas': (0.5, 0.999)}
            kwargs.update(opt_kwargs)
            optimizer = torch.optim.Adam(amsgrad=bool(name=='amsgrad'),
                                         **kwargs)
        elif name=='radam':
            if opt_kwargs is None:
                opt_kwargs = {'betas': (0.5, 0.999)}
            kwargs.update(opt_kwargs)
            optimizer = RAdam(**kwargs)
        elif name=='rmsprop':
            if opt_kwargs is None:
                opt_kwargs = {'alpha': 0.5}
            kwargs.update(opt_kwargs)
            optimizer = torch.optim.RMSprop(**kwargs)
        elif name=='sgd':
            optimizer = torch.optim.SGD(**kwargs)
        else:
            raise ValueError("Optimizer {} not supported.".format(name))
        return optimizer


def count_params(module, trainable_only=True):
    """
    Count the number of parameters in a module.
    """
    parameters = module.parameters()
    if trainable_only:
        parameters = filter(lambda p: p.requires_grad, parameters)
    num = sum([np.prod(p.size()) for p in parameters])
    return num 
