from collections import (defaultdict,
                         OrderedDict)

import os
import warnings

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorboardX as tb
import torch
from tqdm import tqdm
from ignite.engine import Events
from ignite.metrics import Metric


class progress_report(object):
    """
    Log progress to screen and/or file. Relies on `Metric` objects to
    accumulate stats in the way that they should be reported every
    iteration.
    """
    def __init__(self, epoch_length=None, prefix=None, progress_bar=True,
                 log_path=None, append=False):
        self.epoch_length = epoch_length
        self.prefix = prefix
        self.progress_bar = progress_bar
        self.log_path = log_path
        self._pbar = None
        if log_path is not None and not append:
            # If we're not resuming the experiment,
            # make the log file blank.
            open(log_path, 'w').close()

    def log_string(self, desc, metrics, file=None):
        mstr = " ".join(["{}={:.3}".format(*x) for x in metrics.items()])
        desc = "{}: {}".format(desc, mstr) if len(desc) else mstr
        print(desc, file=file)

    def __call__(self, engine):
        # If no epoch length is defined, try to get it from the data loader.
        if self.epoch_length is None:
            if hasattr(engine.state.dataloader, '__len__'):
                self.epoch_length = len(engine.state.dataloader)
            else:
                self.epoch_length = np.inf
        
        # Setup prefix.
        prefix = ""
        if self.prefix is not None:
            prefix = "{}_".format(self.prefix)
        
        # Print to screen.
        desc = ""
        metrics = {}
        if hasattr(engine.state, 'metrics'):
            metrics = OrderedDict(
                [(prefix+key, val.item())
                 if isinstance(val, torch.Tensor)
                 else (prefix+key, val)
                 for key, val in engine.state.metrics.items()])
        if hasattr(engine.state, 'epoch'):
            desc += "Epoch {}".format(engine.state.epoch)
        if self.progress_bar:
            if self._pbar is None:
                self._pbar = tqdm(total=self.epoch_length, desc=desc)
            self._pbar.set_postfix(metrics)
            self._pbar.update(1)
            if engine.state.iteration % self.epoch_length == 0:
                self._pbar.close()
                self._pbar = None
                if self.log_path is not None:
                    with open(self.log_path, 'a') as logfile:
                        self.log_string(desc, metrics, file=logfile)
        else:
            self.log_string(desc, metrics)
    
    def attach(self, engine):
        engine.add_event_handler(Events.ITERATION_COMPLETED, self)


class scoring_function(object):
    """
    metric_name : metric to monitor. This is one of the keys in
        `engine.state.metrics` of the trainer.
    period : only consider saving the best model every this many epochs.
    """
    def __init__(self, metric_name=None, period=1):
        if metric_name is None:
            metric_name = 'val_loss'
        self.metric_name = metric_name
        self.period = period
        self.epochs_since_last_save = 0
        
    def __call__(self, engine):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            quantity = engine.state.metrics[self.metric_name]
            # Since we're always trying to minimize things, return
            # the -ve of whatever that is.
            return -quantity
        else:
            return -np.inf


class summary_tracker(object):
    """
    Accumulate statistics.
    
    path : path to save tensorboard event files to.
    """
    def __init__(self, path, initial_epoch=1):
        self.path = path
        self.initial_epoch = initial_epoch
        self.summary_writer = tb.SummaryWriter(path)
        self._metric_value_dict = []
        self._item_counter = []
        self._output_transform = []
        self._epoch = []
    
    def _iteration_completed(self, engine, prefix, idx):
        output = self._output_transform[idx](engine.state.output)
        self._update(output, prefix, idx)
    
    def _epoch_completed(self, engine, prefix, idx, metric_keys=None):
        if hasattr(engine.state, 'metrics'):
            # Collect metrics stored in engine state.
            # (Assuming these are already accumulated over an epoch.)
            for key in metric_keys:
                key_ = key
                if prefix is not None:
                    key_ = "{}_{}".format(prefix, key)
                self._metric_value_dict[idx][key_] = engine.state.metrics[key]
        self._write(self._epoch[idx], idx)
        self._epoch[idx] += 1
    
    def _update(self, output, prefix, idx):
        value_dict = OrderedDict()
        for key in output.keys():
            # Convert torch tensors to numpy.
            val = output[key]
            if isinstance(val, torch.Tensor):
                value_dict[key] = val.cpu().numpy()
        for key, val in value_dict.items():
            # Skip values with nan or inf. This is useful for metrics like
            # gradient norm when using automated mixed precision since it
            # causes occasional instability -- the optimizer automatically
            # skips the update step at these times.
            if np.any(np.isnan(val)):
                warnings.warn("nan in {} - skipping in summary_tracker"
                              "".format(key))
                continue
            if np.any(np.isinf(val)):
                warnings.warn("inf in {} - skipping in summary_tracker"
                              "".format(key))
                continue
            
            # Prefix key, if requested.
            if prefix is not None:
                key = "{}_{}".format(prefix, key)
                
            # Initialize.
            for k in [key+'_mean', key+'_std', key+'_min', key+'_max']:
                if k not in self._metric_value_dict[idx].keys():
                    self._metric_value_dict[idx][k] = 0

            # Size of value (1 if scalar).
            count = np.product(np.shape(val)) or 1
            
            # Running stats so far.
            mean_a = self._metric_value_dict[idx][key+'_mean']
            var_a  = self._metric_value_dict[idx][key+'_std']
            min_a  = self._metric_value_dict[idx][key+'_min']
            max_a  = self._metric_value_dict[idx][key+'_max']
            
            # Stats on this value.
            mean_b = np.mean(val)
            var_b  = np.var(val)
            min_b  = np.min(val)
            max_b  = np.max(val)
            
            # Update running stats with stats of this value.
            a, b = self._item_counter[idx][key], count
            mean = (mean_a*a + mean_b*b) / float(a+b)
            if len(np.shape(val))==0:
                # Scalar.
                var = (var_a*a+(val-mean_a)*(val-mean))/float(a+b)
            else:
                # Tensor.
                var = ( var_a*a + var_b*b
                       +(mean_b-mean_a)**2 * a*b/float(a+b)  )/float(a+b)
            self._metric_value_dict[idx][key+'_mean'] = mean
            self._metric_value_dict[idx][key+'_std']  = var
            self._metric_value_dict[idx][key+'_min']  = min(min_a, min_b)
            self._metric_value_dict[idx][key+'_max']  = max(max_a, max_b)
                
            # Increment counter.
            self._item_counter[idx][key] += count
    
    def _write(self, epoch, idx):
        '''
        Write summaries to event. Increments step count.
        
        global_step : typically, the current epoch.
        '''
        for key, val in self._metric_value_dict[idx].items():
            if key.endswith('_std'):
                # Turn variance into standard deviation
                val = np.sqrt(val)
            self.summary_writer.add_scalar(tag=key,
                                           scalar_value=val,
                                           global_step=epoch)
            self.summary_writer.file_writer.flush()
        self._item_counter[idx] = defaultdict(int)
        self._metric_value_dict[idx] = OrderedDict()
    
    def attach(self, engine, prefix=None, output_transform=lambda x:x,
               metric_keys=None):
        '''
        prefix : A string to prefix onto the value names.
        output_transform : function mapping Engine output to a dictionary
            of the form: `(key, (tensor, n_items))`. Stats accumulated for
            each tensor, assuming that it corresponds to `n_items` elements.
        metric_keys : A list of keys (string) for metrics in `engine.metrics`
            to log. Assumed that the metrics stored in the engine state are
            already accumulated over an epoch.
        '''
        idx = len(self._output_transform)
        self._metric_value_dict.append(OrderedDict())
        self._item_counter.append(defaultdict(int))
        self._output_transform.append(output_transform)
        self._epoch.append(self.initial_epoch)
        engine.add_event_handler(Events.ITERATION_COMPLETED,
                                 self._iteration_completed, prefix, idx)
        engine.add_event_handler(Events.EPOCH_COMPLETED,
                                 self._epoch_completed, prefix, idx,
                                 metric_keys)
    
    def __del__(self):
        if self.summary_writer is not None:
            self.summary_writer.close()
            self.summary_writer = None


class image_logger(object):
    """
    Expects `output` as a dictionary or list of image lists.
    """
    def __init__(self, num_vis=None, output_transform=lambda x:x,
                 initial_epoch=1, directory=None, summary_tracker=None,
                 suffix=None, min_val=None, max_val=None,
                 output_name='outputs',
                 fontname="LiberationSans-Regular.ttf", fontsize=24,
                 rng=None):
        self.num_vis = num_vis
        self.directory = directory
        self.summary_tracker = summary_tracker
        self.suffix = suffix
        self.min_val = min_val
        self.max_val = max_val
        self.output_name = output_name
        self.fontname = fontname
        self.fontsize = fontsize
        self.rng = rng if rng else np.random.RandomState()
        self._output_transform = output_transform
        self._epoch = initial_epoch
    
    def reset(self):
        self._labels = None
        self._images = []
        self._num_seen = 0
        self._num_collected = 0
    
    def _collect(self, engine):
        output = self._output_transform(engine.state.output)
        if len(output)==0:
            return
        if isinstance(output, dict):
            labels, images = zip(*output.items())
        else:
            labels, images = None, output
        num_images = len(images[0])
        if num_images==0:
            return
        for stack in images:
            assert len(stack)==num_images
        self._num_seen += num_images
        
        # Collect up to `num_vis` images. In order to get a random sample
        # of `num_vis` images from across all batches collected through
        # `_collect`, a fraction of the stored images is replaced with a new
        # sample with each call to `_collect`. For the n'th call, 
        # b_n/(b_0+...+b_n) of the images are re-sampled, where b_i is the
        # number of images in batch i.
        #
        # Init.
        if len(self._images)==0:
            self._images = [[] for _ in images]
        self._labels = labels
        # If collected less than `num_vis`, fill up.
        num_sample = self.num_vis-self._num_collected
        num_sample = min(num_sample, num_images)
        if self._num_collected < self.num_vis:
            indices_sample = self.rng.choice(num_images, size=num_sample,
                                             replace=False)
            for i, stack in enumerate(images):
                self._images[i].extend(stack[indices_sample])
            self._num_collected += num_sample
        # If collected more than `num_vis`, resample.
        num_resample = ( self.num_vis*(num_images-num_sample)
                        /float(self._num_seen))
        num_resample = max(num_resample, 0)
        round_up = self.rng.rand() < num_resample-int(num_resample)
        num_resample = int(num_resample)+int(round_up) # Probabilistic round.
        if self._num_collected >= self.num_vis:
            indices_resample = self.rng.choice(num_images, size=num_resample,
                                               replace=False)
            indices_drop     = self.rng.choice(self.num_vis, size=num_resample,
                                               replace=False)
            for i, stack in enumerate(images):
                for j, k in zip(indices_drop, indices_resample):
                    self._images[i][j] = stack[k]
    
    def _process(self):
        # Select a subset of images.
        if self.num_vis is not None:
            images = [stack[:self.num_vis] for stack in self._images]
        else:
            images = self._images
        
        # Digitize all images.
        images_digitized = []
        for k, image_stack in enumerate(images):
            image_stack_digitized = np.zeros_like(image_stack, dtype=np.uint8)
            for i, im in enumerate(image_stack):
                a = im.min() if self.min_val is None else self.min_val
                b = im.max() if self.max_val is None else self.max_val
                step = (b-a)/255. or 1./255
                bins = np.arange(a, a+255*step, step)
                im_d = np.digitize(im, bins).astype(np.uint8)
                image_stack_digitized[i] = im_d
            images_digitized.append(image_stack_digitized)
        
        # Make rows.
        image_rows = []
        for i, row in enumerate(images_digitized):
            arr = np.concatenate(row, axis=1)
            if self._labels is not None:
                arr_pil = Image.fromarray(arr, mode='L')
                draw = ImageDraw.Draw(arr_pil)
                font = None
                if self.fontname is not None:
                    try:
                        font = ImageFont.truetype(self.fontname, self.fontsize)
                    except OSError:
                        font = ImageFont.load_default()
                draw.text((0, 0), self._labels[i], fill=255, font=font)
                arr = np.array(arr_pil)
            image_rows.append(arr)
        
        # Join rows.
        final_image = np.concatenate(image_rows, axis=0)
        
        # Log to tensorboard.
        if self.summary_tracker is not None:
            self.summary_tracker.summary_writer.add_image(
                self.output_name,
                final_image,
                global_step=self._epoch)
            self.summary_tracker.summary_writer.file_writer.flush()
        
        # Log to file.
        if self.directory is not None:
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)
            _suffix = "_{}".format(self.suffix) if self.suffix else ""
            fn = str(self._epoch)+_suffix+".jpg"
            final_image_pil = Image.fromarray(final_image, mode='L')
            final_image_pil.save(os.path.join(self.directory, fn))
            
        # Update epoch count and clear memory.
        self._epoch += 1
        self.reset()
        
    def attach(self, engine):
        engine.add_event_handler(Events.EPOCH_STARTED, lambda _: self.reset())
        engine.add_event_handler(Events.ITERATION_COMPLETED, self._collect)
        engine.add_event_handler(Events.EPOCH_COMPLETED,
                                 lambda _: self._process())
