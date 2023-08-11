import torch
from ignite.engine import Events
from ignite.metrics import Metric

class metric(Metric):
    """
    Just like ignite's `Metric` except that it computes the metric every
    iteration (not only every epoch) and stores the result in the engine's
    state.
    """
    def __init__(self, *args, **kwargs):
        super(metric, self).__init__(*args, **kwargs)
        self._cached_compute = None
        
    def iteration_completed(self, engine, name):
        output = self._output_transform(engine.state.output)
        self.update(output)
        self._cached_compute = self.compute()
        engine.state.metrics[name] = self._cached_compute
        
    def completed(self, engine, name):
        engine.state.metrics[name] = self._cached_compute
        
    def attach(self, engine, name):
        engine.add_event_handler(Events.EPOCH_STARTED, self.started)
        engine.add_event_handler(Events.ITERATION_COMPLETED,
                                 self.iteration_completed, name)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.completed, name)


class dice_global(metric):
    '''
    Global Dice metric. Accumulates counts over the course of an epoch.
    
    target_class : integer or list (merge these classes in the target)
    prediction_index : integer or list (merge these channels/classes in the
        prediction)
    mask_class : integer or list (mask these classes out in the target)
    '''
    def __init__(self, target_class, prediction_index=0, mask_class=None,
                 output_transform=lambda x: x):
        super(dice_global, self).__init__(output_transform)
        if not hasattr(target_class, '__len__'):
            self.target_class = [target_class]
        else:
            self.target_class = target_class
        if not hasattr(prediction_index, '__len__'):
            self.prediction_index = [prediction_index]
        else:
            self.prediction_index = prediction_index
        self.mask_class = mask_class
        self._smooth = 1
            
    def update(self, output):
        '''
        Expects integer or one-hot class labeling in y_true.
        Expects outputs in range [0, 1] in y_pred.
        
        Computes the soft dice loss considering all classes in target_class as
        one aggregate target class and ignoring all elements with ground truth
        classes in mask_class.
        '''
        
        # Get outputs (and cast to float to avoid buffer overun with AMP).
        y_pred, y_true = output
        if y_true is None or len(y_true)==0 or y_pred is None:
            return
        assert len(y_pred)==len(y_true)
        y_pred, y_true = y_pred.float(), y_true.float()
        y_pred = sum([y_pred[:,i:i+1] for i in self.prediction_index])
        
        # Targer variable must not require a gradient.
        assert(not y_true.requires_grad)
    
        # If needed, change ground truth from categorical to integer format.
        if y_true.ndimension() > y_pred.ndimension():
            y_true = torch.max(y_true, dim=1)[1]   # argmax
            
        # Flatten all inputs.
        y_true_f = y_true.contiguous().view(-1).int()
        y_pred_f = y_pred.view(-1)
        
        # Aggregate target classes, mask out classes in mask_class.
        y_target = sum([y_true_f==t for t in self.target_class]).float()
        if self.mask_class is not None:
            mask_out = sum([y_true_f==t for t in self.mask_class])
            idxs = (mask_out==0).nonzero()
            y_target = y_target[idxs]
            y_pred_f = y_pred_f[idxs]
        
        # Accumulate dice counts.
        self._intersection += torch.sum(y_target * y_pred_f)
        self._y_target_sum += torch.sum(y_target)
        self._y_pred_sum   += torch.sum(y_pred_f)
        
    def compute(self):
        dice_val = -(2.*self._intersection+self._smooth) / \
                    (self._y_target_sum+self._y_pred_sum+self._smooth)
        return dice_val
    
    def reset(self):
        self._intersection = 0.
        self._y_target_sum = 0.
        self._y_pred_sum   = 0.


class dice_per_input(metric):
    '''
    Dice computed per input. Averaged over all inputs over an epoch.
    
    target_class : integer or list (merge these classes in the target)
    prediction_index : integer or list (merge these channels/classes in the
        prediction)
    mask_class : integer or list (mask these classes out in the target)
    '''
    def __init__(self, target_class, prediction_index=0, mask_class=None,
                 output_transform=lambda x: x):
        super(dice_per_input, self).__init__(output_transform)
        if not hasattr(target_class, '__len__'):
            self.target_class = [target_class]
        else:
            self.target_class = target_class
        if not hasattr(prediction_index, '__len__'):
            self.prediction_index = [prediction_index]
        else:
            self.prediction_index = prediction_index
        self.mask_class = mask_class
        self._smooth = 1
            
    def update(self, output):
        '''
        Expects integer or one-hot class labeling in y_true.
        Expects outputs in range [0, 1] in y_pred.
        
        Computes the soft dice loss considering all classes in target_class as
        one aggregate target class and ignoring all elements with ground truth
        classes in mask_class.
        '''
        
        # Get outputs (and cast to float to avoid buffer overun with AMP).
        y_pred, y_true = output
        if y_true is None or len(y_true)==0 or y_pred is None:
            return
        assert len(y_pred)==len(y_true)
        y_pred, y_true = y_pred.float(), y_true.float()
        y_pred = sum([y_pred[:,i:i+1] for i in self.prediction_index])
        
        # Targer variable must not require a gradient.
        assert(not y_true.requires_grad)
    
        # If needed, change ground truth from categorical to integer format.
        if y_true.ndimension() > y_pred.ndimension():
            y_true = torch.max(y_true, dim=1)[1]   # argmax
            
        # Flatten all inputs but keep the batch dimension.
        assert len(y_true)==len(y_pred)
        b = len(y_true)
        y_true_f = y_true.view(b, -1).int()
        y_pred_f = y_pred.view(b, -1)
        
        # Aggregate target classes, mask out classes in mask_class.
        y_target = sum([y_true_f==t for t in self.target_class]).float()
        if self.mask_class is not None:
            mask_out = sum([y_true_f==t for t in self.mask_class])
            idxs = (mask_out==0).nonzero()
            y_target = y_target[idxs]
            y_pred_f = y_pred_f[idxs]
        
        # Compute dice per input.
        intersection = torch.sum(y_target * y_pred_f, dim=1)
        y_target_sum = torch.sum(y_target, dim=1)
        y_pred_sum   = torch.sum(y_pred_f, dim=1)
        dice = -(2.*intersection+self._smooth) / \
                (y_target_sum+y_pred_sum+self._smooth)
        self._dice  += torch.sum(dice)
        self._count += b
        
    def compute(self):
        average_dice = 0
        if self._count:
            average_dice = self._dice/self._count
        return average_dice
    
    def reset(self):
        self._dice = 0
        self._count = 0
    
    
class batchwise_loss_accumulator(metric):
    """
    Accumulates a loss batchwise, weighted by the size of each batch.
    The batch size is determined as the length of the loss input.
    
    output_transform : function that isolates the loss from the engine output.
    skip_zero : if True, do not count loss when it is equal to zero (average
        only over non-zero losses)
    """
    def __init__(self, output_transform=lambda x: x, skip_zero=False):
        self.skip_zero = skip_zero
        super(batchwise_loss_accumulator, self).__init__(output_transform)
    
    def update(self, loss):
        if loss is None:
            return
        if isinstance(loss, torch.Tensor) and loss.dim():
            self._count += len(loss)
            self._total += loss.mean()*len(loss)
        else:
            if loss!=0 or not self.skip_zero:
                self._count += 1
                self._total += loss
        
    def compute(self):
        return self._total/max(1., float(self._count))
    
    def reset(self):
        self._count = 0
        self._total = 0
