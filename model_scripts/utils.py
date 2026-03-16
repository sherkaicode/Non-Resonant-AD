import logging
log = logging.getLogger("run")
class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=20, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        
        if self.best_loss == None:
            self.best_loss = val_loss
        
        elif self.best_loss - val_loss > self.min_delta:
            log.debug(f"Early stopping couter reset: best loss {self.best_loss:.3f} => val loss {val_loss:.3f}.")
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
            
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            log.debug(f"Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                log.debug('Early stopping')
                self.early_stop = True
def equalize_weights(y_train, y_val, weights_train, weights_val):

    class_weights_train = (weights_train[y_train == 0].sum(), weights_train[y_train == 1].sum())

    for i in range(len(class_weights_train)):
        weights_train[y_train == i] *= (
            max(class_weights_train) / class_weights_train[i]
        )  # equalize number of background and signal event

        weights_val[y_val == i] *= (
            max(class_weights_train) / class_weights_train[i]
        )  # likewise for validation set

    log.debug(f"class_weights_train for (bkg, data): {class_weights_train}")
    log.debug(f"Equalized total weights_train (bkg, data): {weights_train[y_train == 0].sum()}, {weights_train[y_train == 1].sum()}")
    log.debug(f"Equalized total weights_val (bkg, data): {weights_val[y_val == 0].sum()}, {weights_val[y_val == 1].sum()}")

    return weights_train, weights_val