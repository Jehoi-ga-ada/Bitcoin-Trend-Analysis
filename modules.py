import matplotlib.pyplot as plt
import seaborn as sns

def train_model(model, loss_fn, optimizer, metrics, X_train, y_train, batch_size=32, epochs=10, shuffle=True, verbose=2, callbacks=None ,validation_data=None):
    model.compile(
        loss=loss_fn,
        optimizer=optimizer,
        metrics = metrics
    )
    history = model.fit(
        x=X_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        shuffle=shuffle,
        verbose=verbose,
        callbacks=callbacks
    )
    
    return history

def plot_train_val_loss(history, model_name):
    plt.figure(figsize=(5, 3))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()