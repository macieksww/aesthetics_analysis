from save_train_data import data_saver
import numpy as np

def training_summary(model, val_ds, file_paths, opt, loss_fn, metrics, epochs, batch_size):
    predictions = model.predict(val_ds, verbose=0)
    print("PREDICTIONS")
    predictions = np.concatenate((file_paths, predictions), axis=1)
    print(predictions)

    loss, accuracy = model.evaluate(val_ds, verbose=1) 
    loss = "{:.4f}".format(loss)
    accuracy = "{:.4f}".format(accuracy)
    print("LOSS")
    print(loss)
    print("ACCURACY")
    print(accuracy)
    
    training_saver = data_saver()
    training_saver.save_data("model_params", [opt, loss_fn, metrics, epochs, batch_size, loss, accuracy])


    act_fun_list = []
    for layer in model.layers:
        if "activation" in layer.get_config().keys():
            act_fun_list.append(layer.get_config()['activation'])
        else:
            act_fun_list.append("---")

    units_list = []
    for layer in model.layers:
        if "units" in layer.get_config().keys():
            units_list.append(layer.get_config()['units'])
        else:
            units_list.append("---")

    # training_saver.save_data("act_fun_params", [act_fun_list, loss, accuracy])
    # training_saver.save_data("layers_params", [units_list, loss, accuracy])



def save_model(model, filename="/Users/maciekswiech/Desktop/Praca/B-Droix/Analiza Estetyki CV/models/nn_model.h5"):
    model.save(filename)
    
def model_layers(model):
    return model.layers

def model_inputs(model):
    return model.inputs

def model_outputs(model):
    return model.layers

def model_get_weights(model):
    return model.get_weights

def plot_images(image_set):
    fig, axes = plt.subplots(1, 3, figsize=(300, 432))
    axes = axes.flatten()
    for img, ax in zip(image_set, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()