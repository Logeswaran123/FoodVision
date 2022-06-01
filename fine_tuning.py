"""
FoodVision101

Fine tuning the model saved from food_vision.py
"""
import argparse
import math
import tensorflow as tf

from preprocess import preprocess
from utils import create_tensorboard_callback

SEED=42
INITIAL_LR = 0.0001 # fine tuning lr should be 10x lower than feature extraction lr
tf.random.set_seed(SEED)

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-tr",  "--train",  required=True, help="Path to train dataset directory", type=str)
ap.add_argument("-te", "--test", required=True, help="Path to test dataset directory", type=str)
ap.add_argument("-m", "--saved_model", required=True,
                help="Path to saved feature extraction model", type=str)
ap.add_argument("-s", "--save", required=True,
                help="Path to save checkpoints, tensorboard event logs, model", type=str)
ap.add_argument("-name", "--model_name", required=True,
                help="Name of the fine tuned model to be saved", type=str)
ap.add_argument("-e", "--epoch", required=True,
                help="Number of epochs to train", default=100, type=int)
args = vars(ap.parse_args())

def lr_step_decay(epoch, lr):
    """
    Learning rate decay function
    """
    drop_rate = 0.5
    epochs_drop = 10.0
    return INITIAL_LR * math.pow(drop_rate, math.floor(epoch/epochs_drop))

def main():
    # Load model
    saved_model_path = args["saved_model"]
    cnn_model = tf.keras.models.load_model(saved_model_path)
    cnn_model.summary()

    train_dir = args["train"]
    test_dir = args["test"]
    # Preprocess for model
    train_data, test_data = preprocess(train_dir=train_dir,
                                        test_dir=test_dir,
                                        label_mode="categorical",
                                        target_size=(224, 224),
                                        seed=SEED)

    # Evaluate the loaded model
    cnn_model.evaluate(test_data)

    # Since we have large dataset, set all layers as trainable (i.e. unfreeze all layers)
    for layer in cnn_model.layers:
        layer.trainable = True # set all layers to trainable
        print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy)

    # Create EarlyStopping callback to stop training
    # if model's val_loss doesn't improve for 3 epochs
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                    patience=3)

    # Create learning rate reduction callback
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                    factor=0.2,
                                                    patience=2,
                                                    verbose=1,
                                                    min_lr=1e-7)

    save_dir = args["save"]

    # Create ModelCheckpoint callback to save model's progress
    checkpoint_path = save_dir + "/model_checkpoints/FineTune/model_1.ckpt"
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                      montior="val_accuracy",
                                                      save_best_only=True,
                                                      verbose=1)

    # Create Learning rate scheduler callback to modify learning rate in model's training
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_step_decay, verbose=1)

    # Create Tensorboard callback
    tensorboard_callback = create_tensorboard_callback(save_dir + "/training_logs",
                                                    "fine_tuned")

    # Compile the model
    cnn_model.compile(loss="categorical_crossentropy",
                      optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR),
                      metrics=["accuracy"])

    # Fit the model
    cnn_model.fit(train_data,
                  epochs=args["epoch"],
                  steps_per_epoch=len(train_data),
                  validation_data=test_data,
                  validation_steps=int(0.15 * len(test_data)),
                  callbacks=[tensorboard_callback,
                            model_checkpoint,
                            lr_scheduler,
                            early_stopping,
                            reduce_lr])

    # Save the model
    cnn_model.save(save_dir + "/" + args["model_name"] + ".h5")

    # Load the saved fine_tuned model, and evaluate on test data
    saved_fine_tuned_model_path = save_dir + "/" + args["model_name"] + ".h5"
    fine_tuned_cnn_model = tf.keras.models.load_model(saved_fine_tuned_model_path)
    fine_tuned_cnn_model.summary()
    fine_tuned_cnn_model.evaluate(test_data)

if __name__ == "__main__":
    main()
